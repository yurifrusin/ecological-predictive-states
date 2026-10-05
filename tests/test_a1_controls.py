"""Independent synthetic A1 controls and chronology acceptance; no native fixtures."""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from epsbench.data.loader import BeforeActionEcologicalView
from epsbench.diagnostics import a1_lifecycle as life
from epsbench.diagnostics.a1_action_contrast import DevelopmentTemplate, cases
from epsbench.diagnostics.a1_controls import ControlledForecast, controlled_forecast
from epsbench.utils.canonical import logical_array_hash
from tests.test_a1_action_contrast import before

SOURCE = Path(__file__).resolve().parents[1]


def test_alignment_shift_and_original_retrieval() -> None:
    mask = np.zeros((5, 10), dtype=np.float64)
    mask[:, 2] = 1
    templates = (DevelopmentTemplate(before(offset=3), mask),)
    result = controlled_forecast(before(offset=4), templates)
    assert result.alignment.shift == 1
    assert result.alignment.median_index == 4
    assert result.alignment.pairs == 10
    assert result.alignment.hamming_distance == 20
    assert np.array_equal(result.original.template_scores, mask)
    expected = np.zeros_like(mask)
    expected[:, 3] = 1
    assert np.array_equal(result.aligned, expected)
    assert np.argmax(result.original.boundary_scores[0]) == 3
    assert np.argmax(result.wrong_action[0]) == 7
    assert result.alignment.clipped == 0


def test_sorted_pairing_even_lower_median_and_unmatched() -> None:
    own = before(offset=4)
    edges = tuple(reversed(own.boundaries[:-1]))
    changed = edges[0].model_copy(update={"column": edges[0].column + 2})
    edges = (changed, *edges[1:])
    own = BeforeActionEcologicalView(own.action, own.segmentation, own.surfaces, edges)
    mask = np.ones((5, 10), dtype=np.float64)
    result = controlled_forecast(own, (DevelopmentTemplate(before(), mask),))
    assert result.alignment.pairs == 9
    assert result.alignment.unmatched_template == 1
    assert result.alignment.unmatched_own == 0
    assert result.alignment.shift == 1
    assert result.alignment.median_index == 4
    assert result.alignment.clipped == 5
    assert np.all(result.aligned[:, 0] == 0)
    assert np.all(result.aligned[:, 1:] == 1)


def test_no_support_null_median_duplicates_fail_and_tie_order() -> None:
    own = before()
    empty = BeforeActionEcologicalView(own.action, own.segmentation, own.surfaces, ())
    mask = np.ones((5, 10), dtype=np.float64)
    templates = (DevelopmentTemplate(empty, mask), DevelopmentTemplate(empty, np.zeros_like(mask)))
    result = controlled_forecast(empty, templates)
    assert result.alignment.selected_ordinal == 0
    assert result.alignment.median_index is None
    assert result.alignment.shift == 0
    assert result.alignment.support == "NO_ALIGNMENT_SUPPORT"
    assert np.array_equal(result.aligned, mask)
    duplicate = BeforeActionEcologicalView(
        own.action, own.segmentation, own.surfaces, (*own.boundaries, own.boundaries[0])
    )
    with pytest.raises(ValueError, match="duplicate"):
        controlled_forecast(duplicate, (DevelopmentTemplate(own, mask),))


def test_nested_source_mutation_cannot_change_snapshot() -> None:
    source = before()
    view = BeforeActionEcologicalView(
        source.action, source.segmentation, source.surfaces, source.boundaries
    )
    original = life.view_identity(view)
    source.action.__dict__["delta_lateral"] = -0.7
    source.boundaries[0].__dict__["column"] = 8
    assert life.view_identity(view) == original
    with pytest.raises(ValueError):
        view.segmentation.setflags(write=True)


class SyntheticReaders:
    def __init__(self, tmp_path: Path) -> None:
        configs = cases(SOURCE)
        self.views = {}
        self.codes = {}
        members = []
        for case in configs:
            small = before(case.config.action.delta_lateral)
            image = np.full((120, 160), 17, dtype=np.int32)
            image[:, 3:6] = 42
            view = BeforeActionEcologicalView(small.action, image, small.surfaces, small.boundaries)
            codes = np.zeros((120, 160), dtype=np.uint8)
            codes[:5, 2 if small.action.delta_lateral > 0 else 6] = 1
            member = life.Member(
                case.ordinal,
                case.partition,
                life.digest(case.config.model_dump(mode="json")),
                life.DatasetIdentity(f"{case.ordinal + 1:064x}"),
                "episode-000000",
                life.digest({"transition": case.ordinal}),
                life.view_identity(view),
                life.digest(view.action.model_dump(mode="json")),
                logical_array_hash(codes),
                "a" * 64,
            )
            members.append(member)
            self.views[member] = view
            self.codes[member] = codes
        self.study = life.Study("b" * 40, "c" * 40, tuple(members), "d" * 64)
        self.journal = life.EvaluationJournal(tmp_path)
        self.reads: list[int] = []
        self.fail_after: int | None = None

    def read_before(self, member: life.Member) -> BeforeActionEcologicalView:
        return self.views[member]

    def target(self, member: life.Member) -> life.Target:
        return life.Target(
            member,
            life.ObservedTargetMetadata(
                member.dataset,
                member.episode,
                member.transition,
                member.before_digest,
                member.action_digest,
                member.fate_digest,
                member.provenance_digest,
            ),
            self.study.source_head,
            self.study.source_tree,
            self.study.root,
            self.codes[member],
        )

    def read_development_target(self, member: life.Member) -> life.Target:
        assert member.partition == "development"
        return self.target(member)

    def read_held_out_target(self, member: life.Member) -> life.Target:
        assert self.journal.exposed_seal() is not None
        self.reads.append(member.ordinal)
        if self.fail_after == len(self.reads):
            raise RuntimeError("synthetic partial release failure")
        return self.target(member)


def test_complete_bundle_before_first_target_and_negative_results(tmp_path: Path) -> None:
    reader = SyntheticReaders(tmp_path)
    bundle = life.assemble(reader.study, SOURCE, reader, reader, reader.journal)
    assert not reader.reads
    assert reader.journal.exposed_seal() is None
    result = life.evaluate(bundle, reader.journal, reader)
    assert reader.reads == [4, 5, 6, 7]
    assert result["original_primary_all_four"] is False  # parity retained
    assert result["additional_alignment_all_four"] is False
    assert result["action_specificity_all_four"] is True
    assert result["coverage"] == {"members": 8, "held_out": 4}
    assert reader.journal.exposed_seal() == bundle.seal


def test_incomplete_changed_or_wrong_member_denied_before_release(tmp_path: Path) -> None:
    reader = SyntheticReaders(tmp_path)
    bundle = life.assemble(reader.study, SOURCE, reader, reader, reader.journal)
    for changed in (
        replace(bundle, forecasts=bundle.forecasts[:-1]),
        replace(bundle, development_root="e" * 64),
        replace(bundle, forecasts=(bundle.forecasts[1], *bundle.forecasts[1:])),
    ):
        with pytest.raises(ValueError):
            life.evaluate(changed, reader.journal, reader)
    assert not reader.reads
    assert reader.journal.exposed_seal() is None
    with pytest.raises(ValueError):
        replace(reader.study, members=reader.study.members[:-1])
    with pytest.raises(ValueError):
        replace(reader.study, members=tuple(reversed(reader.study.members)))
    with pytest.raises(ValueError):
        life.check_target(
            reader.study, reader.study.members[4], reader.target(reader.study.members[5])
        )


def test_persist_failure_prevents_any_target_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = SyntheticReaders(tmp_path)
    bundle = life.assemble(reader.study, SOURCE, reader, reader, reader.journal)

    def fail(_: int) -> None:
        raise OSError("synthetic durability failure")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError):
        life.evaluate(bundle, reader.journal, reader)
    assert not reader.reads


def test_partial_exposure_blocks_reset_reseal_namespace_and_changed_attempt(tmp_path: Path) -> None:
    reader = SyntheticReaders(tmp_path)
    bundle = life.assemble(reader.study, SOURCE, reader, reader, reader.journal)
    reader.fail_after = 1
    with pytest.raises(RuntimeError):
        life.evaluate(bundle, reader.journal, reader)
    assert reader.reads == [4]
    recreated = life.EvaluationJournal(tmp_path)
    for changed in (
        reader.study,
        replace(reader.study, source_head="f" * 40),
        replace(reader.study, addendum_digest="f" * 64),
    ):
        with pytest.raises(ValueError, match="irreversible"):
            life.assemble(changed, SOURCE, reader, reader, recreated)
    reader.fail_after = None
    life.evaluate(bundle, recreated, reader)  # original unchanged attempt may finish
    assert reader.reads == [4, 4, 5, 6, 7]


def test_inconclusive_reasons_all_six_counts_and_class_absence() -> None:
    from epsbench.diagnostics.a1_action_contrast import Forecast, score_forecast

    prediction = Forecast(np.zeros((2, 3)), np.zeros((2, 3)))
    result = score_forecast(prediction, np.ones((2, 3), dtype=np.uint8))
    assert result["boundary_average_precision"] is None
    assert result["inconclusive_reasons"] == ["NO_STABLE_NEGATIVES"]
    result = score_forecast(prediction, np.full((2, 3), 5, dtype=np.uint8))
    assert result["inconclusive_reasons"] == [
        "NO_STABLE_NEGATIVES",
        "NO_DELETION_POSITIVES",
        "UNRESOLVED_OCCLUSION",
    ]
    result = score_forecast(prediction, np.arange(6, dtype=np.uint8).reshape(2, 3))
    assert result["before_fate_code_counts"] == [1] * 6
    assert result["status"] == "INCONCLUSIVE"


def test_even_distinct_shifts_choose_lower_not_average() -> None:
    template = before()
    own = before(offset=4)
    edges = tuple(
        edge.model_copy(update={"column": edge.column + (1 if edge.row >= 2 else 0)})
        for edge in own.boundaries[:8]
    )
    own = BeforeActionEcologicalView(own.action, own.segmentation, own.surfaces, edges)
    result = controlled_forecast(own, (DevelopmentTemplate(template, np.ones((5, 10))),))
    # Eight displacements: four 1s and four 2s; the lower central value is 1.
    assert result.alignment.pairs == 8
    assert result.alignment.median_index == 3
    assert result.alignment.shift == 1


def test_negative_translation_clips_without_wrap() -> None:
    mask = np.zeros((5, 10))
    mask[:, 0] = 1
    mask[:, 5] = 1
    result = controlled_forecast(before(offset=2), (DevelopmentTemplate(before(), mask),))
    expected = np.zeros_like(mask)
    expected[:, 4] = 1
    assert result.alignment.shift == -1
    assert result.alignment.clipped == 5
    assert np.array_equal(result.aligned, expected)


def test_invalid_fixed_action_rejected_even_after_unvalidated_copy() -> None:
    from epsbench.diagnostics.a1_action_contrast import boundary_risk

    view = before()
    for update in ({"name": "lateral_left"}, {"delta_forward": 0.1}, {"delta_yaw": 0.1}):
        invalid = view.action.model_copy(update=update)
        with pytest.raises(ValueError):
            boundary_risk(
                BeforeActionEcologicalView(
                    invalid, view.segmentation, view.surfaces, view.boundaries
                )
            )


def test_commit_failure_or_retained_tampering_denies_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = SyntheticReaders(tmp_path)

    def fail(_: int) -> None:
        raise OSError("commit durability failure")

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail)
        with pytest.raises(OSError):
            life.assemble(reader.study, SOURCE, reader, reader, reader.journal)
    assert not reader.reads
    assert reader.journal.exposed_seal() is None
    bundle = life.assemble(reader.study, SOURCE, reader, reader, reader.journal)
    retained = list(tmp_path.glob("*-bundle-*.json"))
    assert len(retained) == 1
    retained[0].write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="retained"):
        life.evaluate(bundle, reader.journal, reader)
    assert not reader.reads


def test_predictor_call_receives_one_own_view_and_fixed_development_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = SyntheticReaders(tmp_path)
    original = controlled_forecast
    calls: list[str] = []

    def isolated(
        own: BeforeActionEcologicalView, development: tuple[DevelopmentTemplate, ...]
    ) -> ControlledForecast:
        assert isinstance(own, BeforeActionEcologicalView)
        assert isinstance(development, tuple) and len(development) == 4
        assert all(d.before is not own for d in development)
        assert all(
            life.view_identity(d.before) == reader.study.members[i].before_digest
            for i, d in enumerate(development)
        )
        calls.append(life.view_identity(own))
        return original(own, development)

    monkeypatch.setattr(life, "controlled_forecast", isolated)
    life.assemble(reader.study, SOURCE, reader, reader, reader.journal)
    assert calls == [m.before_digest for m in reader.study.members[4:]]
    assert not reader.reads


def test_dataset_composite_identity_allows_repeated_canonical_episode(tmp_path: Path) -> None:
    reader = SyntheticReaders(tmp_path)
    assert {m.episode for m in reader.study.members} == {"episode-000000"}
    assert len({m.dataset for m in reader.study.members}) == 8
    first, second = reader.study.members[:2]
    duplicate = replace(
        second, dataset=first.dataset, transition=first.transition, fate_digest=first.fate_digest
    )
    with pytest.raises(ValueError, match="duplicate canonical"):
        replace(reader.study, members=(first, duplicate, *reader.study.members[2:]))


@pytest.mark.parametrize(
    "field",
    [
        "dataset",
        "provenance_digest",
        "action_digest",
        "before_digest",
        "episode",
        "transition",
        "fate_digest",
    ],
)
def test_observed_target_binding_substitution_fails_with_valid_pixels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    reader = SyntheticReaders(tmp_path)
    bundle = life.assemble(reader.study, SOURCE, reader, reader, reader.journal)
    member = reader.study.members[4]
    valid = reader.target(member)
    observed = valid.observed
    if field == "dataset":
        observed = replace(observed, dataset=life.DatasetIdentity("f" * 64))
    elif field == "episode":
        observed = replace(observed, episode="episode-999999")
    elif field == "transition":
        observed = replace(observed, transition="f" * 64)
    elif field == "before_digest":
        observed = replace(observed, before_digest="f" * 64)
    elif field == "action_digest":
        observed = replace(observed, action_digest="f" * 64)
    elif field == "fate_digest":
        observed = replace(observed, fate_digest="f" * 64)
    else:
        observed = replace(observed, provenance_digest="f" * 64)
    substituted = replace(valid, observed=observed)
    assert substituted.member == member
    assert np.array_equal(substituted.codes, valid.codes)
    monkeypatch.setattr(reader, "read_held_out_target", lambda requested: substituted)
    with pytest.raises(ValueError, match="observed target canonical"):
        life.evaluate(bundle, reader.journal, reader)
    assert reader.journal.exposed_seal() == bundle.seal


def test_unselected_template_duplicate_is_rejected_before_hamming_selection() -> None:
    own = before()
    distant = before(offset=4)
    duplicate = BeforeActionEcologicalView(
        distant.action,
        distant.segmentation,
        distant.surfaces,
        (*distant.boundaries, distant.boundaries[0]),
    )
    mask = np.zeros((5, 10))
    with pytest.raises(ValueError, match="duplicate contour"):
        controlled_forecast(
            own, (DevelopmentTemplate(own, mask), DevelopmentTemplate(duplicate, mask))
        )


def test_every_development_view_validated_at_assembly(tmp_path: Path) -> None:
    reader = SyntheticReaders(tmp_path)
    original = reader.study.members[3]
    view = reader.views[original]
    duplicate = BeforeActionEcologicalView(
        view.action, view.segmentation, view.surfaces, (*view.boundaries, view.boundaries[0])
    )
    replacement = replace(original, before_digest=life.view_identity(duplicate))
    reader.study = replace(
        reader.study, members=(*reader.study.members[:3], replacement, *reader.study.members[4:])
    )
    reader.views[replacement] = duplicate
    reader.codes[replacement] = reader.codes[original]
    with pytest.raises(ValueError, match="duplicate contour"):
        life.assemble(reader.study, SOURCE, reader, reader, reader.journal)
    assert not reader.reads
    assert reader.journal.exposed_seal() is None
    assert not list(tmp_path.glob("*-bundle-*.json"))


def test_same_median_distinct_displacements_are_reported_without_map_change() -> None:
    own = before(offset=4)
    altered = BeforeActionEcologicalView(
        own.action,
        own.segmentation,
        own.surfaces,
        tuple(
            edge.model_copy(update={"column": edge.column + int(edge.row >= 3)})
            for edge in own.boundaries
        ),
    )
    development = (DevelopmentTemplate(before(), np.ones((5, 10))),)
    uniform = controlled_forecast(own, development)
    mixed = controlled_forecast(altered, development)
    assert uniform.alignment.shift == mixed.alignment.shift == 1
    assert np.array_equal(uniform.aligned, mixed.aligned)
    assert uniform.alignment.minimum_displacement == uniform.alignment.maximum_displacement == 1
    assert uniform.alignment.displacements_differing_from_shift == 0
    assert mixed.alignment.minimum_displacement == 1
    assert mixed.alignment.maximum_displacement == 2
    assert mixed.alignment.displacements_differing_from_shift == 4
    empty = BeforeActionEcologicalView(own.action, own.segmentation, own.surfaces, ())
    unsupported = controlled_forecast(empty, (DevelopmentTemplate(empty, np.ones((5, 10))),))
    assert unsupported.alignment.minimum_displacement is None
    assert unsupported.alignment.maximum_displacement is None
    assert unsupported.alignment.displacements_differing_from_shift == 0


def test_canonical_record_factory_accepts_validated_metadata_and_rejects_substitution(
    tmp_path: Path,
) -> None:
    from epsbench.schema import ArtifactRecord, CanonicalPairedEpisodeManifest, Modality

    reader = SyntheticReaders(tmp_path)
    view = reader.views[reader.study.members[4]]
    neutral = ArtifactRecord(
        path="episodes/episode-000000/transition.json",
        modality=Modality.TRANSITION_RECORD,
        media_type="application/json",
        dtype="json",
        shape=(1,),
        logical_sha256="1" * 64,
        file_sha256="2" * 64,
        byte_count=1,
    )
    privileged = neutral.model_copy(
        update={
            "modality": Modality.PRIVILEGED_GENERATION_RECORDS,
            "path": "episodes/episode-000000/instrumentation.json",
        }
    )
    episode = CanonicalPairedEpisodeManifest(
        episode_id="episode-000000",
        episode_index=0,
        episode_seed=1729,
        transition=neutral,
        privileged_instrumentation=privileged,
        scene_content_sha256="3" * 64,
        ecological_label_sha256="4" * 64,
        analytic_transport_sha256="5" * 64,
        oriented_boundary_sha256="6" * 64,
        visibility_event_sha256="7" * 64,
        appearance_instance_sha256="8" * 64,
        rgb_logical_sha256=("9" * 64, "a" * 64),
        paired_output_provenance_sha256=("b" * 64, "c" * 64),
    )
    fate = ArtifactRecord(
        path="episodes/episode-000000/before_fate.npy",
        modality=Modality.ECOLOGICAL_VISIBILITY_EVENTS,
        media_type="application/x-npy",
        dtype="uint8",
        shape=(120, 160),
        logical_sha256="d" * 64,
        file_sha256="e" * 64,
        byte_count=19200,
    )
    dataset = life.DatasetIdentity("f" * 64)
    metadata = life.ObservedTargetMetadata.from_canonical_records(
        dataset,
        episode,
        view.action,
        view,
        fate,
        "0" * 64,
        "1" * 64,
    )
    assert metadata.dataset == dataset
    assert metadata.episode == "episode-000000"
    assert metadata.transition == "1" * 64
    assert metadata.before_digest == life.view_identity(view)
    assert metadata.action_digest == life.digest(view.action.model_dump(mode="json"))
    assert metadata.fate_digest == "d" * 64
    assert metadata.provenance_digest == life.digest(
        {"source": "0" * 64, "paired_endpoints": ("b" * 64, "c" * 64), "content_binding": "1" * 64}
    )
    for invalid in (
        fate.model_copy(update={"modality": Modality.DEPTH}),
        fate.model_copy(update={"dtype": "float32"}),
        fate.model_copy(update={"shape": (120, 159)}),
    ):
        with pytest.raises(ValueError, match="artifact metadata"):
            life.ObservedTargetMetadata.from_canonical_records(
                dataset,
                episode,
                view.action,
                view,
                invalid,
                "0" * 64,
                "1" * 64,
            )
    with pytest.raises(ValueError, match="actual before action"):
        life.ObservedTargetMetadata.from_canonical_records(
            dataset,
            episode,
            before(0.7).action,
            view,
            fate,
            "0" * 64,
            "1" * 64,
        )
    invalid_episode = episode.model_copy(update={"episode_id": "invented-episode"})
    with pytest.raises(ValueError):
        life.ObservedTargetMetadata.from_canonical_records(
            dataset,
            invalid_episode,
            view.action,
            view,
            fate,
            "0" * 64,
            "1" * 64,
        )
    # Paths and container-file identities are not hashed into logical provenance.
    renamed = fate.model_copy(update={"path": "other/before_fate.npy", "file_sha256": "0" * 64})
    same = life.ObservedTargetMetadata.from_canonical_records(
        dataset,
        episode,
        view.action,
        view,
        renamed,
        "0" * 64,
        "1" * 64,
    )
    assert same == metadata
