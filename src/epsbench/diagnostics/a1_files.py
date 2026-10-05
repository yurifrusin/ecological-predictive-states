"""Canonical file adapters for the fixed A1 cooperative lifecycle only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from epsbench.data.dataset_identity import (
    compute_content_provenance_binding,
    compute_dataset_logical_hash,
    compute_renderer_execution_provenance_hash,
    compute_source_provenance_hash,
)
from epsbench.data.loader import BeforeActionEcologicalView, DatasetLoader
from epsbench.diagnostics import a1_lifecycle as life
from epsbench.diagnostics.a1_action_contrast import cases
from epsbench.schema import (
    ArtifactRecord,
    CanonicalPairedEpisodeManifest,
    CanonicalPairedFrameRecord,
    DatasetManifest,
    Modality,
    ModalityPermissionSet,
    TransitionRecord,
)
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes


class A1CanonicalLoader(DatasetLoader):
    """Privileged control-plane metadata and one permissioned fate-map decoder.

    This is not a learner interface. Metadata validation never opens outcome maps.
    Before views still use DatasetLoader's permissioned, immutable projection.
    """

    def read_a1_metadata(self) -> tuple[DatasetManifest, TransitionRecord]:
        if not isinstance(self.permissions, ModalityPermissionSet):
            raise ValueError("typed modality permission set required")
        self._require(
            Modality.TRANSITION_RECORD,
            Modality.SCENE_FAMILY,
            Modality.PRIVILEGED_GENERATION_RECORDS,
            Modality.APPEARANCE_CONTROL,
        )
        # Refresh the actual manifest: cached declarations cannot mask file replacement.
        self._manifest = DatasetLoader(self.root, self.permissions)._manifest
        manifest = self.read_dataset_manifest()
        if manifest.schema_version != "0.1.0-dev.11" or len(manifest.episodes) != 1:
            raise ValueError("A1 requires one canonical paired episode")
        episode = manifest.episodes[0]
        if not isinstance(episode, CanonicalPairedEpisodeManifest):
            raise ValueError("canonical paired episode required")
        transition = self._transition(episode.episode_index)
        if (
            transition.episode_id != episode.episode_id
            or sha256_bytes(canonical_json_bytes(transition)) != episode.transition.logical_sha256
            or not isinstance(transition.before, CanonicalPairedFrameRecord)
            or not isinstance(transition.after, CanonicalPairedFrameRecord)
            or transition.ecological_label_sha256 != episode.ecological_label_sha256
            or transition.oriented_boundary_ownership.oriented_boundary_sha256
            != episode.oriented_boundary_sha256
            or transition.ecological_visibility_events.visibility_event_sha256
            != episode.visibility_event_sha256
        ):
            raise ValueError("canonical episode/transition binding differs")
        if (
            compute_dataset_logical_hash(manifest) != manifest.dataset_logical_sha256
            or compute_source_provenance_hash(manifest.source_provenance)
            != manifest.source_provenance_sha256
            or compute_renderer_execution_provenance_hash(manifest.renderer_provenance)
            != manifest.renderer_execution_provenance_sha256
            or compute_content_provenance_binding(
                manifest.dataset_logical_sha256,
                manifest.source_provenance_sha256,
                manifest.renderer_execution_provenance_sha256,
            )
            != manifest.content_provenance_binding_sha256
        ):
            raise ValueError("canonical dataset/provenance hash differs")
        return manifest, transition

    def read_a1_fate(self, artifact: ArtifactRecord) -> npt.NDArray[np.uint8]:
        self._require(Modality.ECOLOGICAL_VISIBILITY_EVENTS)
        self._require_topology_if_present()
        if artifact.modality != Modality.ECOLOGICAL_VISIBILITY_EVENTS:
            raise ValueError("canonical event artifact required")
        return np.asarray(self._load_npy(artifact), dtype=np.uint8)


@dataclass(frozen=True)
class A1Files:
    """Exactly eight fixed control-plane loaders; never passed to predictors."""

    study: life.Study
    loaders: tuple[A1CanonicalLoader, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "loaders", tuple(self.loaders))
        if len(self.loaders) != 8 or any(
            not isinstance(loader, A1CanonicalLoader) for loader in self.loaders
        ):
            raise ValueError("exactly eight typed canonical loaders required")

    @classmethod
    def membership(
        cls,
        source: Path,
        source_head: str,
        source_tree: str,
        addendum_digest: str,
        loaders: tuple[A1CanonicalLoader, ...],
    ) -> A1Files:
        if len(loaders) != 8:
            raise ValueError("exactly eight canonical loaders required")
        members = []
        for case, loader in zip(cases(source), loaders, strict=True):
            manifest, transition = loader.read_a1_metadata()
            if (
                manifest.config_logical_sha256 != sha256_bytes(canonical_json_bytes(case.config))
                or manifest.root_seed != case.config.seed
                or manifest.scene_family != case.config.scene_family
                or manifest.appearance_profile_id != case.config.appearance.profile_id
                or manifest.episodes[0].episode_seed != case.config.seed
            ):
                raise ValueError("fixed dataset configuration/seed/family differs")
            before = loader.read_before_action(0)
            observed = life.ObservedTargetMetadata.from_canonical_metadata(
                manifest, transition, before
            )
            members.append(
                life.Member(
                    case.ordinal,
                    case.partition,
                    life.digest(case.config.model_dump(mode="json")),
                    observed.dataset,
                    observed.episode,
                    observed.transition,
                    observed.before_digest,
                    observed.action_digest,
                    observed.fate_digest,
                    observed.provenance_digest,
                )
            )
        study = life.Study(source_head, source_tree, tuple(members), addendum_digest)
        study.validate_source_cases(source)
        return cls(study, loaders)

    def _actual(
        self, member: life.Member
    ) -> tuple[A1CanonicalLoader, life.ObservedTargetMetadata, ArtifactRecord]:
        if member not in self.study.members:
            raise ValueError("unknown or changed A1 member")
        loader = self.loaders[member.ordinal]
        manifest, transition = loader.read_a1_metadata()
        view = loader.read_before_action(0)
        observed = life.ObservedTargetMetadata.from_canonical_metadata(manifest, transition, view)
        expected = life.ObservedTargetMetadata(
            member.dataset,
            member.episode,
            member.transition,
            member.before_digest,
            member.action_digest,
            member.fate_digest,
            member.provenance_digest,
        )
        if observed != expected:
            raise ValueError("actual canonical member changed")
        return loader, observed, transition.ecological_visibility_events.before_fate.event_codes

    def before_reader(self) -> A1BeforeReader:
        return A1BeforeReader(self)

    def development_reader(self) -> A1DevelopmentReader:
        return A1DevelopmentReader(self)

    def evaluator(self, bundle: life.Bundle, journal: life.EvaluationJournal) -> A1Evaluator:
        if bundle.study != self.study:
            raise ValueError("bundle belongs to a different canonical membership")
        bundle.validate()
        journal.verify_commit(bundle)
        return A1Evaluator(self, bundle, journal)

    def _target(self, member: life.Member) -> life.Target:
        loader, observed, artifact = self._actual(member)
        target = life.Target(
            member,
            observed,
            self.study.source_head,
            self.study.source_tree,
            self.study.root,
            loader.read_a1_fate(artifact),
        )
        life.check_target(self.study, member, target)
        return target


@dataclass(frozen=True)
class A1BeforeReader:
    files: A1Files

    def read_before(self, member: life.Member) -> BeforeActionEcologicalView:
        loader, _, _ = self.files._actual(member)
        view = loader.read_before_action(0)
        life.check_view(member, view)
        return view


@dataclass(frozen=True)
class A1DevelopmentReader:
    files: A1Files

    def read_development_target(self, member: life.Member) -> life.Target:
        if member.partition != "development":
            raise ValueError("development reader denies held-out target")
        return self.files._target(member)


@dataclass(frozen=True)
class A1Evaluator:
    files: A1Files
    bundle: life.Bundle
    journal: life.EvaluationJournal

    def read_held_out_target(self, member: life.Member) -> life.Target:
        self.bundle.validate()
        self.journal.verify_commit(self.bundle)
        self.journal.deny_changed(self.bundle.seal)
        self.journal.verify_exposure(self.bundle)
        if member not in self.bundle.study.members[4:]:
            raise ValueError("held-out reader denies unknown/development target")
        return self.files._target(member)
