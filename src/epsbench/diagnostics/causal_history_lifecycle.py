"""Finite causal diagnostic: immutable forecasts before whole-member target exposure.

These records are private scientific attempt evidence, never review automation.
No native imports, hidden rule parameters, restored dataset copies or rule reruns.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any, Literal, cast

import numpy as np
from pydantic import Field, model_validator

from epsbench.diagnostics.a1_retention import MIB, RetainedArchive
from epsbench.diagnostics.causal_history_audit import (
    AuditHistory,
    EvaluatorTarget,
    MaskScore,
    archive,
    audit_pair,
    optical_equal,
    score,
    validate_history,
)
from epsbench.diagnostics.causal_history_core import (
    CausalState,
    Forecast,
    Motion,
    OpticalFrame,
    build_state,
    extrapolate,
    persistence,
)
from epsbench.diagnostics.causal_history_sequence import (
    HEIGHT,
    MAX_ARTIFACT_BYTES,
    MEMBERS,
    WIDTH,
    ArtifactProvider,
    Envelope,
    ExactCommand,
    Record,
    canonical,
    commands,
    digest,
    parse,
    validate_manifest,
)
from epsbench.schema import Modality, ModalityPermissionSet

Rule = Literal["persistence", "extrapolate"]
Status = Literal["MASK", "UNKNOWN", "UNKNOWN_MOTION"]
RULES: tuple[Rule, Rule] = ("persistence", "extrapolate")
ROLES = ("support_surface", "occluding_surface", "background_surface")
SEAL_PATH = "control/all-six-seal.json"
EXPOSURE_PATH = "control/whole-membership-exposure.json"
MAX_CONTROL = 256 * 1024
MAX_OUTPUTS = 2 * MIB
MAX_RESULT = MIB
MAX_JOURNAL_READS = 24


class BoundSource(Record):
    head: str = Field(pattern=r"^[0-9a-f]{40}$")
    tree: str = Field(pattern=r"^[0-9a-f]{40}$")
    image: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    renderer: dict[str, Any]


class BoundedReader:
    """Fully verify one committed artifact before admitting computational bytes."""

    def __init__(self, sink: RetainedArchive):
        self.sink = sink
        self.journal_reads = 0

    def read(self, path: str, ceiling: int = MAX_ARTIFACT_BYTES) -> bytes:
        if self.sink.poisoned or self.sink.pending is not None:
            raise ValueError("poisoned/incomplete history denies reads")
        size = self.sink.committed[path][0]
        if size > ceiling:
            raise ValueError("bounded artifact read denied")
        data = bytearray()
        for chunk in self.sink.read_chunks(path):
            if len(data) + len(chunk) > size:
                raise ValueError("changed artifact length")
            data.extend(chunk)
        if len(data) != size:
            raise ValueError("incomplete verified artifact")
        return bytes(data)

    def control(self, path: str, ceiling: int = MAX_CONTROL) -> bytes:
        if self.journal_reads >= MAX_JOURNAL_READS:
            raise ValueError("journal materialization count exhausted")
        size = self.sink.committed[path][0]
        if size > ceiling:
            raise ValueError("control materialization bound")
        self.journal_reads += 1
        # A durable nonrefundable charge before materializing scientific journals.
        self.sink.reserve_copy(f"charges/journal-{self.sink.sequence:08d}", size)
        return self.read(path, ceiling)


def put_once(sink: RetainedArchive, path: str, data: bytes) -> None:
    """Deterministic repeated serialization must match; never emit/refund twice."""
    if sink.poisoned or sink.pending is not None:
        raise ValueError("failed/incomplete sink")
    if path in sink.committed:
        if sink.committed[path][:2] != (len(data), digest(data)):
            raise ValueError("same-path content changed")
        return
    sink.put(path, data)


def _fraction(value: Fraction | None) -> tuple[int, int] | None:
    return None if value is None else (value.numerator, value.denominator)


def _rational(value: tuple[int, int] | None) -> Fraction | None:
    if value is None:
        return None
    a, b = value
    if b <= 0 or max(abs(a), b) > 2**63 - 1:
        raise ValueError("bounded positive reduced rational required")
    result = Fraction(a, b)
    if (result.numerator, result.denominator) != value:
        raise ValueError("noncanonical rational")
    return result


class MotionRecord(Record):
    sums: tuple[int, int]
    count: int = Field(ge=1, le=HEIGHT * WIDTH)
    source_index: int = Field(ge=0, le=3)
    reference: ExactCommand
    reason_counts: tuple[tuple[int, int, int], ...]

    def motion(self) -> Motion:
        if any(abs(v) > (2**31) * HEIGHT * WIDTH for v in self.sums) or any(
            not 0 <= count <= HEIGHT * WIDTH for _, _, count in self.reason_counts
        ):
            raise ValueError("motion count/sum bound")
        return Motion(
            self.sums, self.count, self.source_index, self.reference.command(), self.reason_counts
        )


class ForecastRecord(Record):
    member: str
    context: str = Field(pattern=r"^[0-9a-f]{64}$")
    rule: Literal["persistence", "extrapolate"]
    token: str = Field(pattern=r"^surface-[0-9a-f]{16}$")
    status: Literal["MASK", "UNKNOWN", "UNKNOWN_MOTION"]
    mask: str | None
    mask_sha256: str | None
    last_index: int | None = Field(default=None, ge=0, le=3)
    motion: MotionRecord | None
    cumulative: tuple[int, int] | None
    displacement: tuple[tuple[int, int], tuple[int, int]] | None
    shift: tuple[int, int] | None
    clipped_pixels: int | None = Field(default=None, ge=0, le=HEIGHT * WIDTH)

    @model_validator(mode="after")
    def complete(self) -> ForecastRecord:
        if self.member not in MEMBERS:
            raise ValueError("fixed forecast member")
        if (self.status == "MASK") != (self.mask is not None and self.mask_sha256 is not None):
            raise ValueError("MASK requires actual bytes; UNKNOWN requires absent bytes")
        if self.status != "MASK" and (self.mask is not None or self.mask_sha256 is not None):
            raise ValueError("unknown substituted mask")
        _rational(self.cumulative)
        if self.displacement is not None:
            for value in self.displacement:
                _rational(value)
        if self.motion is not None:
            self.motion.motion()
        if self.rule == "persistence" and any(
            value is not None
            for value in (
                self.last_index,
                self.motion,
                self.cumulative,
                self.displacement,
                self.shift,
                self.clipped_pixels,
            )
        ):
            raise ValueError("persistence metadata differs")
        if self.rule == "extrapolate":
            if self.last_index is None or self.cumulative is None:
                raise ValueError("extrapolation metadata incomplete")
            if self.status == "MASK" and (
                self.displacement is None or self.shift is None or self.clipped_pixels is None
            ):
                raise ValueError("mask displacement incomplete")
            if self.status != "MASK" and any(
                v is not None for v in (self.displacement, self.shift, self.clipped_pixels)
            ):
                raise ValueError("unknown displacement/score substitute")
            if self.status == "UNKNOWN_MOTION" and self.motion is not None:
                raise ValueError("unknown motion supplied")
        return self

    def forecast(self, read: Callable[[str], bytes]) -> Forecast:
        mask = None
        if self.mask is not None:
            data = read(self.mask)
            if (
                len(data) != HEIGHT * WIDTH
                or digest(data) != self.mask_sha256
                or any(v > 1 for v in data)
            ):
                raise ValueError("actual bool mask bytes/hash/shape differ")
            mask = np.frombuffer(data, dtype=np.bool_).reshape(HEIGHT, WIDTH)
        displacement = None
        if self.displacement is not None:
            x, y = (_rational(v) for v in self.displacement)
            assert x is not None and y is not None
            displacement = (x, y)
        return Forecast(
            self.token,
            self.status,
            mask,
            self.last_index,
            self.motion.motion() if self.motion else None,
            _rational(self.cumulative),
            displacement,
            self.shift,
            self.clipped_pixels,
        )


def state_digest(state: CausalState) -> str:
    return digest(
        canonical(
            {
                "version": state.version,
                "current": {
                    "index": state.current.sequence_index,
                    "segmentation": digest(state.current.segmentation.tobytes()),
                    "identities": state.current.identities,
                    "boundaries": [digest(b) for b in state.current.boundaries],
                },
                "tokens": [
                    {
                        "token": t.token,
                        "visible": t.visible,
                        "last_index": t.last_index,
                        "last_mask": digest(t.last_mask.tobytes()),
                        "motion": None
                        if t.motion is None
                        else {
                            **asdict(t.motion),
                            "reference": ExactCommand.from_command(t.motion.reference).model_dump(
                                mode="json"
                            ),
                        },
                    }
                    for t in state.tokens
                ],
                "executed": [
                    ExactCommand.from_command(c).model_dump(mode="json") for c in state.executed
                ],
                "announced": ExactCommand.from_command(state.announced).model_dump(mode="json"),
            }
        )
    )


def _provider(
    env: Envelope,
    reader: BoundedReader,
    decision: int,
    *,
    rgb: bool = False,
    privileged: bool = False,
) -> ArtifactProvider:
    modalities = [
        Modality.SURFACE_REGIONS,
        Modality.ORIENTED_BOUNDARY_OWNERSHIP,
        Modality.ANALYTIC_OPTICAL_TRANSPORT,
        Modality.EXECUTED_ACTION,
    ]
    if rgb:
        modalities.append(Modality.RGB)
    if privileged:
        modalities.append(Modality.PRIVILEGED_GENERATION_RECORDS)
    return ArtifactProvider(
        canonical(env.model_dump(mode="json")),
        tuple(a.path for a in env.artifacts),
        lambda path: reader.read(f"sequences/{env.member}/{path}"),
        ModalityPermissionSet(allowed=frozenset(modalities)),
        decision,
    )


def _roles(env: Envelope, reader: BoundedReader) -> dict[str, str]:
    evidence = _provider(env, reader, 0, privileged=True).sequence_instrumentation()
    raw = evidence.compiled["raw_geom_ids"]
    if set(raw) != set(ROLES) or len(set(raw.values())) != 3:
        raise ValueError("exact compiled semantic roles required")
    mapping = {r: t for r, _, t in evidence.mapping}
    if set(mapping) != set(raw.values()):
        raise ValueError("semantic/raw/opaque bijection differs")
    return {role: mapping[raw[role]] for role in ROLES}


def _context(env: Envelope, reader: BoundedReader) -> tuple[dict[str, Any], AuditHistory]:
    decision = MEMBERS[env.member][4]
    announced = commands(env.config, env.member)[decision]
    # Predictor gets this separately ecological-only state and nothing else.
    state = build_state(_provider(env, reader, decision).projection(), announced)
    history = archive(_provider(env, reader, decision, rgb=True).projection(), announced)
    validate_history(history)
    if state_digest(state) != state_digest(history.state):
        raise ValueError("separate ecological/audit projections differ")
    roles = _roles(env, reader)
    observed = {t.token for t in state.tokens}
    if not observed <= set(roles.values()) or roles["background_surface"] not in observed:
        raise ValueError("missing ecological observed association; no instrument repair")
    paths = [
        path for f in env.frames[: decision + 1] for path in (f.optical, f.segmentation, f.rgb)
    ]
    paths += [path for f in env.flows[:decision] for path in (f.vectors, f.validity, f.reasons)]
    specs = {a.path: a for a in env.artifacts}
    recipe = {
        "member": env.member,
        "sequence": env.logical_identity,
        "decision": decision,
        "frames": [f.identity for f in env.frames[: decision + 1]],
        "flows": [f.identity for f in env.flows[:decision]],
        "past": [specs[p].model_dump(mode="json") for p in paths],
        "announced": ExactCommand.from_command(announced).model_dump(mode="json"),
        "state": state_digest(state),
        "observed": sorted(observed),
        "mapping": env.mapping_commitment,
        "roles": roles,
    }
    if len(canonical(recipe)) > MAX_CONTROL // 6:
        raise ValueError("bounded complete past recipe")
    return recipe, history


@dataclass(frozen=True)
class SealReceipt:
    study: str
    sha256: str


@dataclass(frozen=True)
class ExposureReceipt:
    study: str
    seal_sha256: str
    sha256: str


def _envelopes(reader: BoundedReader, source: BoundSource) -> tuple[Envelope, ...]:
    result = []
    for member in MEMBERS:
        payload = reader.read(f"sequences/{member}/manifest.json", MIB)
        obj = parse(payload, MIB)
        env = validate_manifest(payload, tuple(a["path"] for a in obj["artifacts"]))
        if (
            env.member,
            env.source_head,
            env.source_tree,
            env.image_digest,
            env.renderer_identity,
        ) != (
            member,
            source.head,
            source.tree,
            source.image,
            source.renderer,
        ):
            raise ValueError("fixed membership/source/image/renderer binding differs")
        for a in env.artifacts:
            if reader.sink.committed.get(f"sequences/{member}/{a.path}", ())[:2] != (
                a.size,
                a.sha256,
            ):
                raise ValueError("missing/changed immutable sequence content")
        result.append(env)
    return tuple(result)


def seal_forecasts(source: BoundSource, reader: BoundedReader) -> SealReceipt:
    sink = reader.sink
    if SEAL_PATH in sink.used or EXPOSURE_PATH in sink.used:
        raise ValueError("no reseal or regeneration")
    contexts = []
    outputs = []
    output_bytes = 0
    try:
        envs = _envelopes(reader, source)
        for env in envs:
            recipe, history = _context(env, reader)
            context = digest(canonical(recipe))
            contexts.append(recipe)
            for rule, function in zip(RULES, (persistence, extrapolate), strict=True):
                forecasts = function(history.state)
                if tuple(f.token for f in forecasts) != tuple(recipe["observed"]):
                    raise ValueError("exact observed output coverage required")
                for f in forecasts:
                    path = f"forecasts/{env.member}/{rule}-{f.token}"
                    mask_path = None if f.mask is None else path + ".bin"
                    record = ForecastRecord(
                        member=env.member,
                        context=context,
                        rule=rule,
                        token=f.token,
                        status=cast(Status, f.status),
                        mask=mask_path,
                        mask_sha256=None if f.mask is None else digest(f.mask.tobytes()),
                        last_index=f.last_index,
                        motion=None
                        if f.motion is None
                        else MotionRecord(
                            **{
                                **asdict(f.motion),
                                "reference": ExactCommand.from_command(f.motion.reference),
                            }
                        ),
                        cumulative=_fraction(f.cumulative),
                        displacement=None
                        if f.displacement is None
                        else (
                            (f.displacement[0].numerator, f.displacement[0].denominator),
                            (f.displacement[1].numerator, f.displacement[1].denominator),
                        ),
                        shift=f.shift,
                        clipped_pixels=f.clipped_pixels,
                    )
                    data = canonical(record.model_dump(mode="json"))
                    if len(data) > 16 * 1024:
                        raise ValueError("forecast metadata bound")
                    output_bytes += len(data) + (0 if f.mask is None else f.mask.nbytes)
                    if output_bytes > MAX_OUTPUTS:
                        raise ValueError("all output ceiling")
                    if f.mask is not None and mask_path is not None:
                        put_once(sink, mask_path, f.mask.tobytes())
                    put_once(sink, path + ".json", data)
                    outputs.append({"path": path + ".json", "sha256": digest(data)})
        seal = {
            "version": "causal-history-seal-v1",
            "study": sink.study,
            "source": source.model_dump(mode="json"),
            "members": list(MEMBERS),
            "contexts": contexts,
            "outputs": outputs,
        }
        data = canonical(seal)
        if len(data) > MAX_CONTROL:
            raise ValueError("all-six seal bound")
        sink.put(SEAL_PATH, data)
        receipt = SealReceipt(sink.study, digest(data))
        _verify_seal(receipt, reader)
        return receipt
    except Exception as error:
        sink.failure(("seal_forecasts:" + type(error).__name__ + ":" + str(error))[:128])
        raise


def _verify_seal(
    receipt: SealReceipt, reader: BoundedReader
) -> tuple[
    dict[str, Any],
    tuple[Envelope, ...],
    tuple[AuditHistory, ...],
    dict[tuple[str, str, str], Forecast],
]:
    data = reader.control(SEAL_PATH)
    if (
        type(receipt) is not SealReceipt
        or receipt.study != reader.sink.study
        or digest(data) != receipt.sha256
    ):
        raise ValueError("actual immutable all-six seal required")
    seal = parse(data, MAX_CONTROL)
    if set(seal) != {"version", "study", "source", "members", "contexts", "outputs"} or (
        seal["version"] != "causal-history-seal-v1"
        or seal["study"] != receipt.study
        or seal["members"] != list(MEMBERS)
        or len(seal["contexts"]) != 6
        or not 1 <= len(seal["outputs"]) <= 36
    ):
        raise ValueError("complete fixed seal schema required")
    envs = _envelopes(reader, BoundSource.model_validate_json(canonical(seal["source"])))
    histories = []
    expected = {}
    for env, retained in zip(envs, seal["contexts"], strict=True):
        recipe, history = _context(env, reader)
        if recipe != retained:
            raise ValueError("changed complete past/state/alignment recipe")
        histories.append(history)
        for rule in RULES:
            for token in recipe["observed"]:
                expected[(env.member, rule, token)] = digest(canonical(recipe))
    decoded: dict[tuple[str, str, str], Forecast] = {}
    expected_paths = {f"forecasts/{m}/{r}-{t}.json" for m, r, t in expected}
    paths = set()
    total = 0
    for output in seal["outputs"]:
        if (
            set(output) != {"path", "sha256"}
            or output["path"] not in expected_paths
            or output["path"] in paths
        ):
            raise ValueError("extra/duplicate output")
        paths.add(output["path"])
        payload = reader.read(output["path"], 16 * 1024)
        if digest(payload) != output["sha256"]:
            raise ValueError("sealed actual forecast content changed")
        parse(payload, 16 * 1024)
        record = ForecastRecord.model_validate_json(payload)
        key = (record.member, record.rule, record.token)
        if (
            key not in expected
            or record.context != expected[key]
            or key in decoded
            or output["path"] != f"forecasts/{record.member}/{record.rule}-{record.token}.json"
        ):
            raise ValueError("forecast context/coverage/path changed")
        if record.mask is not None and record.mask != output["path"][:-5] + ".bin":
            raise ValueError("forecast mask path changed")
        decoded[key] = record.forecast(reader.read)
        total += len(payload) + (HEIGHT * WIDTH if record.mask is not None else 0)
    if set(decoded) != set(expected) or total > MAX_OUTPUTS:
        raise ValueError("missing output or output bound")
    return seal, envs, tuple(histories), decoded


def expose_targets(receipt: SealReceipt, reader: BoundedReader) -> ExposureReceipt:
    if EXPOSURE_PATH in reader.sink.used:
        raise ValueError("whole membership already exposed")
    _verify_seal(receipt, reader)
    data = canonical(
        {
            "version": "causal-history-exposure-v1",
            "study": receipt.study,
            "seal": receipt.sha256,
            "members": list(MEMBERS),
        }
    )
    reader.sink.put(EXPOSURE_PATH, data)  # flush+fsync BEFORE any target-capable provider
    return ExposureReceipt(receipt.study, receipt.sha256, digest(data))


class VerifiedTargets:
    """Factory verifies actual seal/exposure and retained history; no caller boolean."""

    def __init__(self, seal: SealReceipt, exposure: ExposureReceipt, reader: BoundedReader):
        self.seal, self.envs, self.histories, self.outputs = _verify_seal(seal, reader)
        data = reader.control(EXPOSURE_PATH)
        required = canonical(
            {
                "version": "causal-history-exposure-v1",
                "study": seal.study,
                "seal": seal.sha256,
                "members": list(MEMBERS),
            }
        )
        if type(exposure) is not ExposureReceipt or (
            exposure.study,
            exposure.seal_sha256,
            exposure.sha256,
            data,
        ) != (seal.study, seal.sha256, digest(required), required):
            raise PermissionError("actual durable exposure capability required")
        self.reader = reader

    def target(self, index: int) -> EvaluatorTarget:
        env, history = self.envs[index], self.histories[index]
        token = self.seal["contexts"][index]["roles"]["background_surface"]
        if token not in {s.token for s in history.state.tokens}:
            raise ValueError("target not ecologically observed")
        t = MEMBERS[env.member][5]
        frame = _provider(env, self.reader, t).frame(t)
        return EvaluatorTarget(t, token, frame.mask(token))


def repeat_consistency(a: ArtifactProvider, b: ArtifactProvider, alignment: dict[str, str]) -> bool:
    """Only fixed Pair1[1:4] vs Pair3[0:3], two flows; never whole states."""
    for j in range(3):
        fa, fb = a.frame(j + 1), b.frame(j)
        normalized = OpticalFrame(j, fa.segmentation, fa.identities, fa.boundaries)
        if not optical_equal(normalized, fb, alignment) or not np.array_equal(
            a.rgb(j + 1), b.rgb(j)
        ):
            return False
    for j in range(2):
        x, y = a.flow(j + 1), b.flow(j)
        if x.command != y.command or any(
            not np.array_equal(u, v)
            for u, v in (
                (x.vectors, y.vectors),
                (x.validity, y.validity),
                (x.reasons, y.reasons),
            )
        ):
            return False
    return True


def all_four(scores: tuple[MaskScore, ...], *, valid: bool, qualified: bool) -> str:
    if len(scores) != 4 or not valid or not qualified:
        return "INCONCLUSIVE"
    return "PASS" if all(s.exact is True for s in scores) else "FAIL"


def evaluate(seal: SealReceipt, exposure: ExposureReceipt, reader: BoundedReader) -> dict[str, Any]:
    capability = VerifiedTargets(seal, exposure, reader)
    targets = tuple(capability.target(i) for i in range(6))
    contexts = capability.seal["contexts"]

    def alignment(a: int, b: int) -> dict[str, str]:
        left, right = contexts[a]["roles"], contexts[b]["roles"]
        return {left[r]: right[r] for r in ROLES}

    repeats = []
    for a, b in ((0, 4), (1, 5)):
        x, y = capability.envs[a], capability.envs[b]
        repeats.append(
            repeat_consistency(
                _provider(x, reader, 3, rgb=True),
                _provider(y, reader, 2, rgb=True),
                alignment(a, b),
            )
        )
    pairs = []
    for p, (a, b) in enumerate(((0, 1), (2, 3), (4, 5)), 1):
        observed = {t.token for t in capability.histories[a].state.tokens}
        pairs.append(
            audit_pair(
                capability.histories[a],
                capability.histories[b],
                targets[a],
                targets[b],
                {k: v for k, v in alignment(a, b).items() if k in observed},
                p,
            )
        )
    cases = []
    aggregates = {}
    for rule in RULES:
        scores = []
        for i, env in enumerate(capability.envs):
            result = score(capability.outputs[(env.member, rule, targets[i].token)], targets[i])
            scores.append(result)
            cases.append(
                {
                    "member": env.member,
                    "rule": rule,
                    "status": result.status,
                    "exact": result.exact,
                    "error_pixels": result.error_pixels,
                    "iou": _fraction(result.iou),
                    "fixture_valid": pairs[i // 2].fixture_valid,
                }
            )
        aggregates[rule] = all_four(
            tuple(scores[:4]), valid=all(p.fixture_valid for p in pairs[:2]), qualified=all(repeats)
        )
    private = {
        "study": seal.study,
        "seal": seal.sha256,
        "cases": cases,
        "pairs": [asdict(p) for p in pairs],
        "repeat": repeats,
        "all_four": aggregates,
        "pair3": pairs[2].classification,
        "interpretation": (
            "INCONCLUSIVE_REPEAT_FAILURE"
            if not all(repeats)
            else "INCONCLUSIVE_FIXTURE_FAILURE"
            if not all(p.fixture_valid for p in pairs)
            else "ADMITTED"
        ),
    }
    data = canonical(private)
    if len(data) > MAX_RESULT:
        raise ValueError("bounded private results")
    reader.sink.put("results/private.json", data)
    # Aggregate receipt intentionally excludes item identifiers, hashes and hidden content.
    receipt = {
        "member_count": 6,
        "case_count": 12,
        "repeat_pass": all(repeats),
        "all_four": aggregates,
        "pair3": pairs[2].classification,
        "phase_gate_effect": "NONE",
    }
    reader.sink.put("results/aggregate.json", canonical(receipt))
    return receipt


def inspect(seal: SealReceipt, exposure: ExposureReceipt, reader: BoundedReader) -> None:
    """Fixed bounded private RGB contact sheet, using verified exposed artifacts."""
    capability = VerifiedTargets(seal, exposure, reader)
    gallery = np.zeros((HEIGHT * 2, WIDTH * 6, 3), dtype=np.uint8)
    for i, env in enumerate(capability.envs):
        provider = _provider(env, reader, MEMBERS[env.member][5], rgb=True)
        for row, t in enumerate(MEMBERS[env.member][4:6]):
            gallery[row * HEIGHT : (row + 1) * HEIGHT, i * WIDTH : (i + 1) * WIDTH] = provider.rgb(
                t
            )
    reader.sink.put("inspection/current-target-rgb.bin", gallery.tobytes())
    reader.sink.put(
        "inspection/recipe.json",
        canonical(
            {
                "width": WIDTH * 6,
                "height": HEIGHT * 2,
                "mode": "RGB",
                "bytes": HEIGHT * 2 * WIDTH * 6 * 3,
            }
        ),
    )
