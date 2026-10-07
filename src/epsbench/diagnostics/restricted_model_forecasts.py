"""Separate forecast evidence archive; closed qualification archive stays read-only."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from epsbench.diagnostics.restricted_learning_contract import (
    ARCHITECTURE,
    CONDITIONS,
    ROSTER,
    Fit,
    Forecast,
    InputEvidence,
    digest,
    sha,
)
from epsbench.diagnostics.restricted_learning_membership import BOOTSTRAP_DOMAIN, decisions
from epsbench.diagnostics.restricted_learning_retention import (
    Archive,
    QualifiedTarget,
    _check_control,
    reject_reparse,
)
from epsbench.diagnostics.restricted_learning_sampling import commitment
from epsbench.diagnostics.restricted_learning_scoring import GeometryScore, Score, decide, score
from epsbench.diagnostics.restricted_model_export import ReadOnlyExport
from epsbench.diagnostics.restricted_model_resources import matched_reports, valid_measurement
from epsbench.diagnostics.restricted_model_training import ModelSource, validate_checkpoint
from epsbench.diagnostics.restricted_models import RestrictedModel
from epsbench.diagnostics.visible_forecast_contract import _json
from epsbench.utils.canonical import canonical_json_bytes


@dataclass(frozen=True)
class ForecastPlan:
    collection_head: str
    collection_tree: str
    membership: str
    seal_a: str
    qualification_archive: str
    model: ModelSource
    inputs: tuple[tuple[str, InputEvidence], ...]
    roster: tuple[str, ...]
    synthetic: bool = False
    target_roots: tuple[tuple[str, str], ...] = ()
    bootstrap_commitment: str | None = None

    def __post_init__(self) -> None:
        for root in (self.membership, self.seal_a, self.qualification_archive):
            sha(root)
        if type(self.model) is not ModelSource or any(
            len(v) != 40 or any(c not in "0123456789abcdef" for c in v)
            for v in (self.collection_head, self.collection_tree)
        ):
            raise ValueError("distinct exact collection/model identities required")
        if (
            type(self.inputs) is not tuple
            or any(
                type(pair) is not tuple or len(pair) != 2 or type(pair[1]) is not InputEvidence
                for pair in self.inputs
            )
            or len(dict(self.inputs)) != len(self.inputs)
        ):
            raise ValueError("immutable evaluator-controlled qualified inputs required")
        if not self.synthetic:
            if (
                {d for d, _ in self.target_roots} != {d for d, _ in self.inputs}
                or len(dict(self.target_roots)) != len(self.inputs)
                or self.bootstrap_commitment is None
            ):
                raise ValueError("complete evaluator-private target/endpoint commitments required")
            for _, root in self.target_roots:
                sha(root)
            sha(self.bootstrap_commitment)
        expected = tuple(f.key + "/" + d for f in ROSTER for d, _ in self.inputs)
        if self.roster != expected or (not self.synthetic and len(self.inputs) != 256):
            raise ValueError("all54 complete groups and fixed evaluation membership required")

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "version": "SYNTHETIC_SOURCE_ONLY"
                if self.synthetic
                else "restricted-model-forecast-v1",
                "architecture": ARCHITECTURE,
                "collection_head": self.collection_head,
                "collection_tree": self.collection_tree,
                "membership": self.membership,
                "seal_a": self.seal_a,
                "qualification_archive": self.qualification_archive,
                "model": self.model.__dict__,
                "inputs": {d: v.digest for d, v in self.inputs},
                "roster": self.roster,
                "targets": dict(self.target_roots),
                "bootstrap": self.bootstrap_commitment,
            }
        )


class ForecastSession:
    """Evaluator process owns this capability; no automatic prediction or target access."""

    def __init__(self, plan: ForecastPlan, archive: Archive) -> None:
        if type(plan) is not ForecastPlan or type(archive) is not Archive:
            raise PermissionError("explicit typed launch plan/new evidence archive required")
        if not plan.synthetic:
            raise PermissionError(
                "actual launch must originate from authenticated read-only export"
            )
        self._initialize(plan, archive)

    @classmethod
    def from_export(
        cls, boundary: ReadOnlyExport, model: ModelSource, forecast_root: Path
    ) -> ForecastSession:
        if type(boundary) is not ReadOnlyExport:
            raise PermissionError("authenticated exporter capability required")
        reject_reparse(forecast_root)
        candidate, qualified = forecast_root.resolve(), boundary._root.resolve()
        if (
            candidate == qualified
            or candidate in qualified.parents
            or qualified in candidate.parents
        ):
            raise PermissionError("new archive must be disjoint from closed qualification bundle")
        if not hasattr(boundary, "_materials"):
            raise PermissionError(
                "complete authenticated schedules/init export required before launch"
            )
        lock = boundary._lock
        inputs = boundary.forecast_inputs()
        report_root = boundary._accepted_archive
        plan = ForecastPlan(
            lock.source_head,
            lock.source_tree,
            lock.digest,
            boundary._seal_a,
            report_root,
            model,
            inputs,
            lock.forecast_roster,
            target_roots=tuple((d, boundary._seal["targets"][d]) for d, _ in inputs),
            bootstrap_commitment=lock.bootstrap_commitment,
        )
        session = cls.__new__(cls)
        session._initialize(plan, Archive(candidate, seconds=8 * 3600))
        if not hasattr(boundary, "_materials"):
            raise PermissionError(
                "complete authenticated schedules/init export required before launch"
            )
        session._expected = {
            f.key: {
                "data": m.data_root,
                "schedule": m.schedule_root,
                "labels": m.labels_root,
                "order": digest(canonical_json_bytes(list(m.order))),
                "initialization": dict(lock.initialization_commitments)[f.initialization],
            }
            for m in boundary._materials
            for f in ROSTER
            if f.budget == m.budget and f.condition in CONDITIONS
        }
        session._controls = boundary.control_rules()
        return session

    def _initialize(self, plan: ForecastPlan, archive: Archive) -> None:
        self._expected: dict[str, dict[str, str]] = {}
        self._controls: dict[
            tuple[str, int], tuple[str, dict[tuple[tuple[str, ...], bool], float]]
        ] = {}
        self.plan, self.archive = plan, archive
        self._seal_b: str | None = None
        self._forecasts: dict[str, Forecast] = {}
        self._measurements: dict[str, dict[str, Any]] = {}
        self._target_access = False
        archive.event("MODEL_LAUNCH", "model-launch.json", plan.canonical_bytes())

    def inspect(self) -> dict[str, Any]:
        self.archive.inspect()
        files = {}
        total = 0
        for path in self.archive.root.iterdir():
            reject_reparse(path)
            if not path.is_file() or path.stat().st_size > 4 * 1024**2:
                raise ValueError("bounded regular forecast evidence required")
            raw = path.read_bytes()
            total += len(raw)
            files[path.name] = digest(raw)
        if total > self.archive.max_bytes:
            raise RuntimeError("inclusive retained forecast payload exceeded")
        return {
            "file_root": digest(canonical_json_bytes(files)),
            "payload_bytes": total,
            "failed": self.archive.failed or (self.archive.root / "operator-failure.json").exists(),
            "seal_a": self.plan.seal_a,
            "seal_b": self._seal_b,
        }

    def retain_fit(
        self, fit: Fit, checkpoint: bytes, measurement: dict[str, Any], run: bytes
    ) -> None:
        if (
            type(fit) is not Fit
            or fit.condition not in CONDITIONS
            or fit.key in self._measurements
            or self._seal_b
        ):
            raise PermissionError("single preforecast checkpoint per24 learned fits required")
        if (
            measurement.get("checkpoint") != digest(checkpoint)
            or measurement.get("updates") != 1000
            or measurement.get("batch") != 16
        ):
            raise ValueError("actual final-fit measurement/checkpoint bindings required")
        if len(checkpoint) < 17 or checkpoint[:9] != b"EPSMODEL1":
            raise ValueError("explicit bounded checkpoint required")
        import struct

        length = struct.unpack("<Q", checkpoint[9:17])[0]
        header = _json(checkpoint[17 : 17 + length])
        if (
            header["fit"] != fit.key
            or header["source"] != self.plan.model.__dict__
            or header["seal_a"] != self.plan.seal_a
            or header["schedule"] != measurement.get("schedule")
            or header["data"] != measurement.get("data")
        ):
            raise ValueError("checkpoint collection/model/data/schedule identity differs")
        if not self.plan.synthetic:
            expected = self._expected[fit.key]
            if (
                any(header.get(k) != v or measurement.get(k) != v for k, v in expected.items())
                or measurement.get("fit") != fit.key
                or measurement.get("model_source") != self.plan.model.__dict__
            ):
                raise ValueError(
                    "independently authenticated exporter/init/source/fit bindings differ"
                )
        expected_header = {
            "architecture": ARCHITECTURE,
            "fit": fit.key,
            "source": self.plan.model.__dict__,
            "seal_a": self.plan.seal_a,
            "updates": 1000,
            "tensors": RestrictedModel.tensor_layout(fit.condition),
            **(
                self._expected[fit.key]
                if not self.plan.synthetic
                else {
                    k: header[k] for k in ("data", "schedule", "labels", "order", "initialization")
                }
            ),
        }
        validate_checkpoint(checkpoint, expected_header, fit.condition)
        i = next(i for i, f in enumerate(ROSTER) if f == fit)
        self.archive.event("CHECKPOINT", f"checkpoint-{i}.bin", checkpoint)
        self.archive.event(
            "MEASUREMENT", f"measurement-{i}.json", canonical_json_bytes(measurement)
        )
        self.archive.event("FIT_RUN", f"run-{i}.json", run)
        self._measurements[fit.key] = {**measurement, "run": digest(run)}
        if (
            not self.plan.synthetic
            and sum(r["elapsed_cpu"] for r in self._measurements.values()) > 8 * 3600
        ):
            error = RuntimeError("cumulative comparative-fit CPU planning allowance exceeded")
            self._terminal(error, tuple(self.plan.roster))
            raise error

    def seal(
        self,
        forecasts: tuple[tuple[str, Forecast], ...],
    ) -> str:
        if self._seal_b or self.archive.failed:
            raise PermissionError("terminal single SealB; no resealing")
        pending = list(self.plan.roster)
        try:
            if not self.plan.synthetic and any(
                not valid_measurement(record) for record in self._measurements.values()
            ):
                raise ValueError("all24 actual structured measurement/provenance records required")
            learned = {f.key for f in ROSTER if f.condition in CONDITIONS}
            if (
                set(self._measurements) != learned
                or type(forecasts) is not tuple
                or len(forecasts) != len(pending)
                or {key for key, _ in forecasts} != set(pending)
            ):
                raise ValueError(
                    "all24 final checkpoints and all54 complete forecast groups required"
                )
            inputs = dict(self.plan.inputs)
            roots = {}
            bindings: dict[str, tuple[str, str]] = {}
            for i, (key, forecast) in enumerate(forecasts):
                decision = key.removeprefix(forecast.fit.key + "/")
                if decision not in inputs or key != forecast.fit.key + "/" + decision:
                    raise ValueError("exact roster/input binding required")
                forecast.validate_input(inputs[decision])
                pair = (forecast.checkpoint_sha256, forecast.run_sha256)
                if forecast.fit.key in bindings and bindings[forecast.fit.key] != pair:
                    raise ValueError("checkpoint/run changed within forecast group")
                bindings[forecast.fit.key] = pair
                if forecast.fit.condition in CONDITIONS:
                    record = self._measurements[forecast.fit.key]
                    if pair != (record["checkpoint"], record["run"]):
                        raise ValueError("forecast does not bind retained measured fit")
                else:
                    if not self.plan.synthetic:
                        _check_control(forecast, inputs[decision], self._controls)
                    else:
                        visible = {h for h, _ in inputs[decision].prefix.frames[-1].masks}
                        for h, value in forecast.channels:
                            expected = {
                                "absent": 0.0,
                                "visible": 1.0,
                                "half": 0.5,
                                "persistence": float(h in visible),
                                "train-frequency": 0.5,
                            }[forecast.fit.condition]
                            if value != expected:
                                raise ValueError("fixed synthetic cheap control differs")
                roots[key] = self.archive.event(
                    "FORECAST", f"forecast-{i}.json", forecast.canonical_bytes()
                )
                self._forecasts[key] = forecast
                pending.remove(key)
            self._seal_b = self.archive.event(
                "SEAL_B",
                "seal-b.json",
                canonical_json_bytes(
                    {
                        "version": "SYNTHETIC_SOURCE_ONLY"
                        if self.plan.synthetic
                        else "restricted-model-forecast-v1",
                        "plan": digest(self.plan.canonical_bytes()),
                        "seal_a": self.plan.seal_a,
                        "roster": self.plan.roster,
                        "forecasts": roots,
                        "bindings": bindings,
                        "measurements": {
                            f: digest(canonical_json_bytes(v))
                            for f, v in self._measurements.items()
                        },
                    }
                ),
            )
            return self._seal_b
        except Exception as error:
            self._terminal(error, tuple(pending))
            raise

    def _terminal(self, error: Exception, pending: tuple[str, ...]) -> None:
        self.archive.failed = True
        try:
            self.archive.write(
                "operator-failure.json",
                canonical_json_bytes(
                    {
                        "status": "FAILED_CLOSED",
                        "pending": pending,
                        "failure_kind": type(error).__name__,
                    }
                ),
                True,
            )
        except Exception as retention_error:
            error.add_note("terminal retention unavailable: " + type(retention_error).__name__)

    def score(self, target_access: Callable[[str], bytes], bootstrap_seed: bytes) -> dict[str, Any]:
        """Only retained target bytes cross this evaluator boundary after complete SealB."""
        if self.inspect()["failed"]:
            raise PermissionError("terminal forecast archive")
        if self.plan.synthetic or not self._seal_b or self.archive.failed or self._target_access:
            raise PermissionError("actual complete SealB required; single target access")
        if commitment(BOOTSTRAP_DOMAIN, bootstrap_seed) != self.plan.bootstrap_commitment:
            raise ValueError("committed endpoint bootstrap release required")
        if (
            digest(self.archive.read("model-launch.json")) != digest(self.plan.canonical_bytes())
            or digest(self.archive.read("seal-b.json")) != self._seal_b
        ):
            raise ValueError("immutable launch/SealB changed")
        seal = _json(self.archive.read("seal-b.json"))
        for key, record in self._measurements.items():
            i = next(i for i, f in enumerate(ROSTER) if f.key == key)
            stored = self.archive.read(f"measurement-{i}.json")
            run = self.archive.read(f"run-{i}.json")
            checkpoint = self.archive.read(f"checkpoint-{i}.bin")
            without_run = {k: v for k, v in record.items() if k != "run"}
            if (
                stored != canonical_json_bytes(without_run)
                or digest(run) != record["run"]
                or digest(checkpoint) != record["checkpoint"]
                or digest(canonical_json_bytes(record)) != seal["measurements"][key]
            ):
                raise ValueError("immutable measured checkpoint/run changed")
        for i, (key, forecast) in enumerate(self._forecasts.items()):
            if (
                digest(self.archive.read(f"forecast-{i}.json")) != seal["forecasts"][key]
                or digest(forecast.canonical_bytes()) != seal["forecasts"][key]
            ):
                raise ValueError("retained full forecast bytes changed")
        matching = [
            matched_reports(
                self._measurements[f"{b}/{i}/relational"], self._measurements[f"{b}/{i}/dense"]
            )
            for b in (16, 64)
            for i in ("init-0", "init-1", "init-2")
        ]
        if any(value != "MATCHED_OBSERVED_DISPATCH_ONLY" for value in matching):
            report = {
                "disposition": "INCONCLUSIVE",
                "reason": "REQUIRED_RESOURCE_MATCHING_MISSING",
                "phase_gate_effect": "NONE",
                "target_access": False,
                "matching": matching,
            }
            self.archive.event("REPORT", "model-report.json", canonical_json_bytes(report))
            self.archive.failed = True
            return report
        self._target_access = True
        self.archive.event(
            "SCORING_ACCESS",
            "scoring-access.json",
            canonical_json_bytes({"seal_a": self.plan.seal_a, "seal_b": self._seal_b}),
        )
        try:
            inputs = dict(self.plan.inputs)
            truths = {}
            for decision, root in self.plan.target_roots:
                raw = target_access(decision)
                if type(raw) is not bytes or digest(raw) != root:
                    raise ValueError(
                        "only original SealA retained targets allowed; no regeneration"
                    )
                truths[decision] = QualifiedTarget.from_bytes(raw).truth
            groups = tuple(dict.fromkeys(d.split("/")[0] for d, _ in self.plan.inputs))
            if len(groups) != 32:
                raise ValueError("complete frozen32 geometry denominator required")
            rows: dict[str, dict[str, dict[str, Score]]] = {}
            for fit in ROSTER:
                rows[fit.key] = {}
                for geometry in groups:
                    expected = decisions(geometry)
                    scored = tuple(
                        (d, score(self._forecasts[fit.key + "/" + d], inputs[d], truths[d]))
                        for d in expected
                    )
                    rows[fit.key][geometry] = GeometryScore(geometry, fit.key, scored).aggregate(
                        expected
                    )
            report = {
                "phase_gate_effect": "NONE",
                "disposition": "INCONCLUSIVE",
                "rows": {
                    f: {
                        g: {
                            s: {
                                "denominator": v.denominator,
                                "unknown": v.unknown,
                                "known_sum": v.known_sum,
                                "coverage": v.coverage,
                                "interval": v.interval,
                            }
                            for s, v in strata.items()
                        }
                        for g, strata in geometry.items()
                    }
                    for f, geometry in rows.items()
                },
            }
            if any(
                values[s].risk is None
                for geometry in rows.values()
                for values in geometry.values()
                for s in ("current-absent", "current-visible")
            ):
                report["reason"] = "COVERAGE_OR_EMPTY_STRATUM"
            else:

                def risks(budget: int, condition: str, stratum: str) -> tuple[float, ...]:
                    result = []
                    for geometry in groups:
                        values = [
                            rows[f"{budget}/{i}/{condition}"][geometry][stratum].risk
                            for i in ("init-0", "init-1", "init-2")
                        ]
                        if any(v is None for v in values):
                            raise ValueError("complete paired risks required")
                        result.append(sum(v for v in values if v is not None) / 3)
                    return tuple(result)

                def difference(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
                    return tuple(x - y for x, y in zip(a, b, strict=True))

                low = risks(16, "relational", "current-absent")
                endpoint = decide(
                    difference(risks(16, "dense", "current-absent"), low),
                    difference(
                        risks(64, "relational", "current-absent"),
                        risks(64, "dense", "current-absent"),
                    ),
                    difference(
                        risks(64, "relational", "current-visible"),
                        risks(64, "dense", "current-visible"),
                    ),
                    low,
                    tuple(
                        difference(risks(16, c, "current-absent"), low)
                        for c in ("persistence", "absent", "visible", "half", "train-frequency")
                    ),
                    tuple(
                        difference(risks(16, c, "current-absent"), low)
                        for c in ("action-zero", "memory-reset")
                    ),
                    bootstrap_seed,
                )
                report["endpoint"] = endpoint.__dict__
                matching = [
                    matched_reports(
                        self._measurements[f"{b}/{i}/relational"],
                        self._measurements[f"{b}/{i}/dense"],
                    )
                    for b in (16, 64)
                    for i in ("init-0", "init-1", "init-2")
                ]
                report["matching"] = matching
                if all(v == "MATCHED_OBSERVED_DISPATCH_ONLY" for v in matching):
                    report["disposition"] = endpoint.disposition
                else:
                    report["reason"] = "REQUIRED_RESOURCE_MATCHING_MISSING"
            self.archive.event("REPORT", "model-report.json", canonical_json_bytes(report))
            return report
        except Exception as error:
            self._terminal(error, ())
            raise
