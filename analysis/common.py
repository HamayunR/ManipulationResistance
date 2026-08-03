"""Shared run loading, condition resolution and normalisation.

This is layer 2 of the analysis stack:

1. *Native adapters* read whatever a method actually wrote to disk
   (:data:`RUN_ADAPTERS`).
2. *This module* turns that into a normalised run: a generic per-example
   result schema plus an explicit :class:`Capabilities` record.
3. *Figure scripts* consume the normalised tables only.

The point of the split is that a figure never learns what PEAR, a benchmark or
a provider is. Adding a model or a benchmark changes nothing here; adding a
competitor system means writing one adapter that maps its native output into
:data:`RESULT_COLUMNS` and declaring which capabilities it has.

Capabilities are declared, never inferred from zeros. A method that logs no
routing has ``has_routing=False``; it does not get an out-degree of 0. Missing
token usage stays ``None``; it does not become 0. Analysis that needs a
capability must skip the method and say so, which is what
``analysis/check_figure_readiness.py`` reports.

The PEAR-specific diagnostics ``analysis/check_logs.py`` and
``analysis/check_inflation_scenario.py`` deliberately do not go through here.
They read routing.jsonl directly and are free to be mechanism-specific.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import yaml

# ----------------------------------------------------------------- constants --

#: Files a PEAR run directory must contain to be analysable at all.
REQUIRED_RUN_FILES: Tuple[str, ...] = ("config.yaml", "summary.json", "results.jsonl")

#: Additionally required for any routing analysis. Its absence is a capability
#: limitation (``has_routing=False``), not a broken run.
ROUTING_FILE = "routing.jsonl"

SUMMARY_FILE = "summary.json"
RESULTS_FILE = "results.jsonl"
CONFIG_FILE = "config.yaml"

#: Raw-log schema versions this analysis code understands. Bumped in lockstep
#: with ``runner.experiment.SCHEMA_VERSION`` after the readers are updated.
#: Older runs are rejected rather than reinterpreted: in v1 the router always
#: scored ``reported_confidence``, in v2 it may score ``verified_confidence``,
#: so silently pooling them would mix two different routing inputs.
SUPPORTED_SCHEMA_VERSIONS: Tuple[int, ...] = (2,)

#: Version of the *normalised* tables written by ``collect_runs.py``. Independent
#: of the raw log schema: it changes when a normalised column changes meaning.
ANALYSIS_SCHEMA_VERSION = 1

#: Routing mechanisms the runner can log. These are different mechanisms, not
#: presentation variants, so analysis must never pool them.
ROUTING_MODES: Tuple[str, ...] = (
    "sampled",
    "enumerated",
    "identity",
    "uniform",
    "subgroup",
    "random_k_regular",
)

#: State-aware PEAR routing. Everything else is a baseline or an ablation.
PEAR_ROUTING_MODES: Tuple[str, ...] = ("sampled", "enumerated")

#: The generic per-example result schema. Every adapter must produce these keys
#: for every row; optional capability-specific keys are listed in
#: :data:`OPTIONAL_RESULT_COLUMNS` and may be ``None``.
RESULT_COLUMNS: Tuple[str, ...] = (
    "schema_version",
    "method",
    "dataset",
    "dataset_split",
    "model",
    "example_id",
    "seed",
    "prediction",
    "correct",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "mock",
    "source_run_dir",
)

OPTIONAL_RESULT_COLUMNS: Tuple[str, ...] = (
    "run_id",
    "condition",
    "perm_seed",
    "routing_mode",
    "mechanism",
    "attack_type",
    "attack_mode",
    "attack_value",
    "attacker_ids",
    "attacker_count",
    "adversarial_fraction",
    "verification_mode",
    "parse_failures",
    "calls",
    "n_messages",
    "answer_history",
    "confidence_history",
    "influence_history",
)


class RunLoadError(Exception):
    """Raised when a run directory cannot be read as a valid run."""


class StrictJsonError(RunLoadError):
    """Raised when a file is not strict, standards-compliant JSON."""


# --------------------------------------------------------------- strict JSON --
def _reject_constant(token: str) -> float:  # pragma: no cover - trivial
    raise ValueError(
        f"non-finite JSON token {token!r}: NaN/Infinity/-Infinity are not valid "
        "JSON. The writers sanitise them to null (utils.tracing.dumps_safe); a "
        "file containing them was not written by this harness or was edited."
    )


def strict_json_loads(text: str, *, where: str = "<string>") -> Any:
    """``json.loads`` that rejects the bare NaN / Infinity extensions.

    Python's parser accepts them by default, which is how an unreadable file
    reaches a dataframe and turns into a silent NaN. ``null`` is untouched:
    a logged null is a real "does not apply" and stays null.
    """
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except ValueError as exc:
        raise StrictJsonError(f"{where}: {exc}") from exc


def read_json(path: str | os.PathLike) -> Any:
    """Read one strict-JSON document."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RunLoadError(f"missing file: {path}") from exc
    if not text.strip():
        raise StrictJsonError(f"{path}: file is empty")
    return strict_json_loads(text, where=str(path))


def read_jsonl(path: str | os.PathLike) -> List[Dict[str, Any]]:
    """Read a strict-JSONL file into a list of dicts.

    Blank lines are skipped (the writers never emit them, but a hand-edited
    file often ends with one). Every other line must parse, and must be an
    object -- a bare scalar row is a malformed record, not a row.
    """
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    try:
        handle = open(path, "r", encoding="utf-8")
    except FileNotFoundError as exc:
        raise RunLoadError(f"missing file: {path}") from exc
    with handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = strict_json_loads(line, where=f"{path}:{lineno}")
            if not isinstance(row, dict):
                raise StrictJsonError(
                    f"{path}:{lineno}: expected a JSON object, got {type(row).__name__}"
                )
            rows.append(row)
    return rows


def is_finite_number(value: Any) -> bool:
    """True for a real, finite number. ``bool`` is not a number here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


# ------------------------------------------------------------- config merging --
def deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (mutating ``base``).

    Byte-for-byte the runner's ``runner.experiment._deep_merge``. Reimplemented
    rather than imported so the analysis layer does not drag in the model
    backends, the dataset loaders and langgraph just to read a config;
    ``tests/test_analysis_common.py`` pins the two implementations together.

    A shallow ``dict.update`` is wrong here: conditions override single keys of
    nested blocks (``debate.robustness.confidence_inflation.value``), and a
    shallow update would drop every sibling key of the block it replaces.
    """
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


# ------------------------------------------------------------------ discovery --
def _looks_like_run_dir(path: Path) -> bool:
    """A run directory is one the runner wrote run artefacts into.

    ``summary.json`` and ``results.jsonl`` are written per run and never to the
    experiment root, so their presence separates a run from its parent. Parent
    directories hold only run subdirectories (plus stray report text files) and
    are therefore never mistaken for runs. A run missing one of them is still
    discovered -- validation is what reports it as broken, so a half-written run
    fails loudly instead of vanishing from the analysis.
    """
    if not path.is_dir():
        return False
    return (path / SUMMARY_FILE).is_file() or (path / RESULTS_FILE).is_file()


def discover_runs(paths: Iterable[str | os.PathLike]) -> List[Path]:
    """Find every run directory at or below each supplied path.

    Recursive, order-stable and de-duplicated. Once a directory is recognised
    as a run its subdirectories are not searched: nothing nests inside a run,
    and descending would be a way to invent duplicates.
    """
    found: List[Path] = []
    seen: set[Path] = set()

    def _add(candidate: Path) -> None:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            found.append(candidate)

    for raw in paths:
        base = Path(raw)
        if not base.exists():
            raise RunLoadError(f"no such path: {base}")
        if not base.is_dir():
            raise RunLoadError(f"not a directory: {base}")
        if _looks_like_run_dir(base):
            _add(base)
            continue
        stack = [base]
        while stack:
            current = stack.pop()
            try:
                children = sorted(p for p in current.iterdir() if p.is_dir())
            except PermissionError:  # pragma: no cover - environment dependent
                continue
            for child in children:
                if child.name.startswith(".") or child.name == "__pycache__":
                    continue
                if _looks_like_run_dir(child):
                    _add(child)
                else:
                    stack.append(child)
    return sorted(found, key=lambda p: str(p))


# -------------------------------------------------------------- file loaders --
def load_run_config(run_dir: str | os.PathLike) -> Dict[str, Any]:
    """Load the run's resolved ``config.yaml``.

    The runner snapshots the config *after* resolving ``extends:``, so no chain
    resolution is needed here; conditions are still unmerged, which is what
    :func:`resolve_condition_config` does.
    """
    path = Path(run_dir) / CONFIG_FILE
    if not path.is_file():
        raise RunLoadError(f"missing file: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise RunLoadError(f"{path}: file is empty")
    try:
        cfg = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RunLoadError(f"{path}: invalid YAML: {exc}") from exc
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise RunLoadError(f"{path}: expected a mapping at the top level")
    return cfg


def load_summary(run_dir: str | os.PathLike) -> Dict[str, Any]:
    payload = read_json(Path(run_dir) / SUMMARY_FILE)
    if not isinstance(payload, dict):
        raise StrictJsonError(f"{Path(run_dir) / SUMMARY_FILE}: expected a JSON object")
    return payload


def load_results(run_dir: str | os.PathLike) -> List[Dict[str, Any]]:
    return read_jsonl(Path(run_dir) / RESULTS_FILE)


def load_routing(run_dir: str | os.PathLike) -> Optional[List[Dict[str, Any]]]:
    """Read ``routing.jsonl``; ``None`` when the run has no routing log.

    ``None`` and ``[]`` mean different things and are both preserved: the first
    is "this method does not log routing" (a capability limitation), the second
    is "it does, and logged nothing" (a broken run, reported by validation).
    """
    path = Path(run_dir) / ROUTING_FILE
    if not path.is_file():
        return None
    return read_jsonl(path)


# --------------------------------------------------------- condition resolution --
def condition_names(config: Mapping[str, Any]) -> List[str]:
    """Condition names in config order, mirroring the runner's default."""
    conditions = config.get("conditions") or [{"name": "default", "debate": {}}]
    return [str(cond.get("name", "default")) for cond in conditions]


def resolve_condition_config(
    config: Mapping[str, Any], condition_name: str
) -> Dict[str, Any]:
    """Return the effective experiment config for one condition.

    Reproduces ``runner.experiment.run_experiment``: the condition's ``debate``
    block is *deep*-merged onto the base ``debate`` block. Everything outside
    ``debate`` (dataset, agents, replication) is not per-condition and is copied
    through unchanged.
    """
    conditions = config.get("conditions") or [{"name": "default", "debate": {}}]
    for cond in conditions:
        if str(cond.get("name", "default")) != str(condition_name):
            continue
        merged_debate = deepcopy(config.get("debate", {}) or {})
        deep_merge(merged_debate, cond.get("debate", {}) or {})
        effective = deepcopy(dict(config))
        effective["debate"] = merged_debate
        effective["condition"] = str(condition_name)
        return effective
    raise RunLoadError(
        f"unknown condition {condition_name!r}; config declares "
        f"{condition_names(config)}"
    )


def _robustness_block(debate_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """The active robustness block, or ``{}``.

    Mirrors ``runner.experiment._robustness_config``: ``enabled: true`` is the
    master switch, so a fully configured attack under ``enabled: false`` is a
    no-op and must be reported as no attack.
    """
    cfg = debate_cfg.get("robustness") or {}
    if not isinstance(cfg, Mapping) or not cfg.get("enabled", False):
        return {}
    return dict(cfg)


def _confidence_inflation_block(
    robust_cfg: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Resolved ``confidence_inflation`` settings, or ``None`` when inactive.

    Mirrors ``runner.experiment._confidence_inflation_config``'s activation
    rules (nested block only; explicit ``enabled`` wins over ``type``). Unlike
    the runner this does not re-validate: the run already happened, so a config
    the runner would have rejected cannot exist in a real run directory.
    """
    if not robust_cfg:
        return None
    nested = robust_cfg.get("confidence_inflation")
    if nested is None or not isinstance(nested, Mapping):
        return None
    if "enabled" in nested:
        enabled = bool(nested.get("enabled"))
    else:
        enabled = str(robust_cfg.get("type", "")).strip().lower() in {
            "confidence_inflation",
            "all",
        }
    if not enabled:
        return None
    raw_ids = nested.get("agent_ids") or []
    agent_ids = [int(i) for i in raw_ids] if isinstance(raw_ids, (list, tuple)) else []
    return {
        "agent_ids": agent_ids,
        "mode": nested.get("mode"),
        "value": nested.get("value"),
    }


def _verification_mode(debate_cfg: Mapping[str, Any]) -> str:
    """``debate.verified_confidence.mode``, defaulting to ``"none"``.

    Mirrors ``runner.experiment._verified_confidence_config``: absent, null and
    ``mode: none`` are all the no-op. Read from ``debate``, never from
    ``debate.robustness`` -- verification is a mechanism, not an attack.
    """
    block = debate_cfg.get("verified_confidence")
    if not isinstance(block, Mapping):
        return "none"
    return str(block.get("mode", "none")).strip().lower() or "none"


@dataclass(frozen=True)
class ConditionMeta:
    """Effective, per-condition experiment metadata.

    Everything a figure may need to group or split on, resolved once from the
    merged config so no figure re-implements the runner's merge rules.
    """

    condition: str
    dataset: Optional[str]
    dataset_split: Optional[str]
    model: Optional[str]
    method: str
    mechanism: Optional[str]
    base_topology: Optional[str]
    n_agents: Optional[int]
    rounds: Optional[int]
    k_regular_degree: Optional[int]
    routing_temperature: Optional[float]
    alpha_targeted_cross: Optional[float]
    alpha_influence: Optional[float]
    alpha_low_confidence: Optional[float]
    influence_beta: Optional[float]
    verification_mode: str
    attack_type: Optional[str]
    attack_mode: Optional[str]
    attack_value: Optional[int]
    attacker_ids: Tuple[int, ...]
    attacker_count: int
    adversarial_fraction: Optional[float]
    seeds: Tuple[Any, ...]
    perm_seeds: Tuple[Any, ...]

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["attacker_ids"] = list(self.attacker_ids)
        payload["seeds"] = list(self.seeds)
        payload["perm_seeds"] = list(self.perm_seeds)
        return payload


def _opt_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _opt_int(value: Any) -> Optional[int]:
    number = _opt_number(value)
    return None if number is None else int(number)


def resolve_condition_metadata(
    config: Mapping[str, Any],
    condition_name: str,
    *,
    method: str = "pear",
    dataset_fallback: Optional[str] = None,
    split_fallback: Optional[str] = None,
) -> ConditionMeta:
    """Effective metadata for one condition of one run.

    ``attacker_count`` / ``adversarial_fraction`` describe the *confidence*
    attack, the only attack whose agent set is explicit in the config. With no
    confidence attack active they are 0 and 0.0 -- a genuine clean arm, not a
    missing value. ``attack_type`` is preserved verbatim so a confidence attack
    is never pooled with a content attack under one name.
    """
    effective = resolve_condition_config(config, condition_name)
    debate = effective.get("debate", {}) or {}
    dataset_cfg = effective.get("dataset", {}) or {}
    agents_cfg = effective.get("agents", {}) or {}
    replication = effective.get("replication", {}) or {}

    robust = _robustness_block(debate)
    inflation = _confidence_inflation_block(robust)
    n_agents = _opt_int(debate.get("n_agents"))

    if inflation:
        attacker_ids = tuple(inflation["agent_ids"])
        attacker_count = len(attacker_ids)
        attack_mode = inflation.get("mode")
        attack_value = _opt_int(inflation.get("value"))
    else:
        attacker_ids = ()
        attacker_count = 0
        attack_mode = None
        attack_value = None

    if n_agents:
        adversarial_fraction = attacker_count / float(n_agents)
    else:
        adversarial_fraction = None

    attack_type = str(robust.get("type")) if robust and robust.get("type") else None
    if robust and not attack_type:
        attack_type = "unspecified"
    if not robust:
        attack_type = "none"

    return ConditionMeta(
        condition=str(condition_name),
        dataset=dataset_cfg.get("name") or dataset_fallback,
        # Usually left unset in the config so each Task can apply its own
        # default. ``split_fallback`` carries the split the run actually
        # resolved (summary.json); with neither present it stays null rather
        # than being guessed.
        dataset_split=dataset_cfg.get("split") or split_fallback,
        model=agents_cfg.get("model"),
        method=method,
        mechanism=str(debate.get("mode")) if debate.get("mode") else None,
        base_topology=str(debate.get("base_topology")) if debate.get("base_topology") else None,
        n_agents=n_agents,
        rounds=_opt_int(debate.get("rounds")),
        k_regular_degree=_opt_int(debate.get("k_regular_degree", debate.get("degree"))),
        routing_temperature=_opt_number(debate.get("routing_temperature")),
        alpha_targeted_cross=_opt_number(debate.get("alpha_targeted_cross")),
        alpha_influence=_opt_number(debate.get("alpha_influence")),
        alpha_low_confidence=_opt_number(debate.get("alpha_low_confidence")),
        influence_beta=_opt_number(debate.get("influence_beta")),
        verification_mode=_verification_mode(debate),
        attack_type=attack_type,
        attack_mode=str(attack_mode) if attack_mode else None,
        attack_value=attack_value,
        attacker_ids=attacker_ids,
        attacker_count=attacker_count,
        adversarial_fraction=adversarial_fraction,
        seeds=tuple(replication.get("seeds") or ()),
        perm_seeds=tuple(replication.get("agent_perm_seeds") or ()),
    )


# ----------------------------------------------------------------- capabilities --
@dataclass(frozen=True)
class Capabilities:
    """What a run can support, declared per run rather than guessed per figure.

    ``False`` means "this method does not produce that quantity". It never
    means zero, and a figure that needs the capability must exclude the method
    and say why.
    """

    has_accuracy: bool = False
    has_token_usage: bool = False
    has_routing: bool = False
    has_confidence_reports: bool = False
    has_influence_history: bool = False
    supports_adversarial_fraction: bool = False

    def as_dict(self) -> Dict[str, bool]:
        return asdict(self)


CAPABILITY_COLUMNS: Tuple[str, ...] = tuple(Capabilities().as_dict())


@dataclass
class NormalizedRun:
    """One run directory, mapped into the generic schema."""

    run_dir: Path
    run_id: str
    adapter: str
    method: str
    schema_version: Optional[int]
    mock: Optional[bool]
    model: Optional[str]
    dataset: Optional[str]
    dataset_split: Optional[str]
    dataset_sha256: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, ConditionMeta] = field(default_factory=dict)
    results: List[Dict[str, Any]] = field(default_factory=list)
    routing: Optional[List[Dict[str, Any]]] = None
    capabilities: Capabilities = field(default_factory=Capabilities)

    @property
    def routing_modes(self) -> List[str]:
        if not self.routing:
            return []
        return sorted({str(r.get("routing_mode")) for r in self.routing if r.get("routing_mode")})


def run_id_for(run_dir: str | os.PathLike) -> str:
    """Stable, human-recognisable id for a run directory.

    ``<parent>/<name>`` rather than the bare directory name, because run names
    are timestamps and two experiment roots can hold the same one.
    """
    path = Path(run_dir)
    parent = path.parent.name
    return f"{parent}/{path.name}" if parent else path.name


# --------------------------------------------------------------- token usage --
def _token_usage(row: Mapping[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """``(prompt, completion, total, calls)`` for one result row.

    Reads the runner's ``budget`` block, falling back to flat keys so a
    competitor adapter can supply usage without inventing a budget object.
    Missing usage stays ``None`` all the way to the CSV: a method that does not
    report tokens must be *excluded* from a cost figure, and a zero would put it
    at the free end of the x-axis instead.
    """
    budget = row.get("budget")
    source: Mapping[str, Any] = budget if isinstance(budget, Mapping) else row
    prompt = _opt_int(source.get("prompt_tokens"))
    completion = _opt_int(source.get("completion_tokens"))
    calls = _opt_int(source.get("calls"))
    if prompt is None and completion is None:
        total = _opt_int(source.get("total_tokens"))
    else:
        total = (prompt or 0) + (completion or 0)
    return prompt, completion, total, calls


# -------------------------------------------------------------------- adapters --
def detect_run_adapter(run_dir: str | os.PathLike) -> str:
    """Pick the native adapter for a run directory.

    An explicit ``adapter`` (or ``method``) key in ``summary.json`` wins, so a
    competitor can name itself. Otherwise a ``debate:`` block in the config
    means the run came from this harness's runner. The fallback is the generic
    adapter, which assumes nothing beyond the result schema.
    """
    run_dir = Path(run_dir)
    summary: Mapping[str, Any] = {}
    if (run_dir / SUMMARY_FILE).is_file():
        try:
            summary = load_summary(run_dir)
        except RunLoadError:
            summary = {}
    declared = summary.get("adapter") or summary.get("method")
    if declared and str(declared) in RUN_ADAPTERS:
        return str(declared)
    if declared:
        raise RunLoadError(
            f"{run_dir}: summary.json declares adapter {declared!r}, which is not "
            f"registered. Known adapters: {sorted(RUN_ADAPTERS)}"
        )
    try:
        config = load_run_config(run_dir)
    except RunLoadError:
        return "generic"
    return "pear" if isinstance(config.get("debate"), Mapping) else "generic"


def _common_result_row(
    *,
    row: Mapping[str, Any],
    meta: ConditionMeta,
    run: Mapping[str, Any],
) -> Dict[str, Any]:
    """Shared projection of one native result row into the generic schema."""
    prompt, completion, total, calls = _token_usage(row)
    prediction = row.get("prediction", row.get("decision"))
    correct = row.get("correct")
    return {
        "schema_version": run.get("schema_version"),
        "method": meta.method,
        "dataset": meta.dataset,
        "dataset_split": meta.dataset_split,
        "model": meta.model,
        "example_id": str(row.get("example_id")) if row.get("example_id") is not None else None,
        "seed": row.get("seed"),
        "prediction": prediction,
        "correct": None if correct is None else bool(correct),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "mock": run.get("mock"),
        "source_run_dir": str(run.get("run_dir")),
        "run_id": run.get("run_id"),
        "condition": meta.condition,
        "perm_seed": row.get("perm_seed"),
        "mechanism": meta.mechanism,
        "attack_type": meta.attack_type,
        "attack_mode": meta.attack_mode,
        "attack_value": meta.attack_value,
        "attacker_ids": list(meta.attacker_ids),
        "attacker_count": meta.attacker_count,
        "adversarial_fraction": meta.adversarial_fraction,
        "verification_mode": meta.verification_mode,
        "parse_failures": row.get("parse_failures"),
        "calls": calls,
        "n_messages": row.get("n_messages"),
        "answer_history": row.get("answer_history"),
        "confidence_history": row.get("confidence_history"),
        "influence_history": row.get("influence_history"),
    }


def load_pear_run(run_dir: str | os.PathLike) -> NormalizedRun:
    """Adapter for runs produced by ``runner.experiment``."""
    run_dir = Path(run_dir)
    config = load_run_config(run_dir)
    summary = load_summary(run_dir)
    results = load_results(run_dir)
    routing = load_routing(run_dir)

    schema_version = _opt_int(summary.get("schema_version"))
    mock = summary.get("mock")
    run_ctx = {
        "schema_version": schema_version,
        "mock": None if mock is None else bool(mock),
        "run_dir": run_dir,
        "run_id": run_id_for(run_dir),
    }

    names = condition_names(config)
    conditions = {
        name: resolve_condition_metadata(
            config,
            name,
            method="pear",
            dataset_fallback=summary.get("dataset"),
            split_fallback=summary.get("dataset_split"),
        )
        for name in names
    }
    # Fall back to the row's own condition tag for a condition the config does
    # not declare: the row is evidence the condition ran, and dropping it would
    # silently shrink the analysis. Validation reports the mismatch.
    normalized: List[Dict[str, Any]] = []
    routing_by_condition: Dict[str, List[Dict[str, Any]]] = {}
    for raw in routing or []:
        routing_by_condition.setdefault(str(raw.get("condition")), []).append(raw)

    for raw in results:
        name = str(raw.get("condition", names[0] if names else "default"))
        meta = conditions.get(name)
        if meta is None:
            meta = ConditionMeta(
                condition=name,
                dataset=(config.get("dataset") or {}).get("name"),
                dataset_split=(config.get("dataset") or {}).get("split"),
                model=(config.get("agents") or {}).get("model"),
                method="pear",
                mechanism=str(raw.get("mode")) if raw.get("mode") else None,
                base_topology=raw.get("base_topology"),
                n_agents=_opt_int(raw.get("n_agents")),
                rounds=_opt_int(raw.get("rounds")),
                k_regular_degree=_opt_int(raw.get("k_regular_degree")),
                routing_temperature=None,
                alpha_targeted_cross=None,
                alpha_influence=None,
                alpha_low_confidence=None,
                influence_beta=None,
                verification_mode="none",
                attack_type=None,
                attack_mode=None,
                attack_value=None,
                attacker_ids=(),
                attacker_count=0,
                adversarial_fraction=None,
                seeds=(),
                perm_seeds=(),
            )
            conditions[name] = meta
        record = _common_result_row(row=raw, meta=meta, run=run_ctx)
        modes = sorted(
            {
                str(r.get("routing_mode"))
                for r in routing_by_condition.get(name, [])
                if r.get("routing_mode")
            }
        )
        record["routing_mode"] = modes[0] if len(modes) == 1 else None
        record["mock"] = bool(raw.get("mock")) if raw.get("mock") is not None else run_ctx["mock"]
        normalized.append(record)

    capabilities = Capabilities(
        has_accuracy=any(r.get("correct") is not None for r in normalized),
        has_token_usage=any(r.get("total_tokens") is not None for r in normalized),
        has_routing=bool(routing),
        has_confidence_reports=any(
            isinstance(r.get("reported_confidence"), Mapping) for r in (routing or [])
        ),
        has_influence_history=any(r.get("influence_history") for r in normalized),
        # The runner logs the attacked agent set on every condition, so the
        # fraction is always derivable -- including the clean 0.0 arm.
        supports_adversarial_fraction=all(
            meta.adversarial_fraction is not None for meta in conditions.values()
        ),
    )

    return NormalizedRun(
        run_dir=run_dir,
        run_id=run_id_for(run_dir),
        adapter="pear",
        method="pear",
        schema_version=schema_version,
        mock=None if mock is None else bool(mock),
        model=summary.get("model") or (config.get("agents") or {}).get("model"),
        dataset=(config.get("dataset") or {}).get("name") or summary.get("dataset"),
        dataset_split=(config.get("dataset") or {}).get("split") or summary.get("dataset_split"),
        dataset_sha256=summary.get("dataset_sha256"),
        config=config,
        summary=summary,
        conditions=conditions,
        results=normalized,
        routing=routing,
        capabilities=capabilities,
    )


def load_generic_run(run_dir: str | os.PathLike) -> NormalizedRun:
    """Adapter for any method that writes the generic schema directly.

    The contract is deliberately small, so wrapping a competitor is a
    half-page of code:

    * ``summary.json`` -- ``schema_version``, ``mock``, ``method`` (or
      ``adapter``), ``model``, and optionally ``dataset`` / ``dataset_split``.
    * ``results.jsonl`` -- one object per example x seed, carrying at least
      ``example_id``, ``prediction`` (or ``decision``) and ``correct``. Token
      usage may be flat (``prompt_tokens`` / ``completion_tokens``) or in a
      ``budget`` block. Omit it entirely and the run simply has
      ``has_token_usage=False``.
    * ``config.yaml`` -- optional; used only for dataset/model/condition
      metadata when summary.json does not carry it.

    No routing, no confidences, no attack metadata are required. Such a run is
    a first-class citizen of the accuracy and cost figures and is excluded,
    with a stated reason, from the routing ones.
    """
    run_dir = Path(run_dir)
    summary = load_summary(run_dir)
    results = load_results(run_dir)
    try:
        config = load_run_config(run_dir)
    except RunLoadError:
        config = {}

    method = str(summary.get("method") or summary.get("adapter") or "generic")
    schema_version = _opt_int(summary.get("schema_version"))
    mock = summary.get("mock")
    dataset = summary.get("dataset") or (config.get("dataset") or {}).get("name")
    dataset_split = summary.get("dataset_split") or (config.get("dataset") or {}).get("split")
    model = summary.get("model") or (config.get("agents") or {}).get("model")

    run_ctx = {
        "schema_version": schema_version,
        "mock": None if mock is None else bool(mock),
        "run_dir": run_dir,
        "run_id": run_id_for(run_dir),
    }

    conditions: Dict[str, ConditionMeta] = {}
    normalized: List[Dict[str, Any]] = []
    for raw in results:
        name = str(raw.get("condition", "default"))
        if name not in conditions:
            n_agents = _opt_int(raw.get("n_agents"))
            attacker_ids = tuple(int(i) for i in (raw.get("attacker_ids") or []))
            attacker_count = _opt_int(raw.get("attacker_count"))
            if attacker_count is None:
                attacker_count = len(attacker_ids)
            fraction = _opt_number(raw.get("adversarial_fraction"))
            if fraction is None and n_agents:
                fraction = attacker_count / float(n_agents)
            conditions[name] = ConditionMeta(
                condition=name,
                dataset=dataset,
                dataset_split=dataset_split,
                model=model,
                method=method,
                mechanism=str(raw.get("mechanism") or raw.get("mode") or method),
                base_topology=raw.get("base_topology"),
                n_agents=n_agents,
                rounds=_opt_int(raw.get("rounds")),
                k_regular_degree=None,
                routing_temperature=None,
                alpha_targeted_cross=None,
                alpha_influence=None,
                alpha_low_confidence=None,
                influence_beta=None,
                verification_mode=str(raw.get("verification_mode") or "none"),
                attack_type=str(raw.get("attack_type")) if raw.get("attack_type") else "none",
                attack_mode=raw.get("attack_mode"),
                attack_value=_opt_int(raw.get("attack_value")),
                attacker_ids=attacker_ids,
                attacker_count=attacker_count,
                adversarial_fraction=fraction,
                seeds=(),
                perm_seeds=(),
            )
        record = _common_result_row(row=raw, meta=conditions[name], run=run_ctx)
        record["routing_mode"] = raw.get("routing_mode")
        if raw.get("mock") is not None:
            record["mock"] = bool(raw.get("mock"))
        normalized.append(record)

    capabilities = Capabilities(
        has_accuracy=any(r.get("correct") is not None for r in normalized),
        has_token_usage=any(r.get("total_tokens") is not None for r in normalized),
        has_routing=False,
        has_confidence_reports=False,
        has_influence_history=any(r.get("influence_history") for r in normalized),
        supports_adversarial_fraction=all(
            meta.adversarial_fraction is not None for meta in conditions.values()
        )
        and bool(conditions),
    )

    return NormalizedRun(
        run_dir=run_dir,
        run_id=run_id_for(run_dir),
        adapter="generic",
        method=method,
        schema_version=schema_version,
        mock=None if mock is None else bool(mock),
        model=model,
        dataset=dataset,
        dataset_split=dataset_split,
        config=config,
        summary=summary,
        conditions=conditions,
        results=normalized,
        routing=None,
        capabilities=capabilities,
    )


#: Native adapters, keyed by the name ``summary.json`` may declare. Adding a
#: competitor means adding one entry here and nothing else -- no figure script
#: knows this mapping exists.
RUN_ADAPTERS: Dict[str, Callable[[str | os.PathLike], NormalizedRun]] = {
    "pear": load_pear_run,
    "generic": load_generic_run,
}


def load_normalized_run(
    run_dir: str | os.PathLike, *, adapter: Optional[str] = None
) -> NormalizedRun:
    """Load one run through its adapter and return the normalised form."""
    run_dir = Path(run_dir)
    name = adapter or detect_run_adapter(run_dir)
    if name not in RUN_ADAPTERS:
        raise RunLoadError(
            f"unknown adapter {name!r}; known adapters: {sorted(RUN_ADAPTERS)}"
        )
    return RUN_ADAPTERS[name](run_dir)


# ------------------------------------------------------------------ routing rows --
def agent_ids_of(routing_row: Mapping[str, Any]) -> List[int]:
    """1-based agent ids covered by one routing decision."""
    n_agents = _opt_int(routing_row.get("n_agents"))
    if not n_agents:
        degrees = routing_row.get("out_degree") or []
        n_agents = len(degrees)
    return list(range(1, int(n_agents or 0) + 1))


def confidence_map(routing_row: Mapping[str, Any], field_name: str) -> Optional[Dict[int, Optional[float]]]:
    """Read one confidence view off a routing row.

    ``None`` when the view does not apply to this run (``g_i`` and
    ``verified_confidence`` are logged as null with verification off). That is
    a real distinction from a confidence of zero and is preserved.
    """
    view = routing_row.get(field_name)
    if not isinstance(view, Mapping):
        return None
    out: Dict[int, Optional[float]] = {}
    for key, value in view.items():
        try:
            agent = int(key)
        except (TypeError, ValueError):
            continue
        out[agent] = None if value is None else _opt_number(value)
    return out


def routing_confidence_field(verification_mode: Optional[str]) -> str:
    """Which confidence view the router actually scored.

    With verification off the router scores the reported value; with
    verification on it scores ``min(reported, g_i)``. Reading the wrong one is
    the single easiest way to draw a routing figure that describes a mechanism
    the run never used.
    """
    mode = str(verification_mode or "none").strip().lower()
    return "reported_confidence" if mode in {"", "none"} else "verified_confidence"


def attacker_ids_of(
    routing_row: Mapping[str, Any], meta: Optional[ConditionMeta] = None
) -> List[int]:
    """Agents acting adversarially in one routing decision.

    The union of the confidence-reporting attackers and any content adversary,
    because ``is_attacker`` on a routing row answers "was this agent under
    attacker control", while ``attack_type`` keeps the two kinds distinguishable.
    """
    ids: set[int] = set()
    for key in ("confidence_inflation_agent_ids", "adversary_ids"):
        for value in routing_row.get(key) or []:
            try:
                ids.add(int(value))
            except (TypeError, ValueError):
                continue
    if meta is not None:
        ids.update(int(i) for i in meta.attacker_ids)
    return sorted(ids)


# ---------------------------------------------------------------- provenance --
def analysis_git_commit(repo_root: Optional[Path] = None) -> Optional[str]:
    """Git commit of the *analysis code*, or ``None`` outside a checkout.

    This is not the commit that generated the experiments: the runner does not
    log its own commit, so claiming otherwise would be a fabricated provenance
    chain. Callers must label it as the analysis-code commit.
    """
    root = Path(repo_root or Path(__file__).resolve().parents[1])
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - env dependent
        return None
    commit = out.stdout.strip()
    return commit or None


def json_dumps_stable(value: Any) -> Optional[str]:
    """Serialise a nested history for a CSV cell.

    Kept as valid JSON with sorted keys so a reader can round-trip it, and so
    two identical histories compare equal as strings. ``None`` stays ``None``
    rather than becoming the string ``"null"``.
    """
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
