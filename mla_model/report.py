"""Metric / report containers with JSON and Markdown rendering (spec 11.4).

Every metric keeps its authored expression after binding, so a bound number is
always traceable to a formula, its bindings, scope, assumptions and evidence.
Unknown values are ``None`` — never 0.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

import sympy as sp

from .symbolic import evaluate, expr_str

SOURCE_TYPES = ("input", "derived", "conditional", "compiled", "measured", "definition")


class _Absent:
    """A metric that this variant does not have at all — distinct from a metric whose value is unknown.

    Spec 10.3 asks a comparison to report the *disappearance* of an intermediate when switching variants;
    rendering that as the same "?" used for unknown values would read as "unquantified" instead of "gone".
    """
    __slots__ = ()

    def __repr__(self):
        return "n/a"

    def __bool__(self):
        return False


ABSENT = _Absent()
ABSENT_JSON = "n/a (not present in this variant)"   # the one spelling every serialiser uses


def is_absent(v) -> bool:
    """Whether a comparison value means "this variant has no such quantity" rather than "unknown"."""
    return isinstance(v, _Absent) or v == ABSENT_JSON


@dataclass
class Metric:
    metric: str
    formula_id: str
    expression: str
    bindings: dict = field(default_factory=dict)
    value: Any = None
    unit: str = ""
    scope: str = ""
    status: str = "symbolic"                 # symbolic | partial | bound | unknown | conflict | not_applicable
    algorithm: str | None = None
    source_type: str = "derived"             # input | derived | conditional | compiled | measured | definition
    assumptions: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    coverage: str = ""                       # what the value is complete with respect to
    missing_fields: list[str] = field(default_factory=list)
    not_equivalent_to: list[str] = field(default_factory=list)
    residual: str | None = None              # expression after partial substitution (when not fully bound)
    section: str = "general"
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        # spec 2 and 11.4 require every metric to say what it is complete with respect to
        if not self.coverage:
            self.coverage = default_coverage(self.status, self.missing_fields)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["value"] = _jsonable(self.value)
        d["bindings"] = {k: _jsonable(v) for k, v in self.bindings.items()}
        return d


def default_coverage(status: str, missing_fields) -> str:
    """What a metric is complete with respect to, derived from its status and missing fields."""
    missing = list(missing_fields or [])
    if status == "bound":
        return "complete w.r.t. the declared task/strategy/numerics scenario and this formula"
    if status == "not_applicable":
        return "not applicable under the declared configuration"
    if status == "conflict":
        return "incomplete: sources disagree; both are recorded in the report's conflict list"
    if missing:
        return "incomplete: depends on undeclared/unbound " + ", ".join(missing)
    return "symbolic" if status in ("symbolic", "partial") else "incomplete: no evidence attached"


def _jsonable(v):
    if isinstance(v, _Absent):
        return ABSENT_JSON
    if isinstance(v, (sp.Integer,)):
        return int(v)
    if isinstance(v, sp.Basic):
        try:
            f = float(v)
            return int(f) if f.is_integer() else f
        except (TypeError, ValueError):
            return str(v)
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return v


# strategy symbols named by the declaration that binds them, so missing_fields stays a list of fields
SYMBOL_TO_FIELD = {"n_buf": "kv_buffers", "s_scratch": "scratch_bytes", "h_pp": "heads_per_program",
                   "R_vreg": "vreg_budget_bytes", "R_vmem": "vmem_budget_bytes"}


def field_names(symbols) -> list[str]:
    """Missing symbols spelled as the input field that declares them (a symbol with no field keeps its name)."""
    return sorted({SYMBOL_TO_FIELD.get(str(s_), str(s_)) for s_ in symbols})


def make_metric(name: str, formula_id: str, expr, bindings: dict, *, unit: str, scope: str,
                section: str, algorithm: str | None = None, source_type: str = "derived",
                assumptions=None, evidence=None, coverage: str = "", not_equivalent_to=None,
                extra_missing=None, notes=None, status_override: str | None = None) -> Metric:
    """Evaluate ``expr`` under ``bindings`` and wrap the result as a Metric."""
    if expr is None:
        return Metric(metric=name, formula_id=formula_id, expression="(undefined: missing declaration)",
                      status="unknown", unit=unit, scope=scope, section=section, algorithm=algorithm,
                      source_type=source_type, assumptions=list(assumptions or []), evidence=list(evidence or []),
                      coverage=coverage, missing_fields=field_names(extra_missing or []),
                      not_equivalent_to=list(not_equivalent_to or []), notes=list(notes or []))
    ev = evaluate(expr, bindings)
    missing = field_names(set(ev.missing) | set(extra_missing or []))
    # a value computed under an undeclared field is a scenario value: status 'partial', never 'bound'
    status = status_override or ("partial" if (extra_missing and ev.value is not None) else ev.status)
    if not coverage:
        coverage = default_coverage(status, missing)
    m = Metric(
        metric=name, formula_id=formula_id, expression=ev.expression, bindings=ev.bindings,
        value=ev.value, unit=unit, scope=scope, status=status, algorithm=algorithm, source_type=source_type,
        assumptions=list(assumptions or []), evidence=list(evidence or []), coverage=coverage,
        missing_fields=missing, not_equivalent_to=list(not_equivalent_to or []), residual=ev.residual,
        section=section, notes=list(notes or []),
    )
    m._expr = expr if isinstance(expr, sp.Basic) else None   # retained for the dependency graph (not serialized)
    return m


@dataclass
class Conflict:
    field: str
    sources: dict[str, Any]
    message: str
    severity: str = "error"

    def to_dict(self) -> dict:
        return {"field": self.field, "sources": _jsonable(self.sources), "message": self.message, "severity": self.severity}


@dataclass
class Report:
    title: str
    mode: str                                   # symbolic | bound | calibrated
    adapter: str
    algorithm: str
    task_fingerprint: str | None = None
    header: dict = field(default_factory=dict)  # bindings, config echoes, provenance
    metrics: list[Metric] = field(default_factory=list)
    constraints: list[dict] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    scenarios: list[dict] = field(default_factory=list)
    extras: dict = field(default_factory=dict)

    # -- access --------------------------------------------------------------
    def get(self, metric_name: str) -> Metric | None:
        for m in self.metrics:
            if m.metric == metric_name:
                return m
        return None

    def __getitem__(self, metric_name: str) -> Metric:
        m = self.get(metric_name)
        if m is None:
            raise KeyError(metric_name)
        return m

    def value(self, metric_name: str):
        m = self.get(metric_name)
        return None if m is None else m.value

    def sections(self) -> dict[str, list[Metric]]:
        out: dict[str, list[Metric]] = {}
        for m in self.metrics:
            out.setdefault(m.section, []).append(m)
        return out

    # -- serialisation ---------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "title": self.title, "mode": self.mode, "adapter": self.adapter, "algorithm": self.algorithm,
            "task_fingerprint": self.task_fingerprint, "header": _jsonable(self.header),
            "metrics": [m.to_dict() for m in self.metrics],
            "constraints": _jsonable(self.constraints),
            "conflicts": [c.to_dict() for c in self.conflicts],
            "unknowns": list(self.unknowns), "warnings": list(self.warnings),
            "scenarios": _jsonable(self.scenarios), "extras": _jsonable(self.extras),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        lines.append(f"- mode: **{self.mode}**  |  adapter: `{self.adapter}`  |  algorithm: `{self.algorithm}`")
        if self.task_fingerprint:
            lines.append(f"- task fingerprint: `{self.task_fingerprint}`")
        if self.header:
            lines.append("")
            lines.append("## Bindings and declarations")
            lines.append("")
            for k, v in self.header.items():
                lines.append(f"- **{k}**: `{_short(v)}`")
        if self.conflicts:
            lines += ["", "## Conflicts", ""]
            for c in self.conflicts:
                lines.append(f"- **{c.field}** ({c.severity}): {c.message}  sources: `{_short(c.sources)}`")
        if self.unknowns:
            lines += ["", "## Unresolved inputs", ""]
            for u in self.unknowns:
                lines.append(f"- {u}")
        if self.warnings:
            lines += ["", "## Warnings", ""]
            for w in self.warnings:
                lines.append(f"- {w}")
        for section, metrics in self.sections().items():
            lines += ["", f"## {section}", ""]
            lines.append("| metric | formula id | expression | value | unit | status | missing | scope | not equivalent to |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for m in metrics:
                val = "" if m.value is None else _fmt(m.value)
                expr = m.expression if m.residual is None or m.value is not None else f"{m.expression} → {m.residual}"
                scope = m.scope or "(scope not stated)"
                lines.append(f"| {m.metric} | `{m.formula_id}` | `{_md(expr)}` | {val} | {m.unit} | {m.status} | "
                             f"{', '.join(m.missing_fields)} | {_md(scope)} | {_md('; '.join(m.not_equivalent_to))} |")
            notes = [(m.metric, a) for m in metrics for a in m.assumptions]
            if notes:
                lines.append("")
                lines.append("Assumptions:")
                for name, a in notes:
                    lines.append(f"- `{name}`: {a}")
            lines.append("")
            lines.append("<details><summary>Provenance (bindings, coverage, source, evidence)</summary>")
            lines.append("")
            lines.append("| metric | source | coverage | bindings | evidence |")
            lines.append("|---|---|---|---|---|")
            for m in metrics:
                lines.append(f"| {m.metric} | {m.source_type} | {_md(m.coverage)} | `{_short(m.bindings, 90)}` | {_md('; '.join(m.evidence)) or ''} |")
            lines.append("")
            lines.append("</details>")
        if self.constraints:
            lines += ["", "## Constraints", ""]
            lines.append("| constraint | relation | holds | severity | missing |")
            lines.append("|---|---|---|---|---|")
            for c in self.constraints:
                lines.append(f"| {c['name']} | `{_md(c['relation'])}` | {c['holds']} | {c['severity']} | {', '.join(c.get('missing_fields', []))} |")
        if self.scenarios:
            lines += ["", "## Scenarios", ""]
            for s in self.scenarios:
                lines.append(f"- **{s.get('name')}**: {s.get('description', '')}")
                for k, v in s.items():
                    if k not in ("name", "description"):
                        lines.append(f"    - {k}: `{_short(v)}`")
        for k, v in self.extras.items():
            lines += ["", f"## {k}", ""]
            if isinstance(v, str):
                lines.append(v)
            else:
                lines.append("```json")
                lines.append(json.dumps(_jsonable(v), indent=2, default=str))
                lines.append("```")
        return "\n".join(lines) + "\n"


def _fmt(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}" if abs(v) >= 10000 else str(v)
    if isinstance(v, float):
        if v != v:
            return "nan"
        if abs(v) >= 1e6 or (abs(v) < 1e-3 and v != 0):
            return f"{v:.4e}"
        return f"{v:.6g}"
    return str(v)


def _md(text: str) -> str:
    return str(text).replace("|", "\\|")


def _short(v, limit: int = 160) -> str:
    text = json.dumps(_jsonable(v), default=str) if not isinstance(v, str) else v
    return text if len(text) <= limit else text[: limit - 3] + "..."
