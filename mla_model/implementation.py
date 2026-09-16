"""Execution strategy (kappa): tiling, residency/buffering, layout, scheduling, materialization.

Any field left ``None`` remains a symbol in the analysis (b_q, b_k, n_buf, h_pp, ...)
or produces explicitly named scenarios.  Nothing is filled from historical examples.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Mapping

from .errors import MetadataError

RECT_POLICIES = ("full_scan", "skip_future", "skip_future_subdivide")
EXTENT_POLICIES = ("logical", "padded_to_tile")
V_LOAD_POLICIES = ("early", "delayed")
MATERIALIZE_KEYS = ("expanded_kv", "q_nope", "q_tilde", "z")   # spec 6.2 intermediates this model can materialize
# resource classes a serial group may name (spec 9.2: sum the bounds of parts known to be serial)
SERIAL_RESOURCE_CLASSES = ("mxu", "vpu", "exp", "reduce", "layout",
                           "path:hbm_to_vmem", "path:vmem_to_hbm", "path:vmem_to_vreg", "path:vreg_to_vmem")
SCHEDULES = ("q_outer_kv_inner",)


@dataclass
class ExecutionStrategy:
    name: str = "unnamed"
    # tiles
    b_q: int | None = None
    b_k: int | None = None
    b_rq: int | None = None            # Q projection rank sub-tile (None -> full R_q window)
    b_rk: int | None = None            # KV projection rank sub-tile (None -> full R_k window)
    # scheduled extents (spec 1.3): the extents the kernel actually iterates over; active <= scheduled <= capacity
    scheduled_lengths: dict | None = None      # {'Sq': int, 'Sk': int} (uniform tasks only)
    # rectangle selection and execution extents (spec 4.2)
    rect_policy: str | None = None     # 'full_scan' | 'skip_future' | 'skip_future_subdivide'
    subdivide: tuple[int, int] | None = None   # (p_q, p_k) sub-tile extents for the subdivision policy
    executed_extent_policy: str | None = None  # 'logical' | 'padded_to_tile'
    # loop nest / residency (rho)
    schedule: str | None = None        # 'q_outer_kv_inner'
    kv_buffers: int | None = None      # n_buf: streamed KV window buffer count
    heads_per_program: int | None = None   # h_pp: heads sharing one program (latent / K^r reuse granularity)
    q_resident: bool | None = None     # Q operands resident across the KV loop
    projection_in_kv_loop: bool | None = None  # Q projection repeated per KV block
    v_load: str | None = None          # 'early' (V loaded with K, overlapped with the score matmul) | 'delayed' (loaded at PV)
    # aliasing of softmax temporaries
    score_alias_exp: bool | None = None       # E overwrites X
    exp_alias_p_operand: bool | None = None   # P operand is E itself (no separate cast copy)
    both_qk_branches_live: bool | None = None # Q^n K^n^T and Q^r K^r^T partial scores coexist
    # HBM materialization flags (spec 6.2)
    materialize: dict = field(default_factory=dict)  # keys: expanded_kv, q_nope, q_tilde, z
    # layout (lambda): per-object kind 'vector_tiled' | 'compact' | 'replicated'
    layout: dict = field(default_factory=dict)
    # fixed scratch and budgets
    scratch_bytes: int | None = None
    vmem_budget_bytes: int | None = None
    vreg_budget_bytes: int | None = None
    # calibration / overlap model declaration
    overlap_model: str | None = None   # 'full_overlap_max' | 'serial_stage_sum'
    serial_stage_groups: list | None = None    # groups of RESOURCE CLASSES (see SERIAL_RESOURCE_CLASSES) declared
                                               # not to overlap; not pipeline stages (see docs/SPEC_REVIEW.md D19)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        for f in ("b_q", "b_k", "b_rq", "b_rk", "kv_buffers", "heads_per_program", "scratch_bytes",
                  "vmem_budget_bytes", "vreg_budget_bytes"):
            v = getattr(self, f)
            if v is not None:
                if isinstance(v, bool) or not isinstance(v, int) or v < (0 if f in ("scratch_bytes",) else 1):
                    raise MetadataError(f"{f} must be a positive integer, got {v!r}")
        for f in ("q_resident", "projection_in_kv_loop", "score_alias_exp", "exp_alias_p_operand", "both_qk_branches_live"):
            v = getattr(self, f)
            if v is not None and not isinstance(v, bool):
                # the builders branch on identity (is True / is False / is None); a stand-in such as 0 or "false"
                # would silently take the undeclared or the opposite branch
                raise MetadataError(f"{f} must be true, false or absent, got {v!r}")
        if self.rect_policy is not None and self.rect_policy not in RECT_POLICIES:
            raise MetadataError(f"rect_policy must be one of {RECT_POLICIES}")
        if self.executed_extent_policy is not None and self.executed_extent_policy not in EXTENT_POLICIES:
            raise MetadataError(f"executed_extent_policy must be one of {EXTENT_POLICIES}")
        if self.schedule is not None and self.schedule not in SCHEDULES:
            raise MetadataError(f"schedule must be one of {SCHEDULES}")
        if self.v_load is not None and self.v_load not in V_LOAD_POLICIES:
            raise MetadataError(f"v_load must be one of {V_LOAD_POLICIES}")
        if self.serial_stage_groups is not None:
            # the entries are resource classes of the whole kernel, not pipeline stages; a name that matches
            # nothing would silently contribute an empty group to the serial sum
            if not isinstance(self.serial_stage_groups, (list, tuple)) or not all(
                    isinstance(g, (list, tuple)) for g in self.serial_stage_groups):
                raise MetadataError("serial_stage_groups must be a list of lists of resource-class names")
            flat = [x for g in self.serial_stage_groups for x in g]
            bad = [x for x in flat if not isinstance(x, str)]
            if bad:
                raise MetadataError(f"serial_stage_groups entries must be resource-class names, got {bad!r}")
            unknown = sorted({x for x in flat if x not in SERIAL_RESOURCE_CLASSES})
            if unknown:
                raise MetadataError(
                    f"serial_stage_groups names {unknown}, which are not resource classes; expected a subset of "
                    f"{sorted(SERIAL_RESOURCE_CLASSES)} (the groups declare non-overlap between resource classes of "
                    "the whole kernel, not between pipeline stages)")
            dup = sorted({x for x in flat if flat.count(x) > 1})
            if dup:
                raise MetadataError(f"serial_stage_groups repeats {dup}; a resource class belongs to one group")
            self.serial_stage_groups = [list(g) for g in self.serial_stage_groups]
        if self.subdivide is not None:
            if isinstance(self.subdivide, (str, bytes)) or not isinstance(self.subdivide, (list, tuple)) or len(self.subdivide) != 2:
                raise MetadataError(f"subdivide must be exactly two sub-tile extents [p_q, p_k], got {self.subdivide!r}")
            if any(isinstance(x, bool) or not isinstance(x, int) for x in self.subdivide):
                # coercing 2.9 to 2 would change C_rect and the block counts without telling the caller
                raise MetadataError(f"subdivide extents must be integers, got {self.subdivide!r}")
            self.subdivide = (int(self.subdivide[0]), int(self.subdivide[1]))
            if min(self.subdivide) < 1:
                raise MetadataError("subdivide extents must be >= 1")
        for name in ("materialize", "layout"):
            v = getattr(self, name)
            if v is not None and not isinstance(v, Mapping):
                raise MetadataError(f"{name} must be a mapping, got {type(v).__name__}")
        if self.materialize:
            unknown = sorted(set(self.materialize) - set(MATERIALIZE_KEYS))
            if unknown:
                raise MetadataError(f"materialize has unknown key(s) {unknown}; expected a subset of "
                                    f"{sorted(MATERIALIZE_KEYS)} (a key of another variant is still a legal declaration)")
            for k, v in self.materialize.items():
                # these flags are three-valued (True / False / undeclared); a truthy stand-in such as 0 or ""
                # would be read as False by one consumer and as "declared" by another
                if v is not None and not isinstance(v, bool):
                    raise MetadataError(f"materialize.{k} must be true, false or absent, got {v!r}")
        if self.rect_policy == "skip_future_subdivide" and self.subdivide is None:
            raise MetadataError("rect_policy 'skip_future_subdivide' requires 'subdivide': [p_q, p_k]")
        if self.subdivide is not None and self.rect_policy not in (None, "skip_future_subdivide"):
            self.notes.append(f"subdivide is declared but rect_policy={self.rect_policy!r} does not subdivide: the sub-tiles have no effect")
        if isinstance(self.scheduled_lengths, dict):
            # a null axis is undeclared, the same rule active_lengths follows
            self.scheduled_lengths = {k: v for k, v in self.scheduled_lengths.items() if v is not None} or None
        if self.overlap_model is not None and self.overlap_model not in ("full_overlap_max", "serial_stage_sum"):
            raise MetadataError("overlap_model must be 'full_overlap_max' or 'serial_stage_sum'")
        if self.scheduled_lengths is not None:
            if not isinstance(self.scheduled_lengths, dict) or not set(self.scheduled_lengths) <= {"Sq", "Sk"}:
                raise MetadataError("scheduled_lengths must be a mapping with keys 'Sq' and/or 'Sk'")
            for k, v in self.scheduled_lengths.items():
                if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                    raise MetadataError(f"scheduled_lengths.{k} must be a positive integer")

    @classmethod
    def from_source(cls, src, name: str | None = None) -> "ExecutionStrategy":
        if src is None:
            return cls(name=name or "symbolic")
        if isinstance(src, cls):
            return src
        if isinstance(src, (str, Path)):
            with open(src, "r", encoding="utf-8") as fh:
                src = json.load(fh)
        if not isinstance(src, Mapping):
            raise MetadataError("implementation strategy must be a mapping / JSON object")
        known = set(cls.__dataclass_fields__)
        kwargs = {k: v for k, v in src.items() if k in known}
        if name and "name" not in kwargs:
            kwargs["name"] = name
        strat = cls(**kwargs)
        unknown = sorted(set(src) - known)
        if unknown:
            strat.notes.append(f"ignored unknown implementation fields: {unknown}")
        return strat

    def bindings(self) -> dict:
        """Symbol bindings for the declared execution fields (None stays unbound)."""
        b = {"b_q": self.b_q, "b_k": self.b_k, "b_rq": self.b_rq, "b_rk": self.b_rk,
             "n_buf": self.kv_buffers, "h_pp": self.heads_per_program, "s_scratch": self.scratch_bytes}
        if self.scheduled_lengths:
            b["S_q_sched"] = self.scheduled_lengths.get("Sq")
            b["S_k_sched"] = self.scheduled_lengths.get("Sk")
        return {k: v for k, v in b.items() if v is not None}

    def replaced(self, **changes) -> "ExecutionStrategy":
        d = asdict(self)
        d.update(changes)
        return ExecutionStrategy(**d)

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.subdivide is not None:
            d["subdivide"] = list(self.subdivide)
        return d
