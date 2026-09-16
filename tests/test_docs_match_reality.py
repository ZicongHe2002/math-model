"""The documents must not drift from the code.

Every figure a document quotes as measured is re-derived here, so a change that makes a claim false fails
the suite instead of surviving until someone re-reads the prose.
"""
import json
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
HANDOFF = (ROOT / "docs" / "HANDOFF.md").read_text()
README = (ROOT / "README.md").read_text()


def test_the_quoted_test_count_is_the_real_one():
    out = subprocess.run([sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q", "--collect-only"],
                         capture_output=True, text=True, cwd=ROOT).stdout
    collected = sum(1 for line in out.splitlines() if "::" in line)
    quoted = {int(m) for m in re.findall(r"(\d+) passed in", HANDOFF)}
    assert quoted == {collected}, f"HANDOFF quotes {quoted}, the suite collects {collected}"
    per_file = dict(re.findall(r"\| `(tests/[^`]+)` \| (\d+) \|", HANDOFF))
    actual = {}
    for line in out.splitlines():
        if "::" in line:
            actual[line.split("::")[0]] = actual.get(line.split("::")[0], 0) + 1
    assert {k: int(v) for k, v in per_file.items()} == actual


def test_the_quoted_report_shape_is_the_real_one():
    rep = json.loads((ROOT / "examples" / "generated" / "01_square_causal_prefill_expanded.json").read_text())
    counts = {}
    for m in rep["metrics"]:
        counts[m["status"]] = counts.get(m["status"], 0) + 1
    quoted_metrics = int(re.search(r"\| Metrics \| (\d+) across (\d+) sections \|", HANDOFF).group(1))
    quoted_sections = int(re.search(r"\| Metrics \| (\d+) across (\d+) sections \|", HANDOFF).group(2))
    b, p, u, na = (int(x) for x in re.search(
        r"\| Bound / partial / unknown / not applicable \| (\d+) / (\d+) / (\d+) / (\d+) \|", HANDOFF).groups())
    assert (quoted_metrics, quoted_sections) == (len(rep["metrics"]), len({m["section"] for m in rep["metrics"]}))
    assert (b, p, u, na) == (counts.get("bound", 0), counts.get("partial", 0),
                             counts.get("unknown", 0), counts.get("not_applicable", 0))
    assert int(re.search(r"\| Constraints \| (\d+) \|", HANDOFF).group(1)) == len(rep["constraints"])
    assert int(re.search(r"\| Named unresolved input fields \| (\d+) \|", HANDOFF).group(1)) == len(rep["unknowns"])


def test_every_metric_name_the_readme_prints_exists():
    from conftest import NUMERICS, make_metadata
    from mla_model import MLAForwardModel
    sem = {"mask": "causal", "position_offset": 0, "scale_policy": "standard", "outputs": ["O"],
           "projection_scope": "full", "empty_row_policy": "zero_output_neg_inf_lse",
           "active_lengths": {"Sq": 32, "Sk": 32}}
    impl = {"b_q": 8, "b_k": 8, "rect_policy": "skip_future", "executed_extent_policy": "logical",
            "kv_buffers": 2, "heads_per_program": 1, "vreg_budget_bytes": 65536, "overlap_model": "full_overlap_max",
            "materialize": {"expanded_kv": True, "q_nope": False}, "scratch_bytes": 0}
    hw = {"name": "p", "scope_default": "per_chip",
          "compute": {"mxu": {"peak_flops": {"bfloat16": {"value": 1e14, "unit": "FLOP/s", "scope": "per_chip"}}}},
          "paths": {"hbm_to_vmem": {"bandwidth": {"value": 8e11, "unit": "byte/s", "scope": "per_chip"}}}}
    rep = MLAForwardModel().bind(make_metadata(1, 2, 32, 32, 4, 6, 8, 4, 8), sem, impl, NUMERICS, hw,
                                 algorithm="expanded").analyze()
    printed = re.findall(r"`([A-Za-z_][A-Za-z_0-9]*\[[^`]+\])`", README)
    missing = sorted({n for n in printed if rep.get(n) is None})
    assert not missing, f"README prints metric names no report emits: {missing}"


def test_documented_commands_exist_and_parse():
    for cmd in re.findall(r"^python3 -m mla_model ([a-z]+)", README, flags=re.M):
        assert cmd in ("describe", "analyze", "compare"), cmd
    r = subprocess.run([sys.executable, "-m", "mla_model", "describe", "--algorithm", "expanded", "--format", "md"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and r.stdout
