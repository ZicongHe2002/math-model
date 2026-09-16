"""Run every example config through the model and write JSON + Markdown reports to examples/generated/.

All configurations are ILLUSTRATIVE fixtures: none is a production default.  Configs whose
name contains 'expected_error' must raise; the error text is recorded instead of a report.

    python examples/run_examples.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mla_model import MLAForwardModel, MLAModelError, comparison_to_markdown  # noqa: E402


def main() -> int:
    out_dir = ROOT / "examples" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = MLAForwardModel()
    summary = []
    for cfg_path in sorted((ROOT / "examples" / "configs").glob("*.json")):
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        name = cfg_path.stem
        hw = None
        if cfg.get("hardware_file"):
            hw = json.loads((ROOT / cfg["hardware_file"]).read_text(encoding="utf-8"))
        try:
            bound = model.bind(cfg["tensors"], cfg.get("semantics"), cfg.get("implementation"), cfg.get("numerics"), hw,
                               algorithm=cfg.get("algorithm", "expanded"))
            rep = bound.analyze()
            (out_dir / f"{name}.json").write_text(rep.to_json(), encoding="utf-8")
            (out_dir / f"{name}.md").write_text(rep.to_markdown(), encoding="utf-8")
            row = {"config": name, "status": "ok", "mode": rep.mode, "fingerprint": rep.task_fingerprint,
                   "conflicts": [c.message for c in rep.conflicts], "n_unknown_fields": len(rep.unknowns),
                   "F_total_useful": rep.value("F_total[useful]"), "C_valid": rep.value("C_valid")}
            if cfg.get("candidates"):
                cmp = bound.compare(cfg["candidates"])
                (out_dir / f"{name}.compare.json").write_text(json.dumps(cmp, indent=2, default=str), encoding="utf-8")
                (out_dir / f"{name}.compare.md").write_text(comparison_to_markdown(cmp), encoding="utf-8")
                row["pareto_set"] = cmp["pareto_set"]
            summary.append(row)
        except MLAModelError as e:
            text = f"{type(e).__name__}: {e}"
            (out_dir / f"{name}.error.txt").write_text(text, encoding="utf-8")
            summary.append({"config": name, "status": "explicit_error", "error": text.splitlines()[0][:200],
                            "expected": "expected_error" in name})
        except Exception:  # pragma: no cover
            (out_dir / f"{name}.traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            summary.append({"config": name, "status": "UNEXPECTED_EXCEPTION"})
    sym = model.describe(algorithm="expanded")
    (out_dir / "describe_expanded.md").write_text(sym.to_markdown(), encoding="utf-8")
    (out_dir / "describe_expanded.json").write_text(sym.to_json(), encoding="utf-8")
    sym2 = model.describe(algorithm="absorbed_two_step")
    (out_dir / "describe_absorbed_two_step.md").write_text(sym2.to_markdown(), encoding="utf-8")
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    for row in summary:
        print(json.dumps(row, default=str))
    bad = [r for r in summary if r["status"] == "UNEXPECTED_EXCEPTION" or (r["status"] == "explicit_error" and not r.get("expected") and "conflict" not in r["config"])]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
