"""CPU-only command line interface.

    python -m mla_model describe  [--algorithm ALG] [--adapter A] [--format json|md] [--out FILE]
    python -m mla_model analyze   --metadata M.json [--semantics S.json] [--implementation I.json]
                                  [--numerics N.json] [--hardware H.json] [--algorithm ALG] [--format json|md] [--out FILE]
    python -m mla_model compare   (same inputs) --candidates C.json
    python -m mla_model analyze   --config bundle.json      # one JSON with keys tensors/semantics/implementation/numerics/hardware/algorithm/candidates

    Omitting the algorithm on `analyze` preserves the variant as unknown (spec 11.1): the report carries the
    algorithm-independent metrics once and every variant-dependent metric once per named variant scenario.
    `compare` needs a declared variant.

Legacy bundles are JSON; the model command safely parses workload YAML. Nothing is executed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import MLAModelError
from .model import MLAForwardModel, comparison_to_markdown


KNOWN_BUNDLE_KEYS = {"tensors", "semantics", "implementation", "numerics", "hardware", "hardware_file", "algorithm", "adapter", "candidates"}


def _load(path):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise MLAModelError(f"file not found: {p}")
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as e:
        raise MLAModelError(f"{p}: invalid JSON ({e})") from e


def _bundle(args):
    cfg = _load(args.config) if args.config else {}
    if not isinstance(cfg, dict):
        raise MLAModelError("--config must contain a JSON object")
    unknown = sorted(k for k in cfg if k not in KNOWN_BUNDLE_KEYS and not k.startswith("_"))
    if unknown:
        sys.stderr.write(f"warning: ignored unknown bundle keys {unknown}\n")

    def pick(key, arg):
        return _load(arg) if arg else cfg.get(key)

    hardware = pick("hardware", args.hardware)
    if hardware is None and cfg.get("hardware_file"):
        hw_path = Path(cfg["hardware_file"])
        if not hw_path.is_absolute():
            base = Path(args.config).resolve().parent
            candidates = [base / hw_path, Path.cwd() / hw_path]
            # also try relative to the repository root (two levels above examples/configs)
            candidates.append(base.parent.parent / hw_path)
            hw_path = next((c for c in candidates if c.exists()), candidates[0])
        hardware = _load(hw_path)
    algorithm = args.algorithm or cfg.get("algorithm")
    candidates = pick("candidates", getattr(args, "candidates", None))
    if candidates is not None and (not isinstance(candidates, list) or not all(isinstance(c, dict) for c in candidates)):
        raise MLAModelError("candidates must be a JSON list of strategy objects")
    return {
        "tensors": pick("tensors", args.metadata),
        "semantics": pick("semantics", args.semantics),
        "implementation": pick("implementation", args.implementation),
        "numerics": pick("numerics", args.numerics),
        "hardware": hardware,
        "algorithm": algorithm,
        "adapter": args.adapter or cfg.get("adapter", "seven_input_latent"),
        "candidates": candidates,
    }


def _emit(text: str, out: str | None):
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    else:
        sys.stdout.write(text)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="mla_model", description="Parameter-driven mathematical model of MLA forward (CPU-only)")
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="bundle JSON with tensors/semantics/implementation/numerics/hardware/algorithm/candidates")
    common.add_argument("--metadata", help="tensor metadata JSON (role -> {shape, dtype, ...})")
    common.add_argument("--semantics")
    common.add_argument("--implementation")
    common.add_argument("--numerics")
    common.add_argument("--hardware")
    common.add_argument("--algorithm", choices=list(MLAForwardModel.algorithms))
    common.add_argument("--adapter")
    common.add_argument("--format", choices=["json", "md"], default="md")
    common.add_argument("--out")
    d = sub.add_parser("describe", parents=[common], help="symbolic mode")
    a = sub.add_parser("analyze", parents=[common], help="bind current metadata/config and analyze")
    c = sub.add_parser("compare", parents=[common], help="compare candidate strategies for the same task")
    c.add_argument("--candidates", help="JSON list of strategy dicts")
    workload = sub.add_parser("model", help="model a workload from YAML + device + algorithm")
    workload.add_argument("--yaml", required=True, help="workload YAML, e.g. mla_fwd.yaml")
    workload.add_argument("--device", required=True, help="device name, e.g. tpu-v6-e")
    workload.add_argument("--algorithm", required=True, choices=["mla", *MLAForwardModel.algorithms],
                          help="mla for all variants, or a specific variant")
    workload.add_argument("--mask", choices=["causal", "none"], help="explicit mask; otherwise use YAML or both scenarios")
    workload.add_argument("--out-dir", help="report directory (default: reports/<yaml stem>/<device>)")
    workload.add_argument("--format", choices=["md", "json"], default="md", help="stdout summary format")
    args = p.parse_args(argv)
    model = MLAForwardModel()
    try:
        if args.cmd == "model":
            result = model.analyze_yaml(args.yaml, device=args.device, algorithm=args.algorithm, mask=args.mask)
            out_dir = args.out_dir or Path("reports") / Path(args.yaml).stem / result.device["name"]
            result.write(out_dir)
            sys.stdout.write(result.to_json() if args.format == "json" else result.to_markdown())
            sys.stderr.write(f"wrote modelling reports to {Path(out_dir).resolve()}\n")
            return 0
        if args.cmd == "describe":
            rep = model.describe(adapter=args.adapter or "seven_input_latent", algorithm=args.algorithm or "expanded")
            _emit(rep.to_json() if args.format == "json" else rep.to_markdown(), args.out)
            return 0
        cfg = _bundle(args)
        if cfg["tensors"] is None:
            p.error("analyze/compare need --metadata or --config with 'tensors'")
        if cfg["algorithm"] is None and args.cmd == "compare":
            p.error("compare needs a declared algorithm path (--algorithm or 'algorithm' in --config): with the variant "
                    "preserved as unknown the report holds one metric per named variant scenario, so there is nothing "
                    "to difference against")
        bound = model.bind(cfg["tensors"], cfg["semantics"], cfg["implementation"], cfg["numerics"], cfg["hardware"],
                           algorithm=cfg["algorithm"], adapter=cfg["adapter"])
        if args.cmd == "analyze":
            rep = bound.analyze()
            _emit(rep.to_json() if args.format == "json" else rep.to_markdown(), args.out)
            return 0
        cands = cfg["candidates"]
        if not cands:
            p.error("compare needs --candidates or 'candidates' in --config")
        cmp = bound.compare(cands)
        _emit(json.dumps(cmp, indent=2, default=str) if args.format == "json" else comparison_to_markdown(cmp), args.out)
        return 0
    except MLAModelError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2
    except (OSError, ValueError, TypeError) as e:   # malformed inputs that escape the model's own validation
        sys.stderr.write(f"error: {type(e).__name__}: {e}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
