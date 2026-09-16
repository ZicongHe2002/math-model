"""Workload entry integration, with integer oracles independent of the model helpers.

The shipped YAML has B=1, H=32, S=4096, Rq=1536, Rk=512, Dn=Dv=128,
Dr=64 and two-byte input storage. Constants below were calculated from spec 4–6.
"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from mla_model import CapabilityError, MLAForwardModel, MetadataError, MLAModelError
from mla_model.devices import load_device_profile
from mla_model.workload import load_workload


ROOT = Path(__file__).resolve().parents[1]
WORKLOAD = ROOT / "mla_fwd.yaml"
VARIANTS = ("expanded", "absorbed_two_step", "absorbed_precomputed")
VALID_CELLS = {"causal": 268_500_992, "none": 536_870_912}
# F_E = projection + 640*C; F_A = projection + 2176*C.
# The precomputed path has a different Q projection; merge amortization is separate.
USEFUL_FLOPS = {
    "causal.expanded": 257_739_980_800,
    "none.expanded": 429_496_729_600,
    "causal.absorbed_two_step": 670_157_504_512,
    "none.absorbed_two_step": 1_254_130_450_432,
    "causal.absorbed_precomputed": 807_596_457_984,
    "none.absorbed_precomputed": 1_391_569_403_904,
}
INPUT_BYTES = {
    "q_latent": 12_582_912,
    "kv_latent": 4_194_304,
    "q_pe": 16_777_216,
    "k_pe": 524_288,
    "w_q_nope": 12_582_912,
    "w_k_nope": 4_194_304,
    "w_v": 4_194_304,
}


@pytest.fixture
def source():
    return yaml.safe_load(WORKLOAD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def default_analysis():
    return MLAForwardModel().analyze_yaml(WORKLOAD, device="tpu-v6-e", algorithm="mla")


def test_workload_translation_preserves_declarations_and_owns_its_data(source):
    original = deepcopy(source)
    bundle = load_workload(source)
    assert source == original
    assert bundle["tensors"]["q_latent"] == {"shape": [1, 4096, 1536], "dtype": "bfloat16"}
    assert bundle["tensors"]["q_pe"]["shape"] == [1, 32, 4096, 64]
    assert bundle["tensors"]["w_k_nope"]["shape"] == [32, 512, 128]
    assert bundle["implementation"] == {"b_q": 512, "b_k": 512, "b_rq": 256, "b_rk": 128}
    # Parsing a workload must not turn an analytical scenario into a declaration.
    assert bundle["semantics"] == {"active_lengths": {"Sq": 4096, "Sk": 4096}}
    assert bundle["numerics"] == {"acceptance": {"atol": 0.01, "rtol": 0.01}}
    assert bundle["_workload"]["seed"] == 0
    assert bundle["_workload"]["devices"] == {"num_devices": 1}
    bundle["semantics"]["active_lengths"]["Sq"] = 1
    bundle["_workload"]["shape"]["seq_len"] = 1
    assert source == original
    assert load_workload(source)["tensors"]["q_latent"]["shape"] == [1, 4096, 1536]


def test_all_mla_scenarios_match_independent_flop_and_byte_totals(default_analysis):
    assert set(default_analysis.reports) == set(USEFUL_FLOPS)
    for scenario, expected_flops in USEFUL_FLOPS.items():
        mask, variant = scenario.split(".")
        report = default_analysis.reports[scenario]
        assert report.value("C_valid") == VALID_CELLS[mask]
        assert report.value("F_total[useful]") == expected_flops
        assert report.value("M_in") == 55_050_240
        assert report.value("M_in[metadata_ledger]") == 55_050_240
        assert report.value("M_O") == 33_554_432
        assert report.value("M_LSE") == 524_288
        for role, expected_bytes in INPUT_BYTES.items():
            assert report.value(f"bytes[{role}]") == expected_bytes
        if variant == "absorbed_precomputed":
            # The YAML supplies no reuse count: no fabricated amortized merge cost.
            assert report.value("F_Wmerge_amortized[useful]") is None


def test_assumed_execution_and_numerics_are_not_reported_as_facts(default_analysis):
    for report in default_analysis.reports.values():
        for name in ("C_valid", "F_total[useful]", "F_total[executed_graph]", "M_O"):
            metric = report[name]
            assert metric.status == "partial", (name, metric.status)
            assert metric.missing_fields, name
        assert report["M_in"].status == "bound"
        assert report["bytes[q_latent]"].status == "bound"
    summary = default_analysis.to_dict()
    assert len(summary["scenarios"]) == 6
    assert all(scenario["assumptions"] for scenario in summary["scenarios"])


def test_explicit_mask_variant_and_output_dtype_override_scenarios_without_mutation(source):
    source["semantics"] = {"mask": "none", "projection_scope": "full"}
    source["numerics"] = {"output_dtype": "float32"}
    original = deepcopy(source)
    result = MLAForwardModel().analyze_yaml(source, device="v6e", algorithm="expanded")
    assert source == original
    assert set(result.reports) == {"none.expanded"}
    report = result.reports["none.expanded"]
    assert report.value("C_valid") == 536_870_912
    assert report.value("F_total[useful]") == 429_496_729_600
    assert report["F_total[useful]"].status == "bound"
    assert report.value("M_O") == 67_108_864
    assert report.header["numerics"]["output_dtype"] == "float32"
    assert report.value("M_in") == 55_050_240


def test_mask_argument_rejects_a_conflicting_yaml_declaration(source):
    source["semantics"] = {"mask": "causal", "position_offset": 0}
    with pytest.raises(MLAModelError, match="mask"):
        MLAForwardModel().analyze_yaml(source, device="tpu-v6-e", algorithm="expanded", mask="none")


def test_malformed_shapes_fail_before_any_analysis(source):
    for invalid in (True, 4096.5, -1, 0, "4096"):
        bad = deepcopy(source)
        bad["shape"]["seq_len"] = invalid
        with pytest.raises(MetadataError, match="seq_len"):
            load_workload(bad)
    missing = deepcopy(source)
    del missing["shape"]["kv_latent_dim"]
    with pytest.raises(MetadataError, match="kv_latent_dim"):
        load_workload(missing)


def test_yaml_python_tags_are_rejected_without_executing_them(tmp_path):
    marker = tmp_path / "unexpected_side_effect"
    payload = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    malicious = tmp_path / "unsafe.yaml"
    malicious.write_text("!!python/object/apply:builtins.eval\n- " + json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(MetadataError):
        load_workload(malicious)
    assert not marker.exists()


def test_multiple_devices_and_batch_tiling_are_explicitly_unsupported(source):
    multiple = deepcopy(source)
    multiple["devices"]["num_devices"] = 2
    with pytest.raises(CapabilityError):
        load_workload(multiple)
    batched_tile = deepcopy(source)
    batched_tile["shape"]["batch"] = 4
    batched_tile["tiling"]["block_b"] = 2
    with pytest.raises(CapabilityError):
        load_workload(batched_tile)


def test_named_device_profile_keeps_physical_capacity_separate_from_kernel_budgets():
    profile = load_device_profile("tpu-v6-e")
    assert profile.name == "tpu-v6e"
    peak = profile.peak_matmul("bfloat16")
    assert (peak.value, peak.unit, peak.scope) == (918e12, "FLOP/s", "per_chip")
    bandwidth = profile.paths["hbm_to_vmem.bandwidth"]
    assert (bandwidth.value, bandwidth.unit, bandwidth.scope) == (1638e9, "byte/s", "per_chip")
    assert profile.memory["hbm.capacity_bytes"].value == 32 * 2**30
    assert profile.memory["vmem.capacity_bytes"].value == 128 * 2**20
    assert profile.memory["vmem.scoped_budget_bytes"].value is None
    assert profile.memory["vreg.usable_capacity_bytes"].value is None
    assert profile.paths["vmem_to_hbm.bandwidth"].value is None
    assert profile.calibration == []
    profile.memory["hbm.capacity_bytes"].value = 1
    assert load_device_profile("v6e").memory["hbm.capacity_bytes"].value == 32 * 2**30
    with pytest.raises(MetadataError, match="device"):
        load_device_profile("unknown-accelerator")


def test_report_bundle_serializes_and_writes_all_named_scenarios(default_analysis, tmp_path):
    summary = json.loads(default_analysis.to_json())
    assert set(summary["reports"]) == set(USEFUL_FLOPS)
    assert {scenario["id"] for scenario in summary["scenarios"]} == set(USEFUL_FLOPS)
    default_analysis.write(tmp_path)
    on_disk = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert on_disk == summary
    assert (tmp_path / "summary.md").read_text(encoding="utf-8") == default_analysis.to_markdown()
    normalized = json.loads((tmp_path / "normalized_input.json").read_text(encoding="utf-8"))
    assert normalized["tensors"]["q_latent"]["shape"] == [1, 4096, 1536]
    for scenario in USEFUL_FLOPS:
        assert json.loads((tmp_path / f"{scenario}.json").read_text(encoding="utf-8")) == summary["reports"][scenario]
        assert (tmp_path / f"{scenario}.md").read_text(encoding="utf-8")


def test_cli_model_json_matches_api_and_reports_invalid_device(tmp_path):
    output = tmp_path / "reports"
    command = [sys.executable, "-m", "mla_model", "model", "--yaml", str(WORKLOAD),
               "--device", "tpu-v6-e", "--algorithm", "expanded", "--mask", "none",
               "--format", "json", "--out-dir", str(output)]
    run = subprocess.run(command, capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert run.returncode == 0, run.stderr
    summary = json.loads(run.stdout)
    assert set(summary["reports"]) == {"none.expanded"}
    metrics = {metric["metric"]: metric for metric in summary["reports"]["none.expanded"]["metrics"]}
    assert metrics["F_total[useful]"]["value"] == 429_496_729_600
    assert metrics["M_in"]["value"] == 55_050_240
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == summary
    invalid = command.copy()
    invalid[invalid.index("--device") + 1] = "unknown-accelerator"
    failure = subprocess.run(invalid, capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert failure.returncode == 2
    assert "device" in failure.stderr.lower()
    assert "Traceback" not in failure.stderr
