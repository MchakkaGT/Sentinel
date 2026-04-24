import os
import json
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from sentinel.api.llm_client import LLMClient
from sentinel.drift.drift_runner import run_drift
from sentinel.report.report_builder import write_report, generate_charts, write_consolidated_report

def run_benchmark():
    os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":."
    client = LLMClient(model_name="google/gemini-2.0-flash-exp:free")
    
    ref_path = "data/benchmarks/ref_balanced.csv"
    cur_files = {
        "no_drift": "data/benchmarks/cur_no_drift.csv",
        "label_shift": "data/benchmarks/cur_label_shift.csv",
        "noise_shift": "data/benchmarks/cur_noise_shift.csv",
        "ood_shift": "data/benchmarks/cur_ood_shift.csv",
        "mixed_drift": "data/benchmarks/cur_mixed_drift.csv"
    }
    
    output_base = Path("outputs/investigations")
    assets_dir = output_base / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    master_results = {}
    chart_map = {}
    print("Starting Consolidated Drift Investigation...\n")
    
    for name, path in cur_files.items():
        print(f"Analyzing Scenario: {name.upper()}...")
        
        drift = run_drift(ref_path, path, client)
        master_results[name] = drift
        
        print(f"   Generating charts for {name}...")
        chart_paths = generate_charts(drift, str(assets_dir), prefix=name)
        chart_map[name] = chart_paths
    
    summary_path = output_base / "consolidated_summary.json"
    write_report(master_results, str(summary_path))
    
    final_md_path = output_base / "weekly_investigation_report.md"
    write_consolidated_report(master_results, str(final_md_path), chart_map)
    
    print(f"\nInvestigation complete. Central report saved to: {final_md_path}")
    
    print("\n| Scenario | Status | Semantic | MMD | Label P-Val | Familiarity |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for name, d in master_results.items():
        if not isinstance(d, dict):
            continue
        t1 = d.get("tier_1_standard", {})
        t2 = d.get("tier_2_novel", {})
        status = "DRIFT" if d.get("drift_detected") else "SAFE"
        print(f"| {name:15} | {status:8} | {t1.get('semantic_drift', 0):.4f} | {t1.get('mmd_drift', 0):.4f} | {str(t1.get('chi2_p_value', 'n/a')):10} | {t2.get('familiarity_drift', 0):.4f} |")

if __name__ == "__main__":
    run_benchmark()
