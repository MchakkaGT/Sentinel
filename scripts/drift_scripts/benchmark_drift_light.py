import os
import json
import sys
import csv
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from sentinel.api.llm_client import LLMClient
from sentinel.drift.drift_runner import run_drift
from sentinel.report.report_builder import write_report, generate_charts, write_consolidated_report

def subset_csv(input_path, output_path, n=10):
    """Creates a smaller version of a CSV file for faster benchmarking."""
    p = Path(input_path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        subset = reader[:n]
    
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=reader[0].keys())
        writer.writeheader()
        writer.writerows(subset)

def run_benchmark_light():
    client = LLMClient()
    
    # Create temporary light data
    tmp_data_dir = Path("data/benchmarks_light")
    tmp_data_dir.mkdir(parents=True, exist_ok=True)
    
    ref_path = "data/benchmarks/ref_balanced.csv"
    light_ref = tmp_data_dir / "ref_balanced.csv"
    subset_csv(ref_path, light_ref)
    
    cur_files = {
        "no_drift": "data/benchmarks/cur_no_drift.csv",
        "label_shift": "data/benchmarks/cur_label_shift.csv",
        "noise_shift": "data/benchmarks/cur_noise_shift.csv",
        "ood_shift": "data/benchmarks/cur_ood_shift.csv",
        "mixed_drift": "data/benchmarks/cur_mixed_drift.csv"
    }
    
    light_cur_files = {}
    for name, path in cur_files.items():
        light_path = tmp_data_dir / Path(path).name
        subset_csv(path, light_path)
        light_cur_files[name] = str(light_path)
    
    output_base = Path("outputs/investigations")
    assets_dir = output_base / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    master_results = {}
    chart_map = {}
    print("Starting LIGHT Consolidated Drift Investigation (Subset of 10 rows per scenario)...\n")
    
    for name, path in light_cur_files.items():
        print(f"Analyzing Scenario: {name.upper()}...")
        
        drift = run_drift(str(light_ref), path, client)
        master_results[name] = drift
        
        print(f"   Generating charts for {name}...")
        chart_paths = generate_charts(drift, str(assets_dir), prefix=name)
        chart_map[name] = chart_paths
    
    summary_path = output_base / "consolidated_summary.json"
    write_report(master_results, str(summary_path))
    
    final_md_path = output_base / "weekly_investigation_report.md"
    write_consolidated_report(master_results, str(final_md_path), chart_map)
    
    print(f"\nInvestigation complete. Central report saved to: {final_md_path}")

if __name__ == "__main__":
    run_benchmark_light()
