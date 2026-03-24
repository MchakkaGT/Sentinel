import os
import json
from pathlib import Path
from sentinel.api.llm_client import LLMClient
from sentinel.drift.drift_runner import run_drift

def run_benchmark():
    os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":."
    client = LLMClient(model_name="stub-llm")
    
    ref_path = "data/benchmarks/ref_balanced.csv"
    cur_files = {
        "No Drift": "data/benchmarks/cur_no_drift.csv",
        "Label Shift": "data/benchmarks/cur_label_shift.csv",
        "Noise Shift": "data/benchmarks/cur_noise_shift.csv",
        "OOD Shift": "data/benchmarks/cur_ood_shift.csv",
        "Mixed Drift": "data/benchmarks/cur_mixed_drift.csv"
    }
    
    results = {}
    for name, path in cur_files.items():
        print(f"Running benchmark: {name}...")
        drift = run_drift(ref_path, path, client)
        results[name] = drift
    
    # Save a summary report
    summary_path = Path("outputs/benchmark_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Benchmark complete. Summary saved to {summary_path}")
    
    # Print a nice table
    print("\n| Scenario | Drift Detected | MMD | Chi2 P-Value | Familiarity Drift | Types |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for name, d in results.items():
        t1 = d["tier_1_standard"]
        t2 = d["tier_2_novel"]
        types = ", ".join(t2["drift_types"]) if t2["drift_types"] else "None"
        print(f"| {name} | {d['drift_detected']} | {t1['mmd_drift']} | {t1['chi2_p_value']} | {t2['familiarity_drift']} | {types} |")

if __name__ == "__main__":
    run_benchmark()
