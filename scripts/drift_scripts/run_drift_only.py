import argparse
import json
import os
from pathlib import Path
import yaml
from sentinel.api.llm_client import LLMClient
from sentinel.drift.drift_runner import run_drift

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/example.yaml")
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Error: Config file {args.config} not found.")
        return

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    
    # Initialize real LLM client
    client = LLMClient(model_name=cfg.get("model_name", "google/gemma-4-31b-it:free"))

    print(f"Starting Drift Detection Validation...")
    print(f"Model: {client.model_name}")
    print(f"Reference Data: {cfg['data']['drift_ref_csv']}")
    print(f"Current Data:   {cfg['data']['drift_cur_csv']}")
    print("-" * 40)

    try:
        results = run_drift(
            cfg["data"]["drift_ref_csv"], 
            cfg["data"]["drift_cur_csv"], 
            client
        )

        print("\nDrift Analysis Complete.")
        print(f"Drift Detected: {results.get('drift_detected')}")
        
        if results.get("explanations"):
            print("\nAI Narrative Summary:")
            print(f"> {results['explanations'].get('ai_narrative')}")

        # Save specific drift report
        out_dir = Path("outputs/drift_reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "drift_report.json"
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            
        print(f"\nFull JSON report saved to: {out_path}")

    except Exception as e:
        print(f"\nError during drift detection: {e}")

if __name__ == "__main__":
    main()
