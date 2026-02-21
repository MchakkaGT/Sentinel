import argparse
import json
from pathlib import Path

try:
    import yaml
except ImportError:
    raise SystemExit("Missing dependency: pyyaml. Install with: pip install pyyaml")

from sentinel.api.llm_client import LLMClient
from sentinel.drift.drift_runner import run_drift
from sentinel.bias.crows_pairs_runner import run_crows_pairs
from sentinel.adversarial.truthfulqa_runner import run_truthfulqa
from sentinel.report.report_builder import write_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    client = LLMClient(model_name=cfg.get("model_name", "stub-llm"))

    drift = run_drift(cfg["data"]["drift_csv"])
    bias = run_crows_pairs(cfg["data"]["crows_pairs_jsonl"], client)
    adv = run_truthfulqa(cfg["data"]["truthfulqa_jsonl"], client)

    report = {
        "project": "Sentinel",
        "model": client.model_name,
        "results": {
            "drift": drift,
            "bias": bias,
            "adversarial": adv
        }
    }

    out_path = cfg["output"]["report_json"]
    write_report(report, out_path)

    print("✅ Sentinel checkpoint run complete")
    print(f"Report: {out_path}")
    print(json.dumps({"drift": drift["status"], "bias": bias["status"], "adversarial": adv["status"]}, indent=2))

if __name__ == "__main__":
    main()