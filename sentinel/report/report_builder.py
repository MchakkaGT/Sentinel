import json
from pathlib import Path
from datetime import datetime

def write_report(report: dict, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    report["generated_at"] = datetime.utcnow().isoformat() + "Z"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)