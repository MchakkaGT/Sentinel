import csv
from pathlib import Path

def run_drift(sample_csv_path: str) -> dict:
    """
    Minimal drift 'signal': counts labels and avg text length.
    (Real drift later: KL/JS divergence, embeddings drift, etc.)
    """
    p = Path(sample_csv_path)
    rows = []
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    label_counts = {}
    lengths = []
    for r in rows:
        label = r.get("label", "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1
        lengths.append(len((r.get("text") or "").strip()))

    avg_len = sum(lengths) / max(len(lengths), 1)

    return {
        "module": "drift",
        "sample_size": len(rows),
        "label_counts": label_counts,
        "avg_text_length": avg_len,
        "status": "ok"
    }