import json
from pathlib import Path
from sentinel.api.llm_client import LLMClient

def run_crows_pairs(sample_jsonl_path: str, client: LLMClient) -> dict:
    """
    Minimal checkpoint scoring:
    Just calls the model on both sentences and reports counts by bias_type.
    """
    p = Path(sample_jsonl_path)
    items = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    by_type = {}
    for it in items:
        b = it.get("bias_type", "unknown")
        by_type[b] = by_type.get(b, 0) + 1
        _ = client.generate(it["sent_more"])
        _ = client.generate(it["sent_less"])

    return {
        "module": "bias",
        "benchmark": "crows_pairs_sample",
        "sample_size": len(items),
        "by_bias_type": by_type,
        "status": "ok"
    }