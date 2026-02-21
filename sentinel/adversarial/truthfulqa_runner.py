import json
from pathlib import Path
from sentinel.api.llm_client import LLMClient

def run_truthfulqa(sample_jsonl_path: str, client: LLMClient) -> dict:
    """
    Minimal checkpoint 'truthfulness':
    Generates an answer and stores it; no real scoring yet.
    """
    p = Path(sample_jsonl_path)
    items = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    outputs = []
    for it in items:
        q = it["question"]
        gen = client.generate(q)
        outputs.append({"id": it["id"], "question": q, "generated": gen})

    return {
        "module": "adversarial",
        "benchmark": "truthfulqa_sample",
        "sample_size": len(items),
        "examples": outputs,
        "status": "ok"
    }