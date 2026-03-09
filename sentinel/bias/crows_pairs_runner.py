import json
from pathlib import Path
from sentinel.api.llm_client import LLMClient
from .dataset_loader import load_crows_pairs
from .evaluator import evaluate_bias

def run_crows_pairs(sample_jsonl_path: str, client: LLMClient) -> dict:
    """
    Evaluate model bias using CrowS-Pairs dataset and LLM client.
    Returns detailed bias scores for reporting.
    """
    # Load dataset
    items = load_crows_pairs(sample_jsonl_path)

    # Use the LLM client to score sentences
    def score_fn(sentence):
        # Replace with actual scoring logic as needed
        return client.score(sentence)

    # Evaluate bias (overall and per-category)
    results = evaluate_bias(items, score_fn=score_fn)

    return {
        "module": "bias",
        "benchmark": "crows_pairs_sample",
        "sample_size": len(items),
        "bias_score": results["bias_score"],
        "category_breakdown": results["category_breakdown"],
        "status": "ok"
    }