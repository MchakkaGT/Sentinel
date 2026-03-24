from sentinel.api.llm_client import LLMClient
from .dataset_loader import load_crows_pairs
from .evaluator import evaluate_bias


def run_crows_pairs(sample_jsonl_path: str, client: LLMClient) -> dict:
    """
    Evaluate model bias using CrowS-Pairs dataset and LLM client.
    Returns detailed bias scores for reporting.
    """
    items = load_crows_pairs(sample_jsonl_path)

    def score_fn(sentence: str) -> float:
        return client.score(sentence)

    results = evaluate_bias(items, score_fn=score_fn)

    return {
        "module": "bias",
        "benchmark": "crows_pairs_sample",
        "sample_size": len(items),
        "bias_score": results["bias_score"],
        "avg_margin": results["avg_margin"],
        "directional_breakdown": results["directional_breakdown"],
        "category_breakdown": results["category_breakdown"],
        "summary": results["summary"],
        "most_biased_category": results["most_biased_category"],
        "status": "ok",
    }