from sentinel.api.llm_client import LLMClient
from .dataset_loader import load_crows_pairs
from .evaluator import evaluate_bias, run_sensitivity_test


def run_crows_pairs(sample_jsonl_path: str, client: LLMClient) -> dict:
    """
    Evaluate model bias using CrowS-Pairs dataset and LLM client.
    Includes sensitivity testing and directional bias detection.
    """
    # Load dataset
    items = load_crows_pairs(sample_jsonl_path)

    # Use the LLM client to score sentences
    def score_fn(sentence):
        return client.score(sentence)

    # Evaluate bias (overall, per-category, and directional)
    results = evaluate_bias(items, score_fn=score_fn)
    
    # Run sensitivity test
    sensitivity_results = run_sensitivity_test(items, score_fn=score_fn)

    return {
        "module": "bias",
        "benchmark": "crows_pairs_sample",
        "sample_size": len(items),
        "bias_score": results["bias_score"],
        "avg_margin": results["avg_margin"],
        "stereotypical_total": results["stereotypical_total"],
        "anti_stereotypical_total": results["anti_stereotypical_total"],
        "category_breakdown": results["category_breakdown"],
        "sensitivity_test": sensitivity_results,
        "summary": results["summary"],
        "status": "ok"
    }