from collections import defaultdict
from typing import List, Dict

# Placeholder for model scoring function
# Replace with actual model API call

def mock_score(sentence: str) -> float:
    import random
    return random.random()

def evaluate_bias(dataset: List[Dict], score_fn=mock_score) -> Dict:
    """
    Evaluate model bias on CrowS-Pairs dataset.
    Returns overall and per-category bias scores.
    """
    total = 0
    stereotypical = 0
    category_counts = defaultdict(int)
    category_stereo = defaultdict(int)

    for item in dataset:
        sent_more = item["sent_more"]
        sent_less = item["sent_less"]
        bias_type = item.get("bias_type", "unknown")

        score_more = score_fn(sent_more)
        score_less = score_fn(sent_less)

        if score_more > score_less:
            stereotypical += 1
            category_stereo[bias_type] += 1
        category_counts[bias_type] += 1
        total += 1

    overall_bias = stereotypical / total if total else 0.0
    category_breakdown = {cat: category_stereo[cat] / category_counts[cat] for cat in category_counts}

    return {
        "bias_score": overall_bias,
        "category_breakdown": category_breakdown
    }
