import csv
import math
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from scipy import stats
from sentinel.api.llm_client import LLMClient

def _load_csv(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def run_drift(ref_csv_path: str, cur_csv_path: str, client: LLMClient) -> Dict[str, Any]:
    """
    Implements a two-tiered Semantic Drift report.
    1. Standard metrics (Embeddings, P-values)
    2. Sentinel Novel metric (LLM Familiarity)
    """
    ref_rows: List[Dict[str, Any]] = _load_csv(ref_csv_path)
    cur_rows: List[Dict[str, Any]] = _load_csv(cur_csv_path)

    if not ref_rows or not cur_rows:
        return {"status": "error", "message": "Missing reference or current data"}

    # --- 1. Standard Baselines ---

    # 1. Semantic Drift
    ref_embeddings = [client.embed(r.get("text", "")) for r in ref_rows]
    cur_embeddings = [client.embed(r.get("text", "")) for r in cur_rows]
    
    centroid_ref = np.mean(ref_embeddings, axis=0)
    centroid_cur = np.mean(cur_embeddings, axis=0)
    semantic_drift = 1.0 - _cosine_similarity(centroid_ref, centroid_cur)

    # 2. P-values
    ref_scores: List[float] = [client.score(r.get("text", "")) for r in ref_rows]
    cur_scores: List[float] = [client.score(r.get("text", "")) for r in cur_rows]
    
    ks_stat, ks_p_value = stats.ks_2samp(ref_scores, cur_scores)

    # 3. Label Distribution (If the input data has labels)
    has_labels = "label" in ref_rows[0] and "label" in cur_rows[0]
    dist_shift = None
    chi2_p_value = None

    if has_labels:
        def get_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
            counts: Dict[str, int] = {}
            for r in rows:
                l = str(r.get("label", "unknown"))
                counts[l] = counts.get(l, 0) + 1
            return counts

        ref_counts = get_counts(ref_rows)
        cur_counts = get_counts(cur_rows)
        all_labels = sorted(list(set(ref_counts.keys()) | set(cur_counts.keys())))
        
        # Distributions for distance
        total_ref = len(ref_rows)
        total_cur = len(cur_rows)
        ref_dist_list = [ref_counts.get(l, 0) / total_ref for l in all_labels]
        cur_dist_list = [cur_counts.get(l, 0) / total_cur for l in all_labels]
        dist_shift = math.sqrt(sum((ref_dist_list[i] - cur_dist_list[i])**2 for i in range(len(all_labels))))

        # Chi-Squared test
        observed = [cur_counts.get(l, 0) for l in all_labels]
        expected = [ref_counts.get(l, 0) * (total_cur / total_ref) for l in all_labels]
        
        if any(e == 0 for e in expected):
            expected = [e + 1e-6 for e in expected]
            expected_sum = sum(expected)
            expected = [e * (total_cur / expected_sum) for e in expected]
            
        _, chi2_p_value = stats.chisquare(f_obs=observed, f_exp=expected)

    # --- 2. Sentinel Novel Scoring ---

    avg_ref_score = np.mean(ref_scores)
    avg_cur_score = np.mean(cur_scores)
    familiarity_drift = abs(avg_ref_score - avg_cur_score)

    novelty_drifted = familiarity_drift > 2.0
    statistically_drifted = ks_p_value < 0.05 or (chi2_p_value is not None and chi2_p_value < 0.05)
    semantic_drifted = semantic_drift > 0.1

    is_drifted = novelty_drifted or statistically_drifted or semantic_drifted

    return {
        "module": "drift",
        "method": "Tiered Semantic Drift Suite",
        "tier_1_standard": {
            "semantic_drift": np.round(semantic_drift, 4),
            "ks_p_value": np.round(ks_p_value, 4),
            "distribution_shift": np.round(dist_shift, 4) if dist_shift is not None else "n/a",
            "chi2_p_value": np.round(chi2_p_value, 4) if chi2_p_value is not None else "n/a"
        },
        "tier_2_novel": {
            "familiarity_drift": np.round(familiarity_drift, 4),
            "interpretation": "High familiarity drift detected." if novelty_drifted else "Familiarity stable."
        },
        "drift_detected": bool(is_drifted),
        "status": "ok"
    }