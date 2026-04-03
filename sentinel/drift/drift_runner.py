import csv
import math
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from scipy import stats
from sentinel.api.llm_client import LLMClient

def _load_csv(path: str) -> List[Dict[str, Any]]:
    """Loads and parses a CSV file into a list of dictionaries."""
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculates the cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def _calculate_mmd(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    """
    Computes Maximum Mean Discrepancy (MMD) with an RBF kernel to detect 
    distributional shifts between two sets of embeddings.
    """
    def rbf_kernel(A, B, s):
        sq_dist = np.sum(A**2, axis=1).reshape(-1, 1) + np.sum(B**2, axis=1) - 2 * np.dot(A, B.T)
        return np.exp(-sq_dist / (2 * s**2))

    K_xx = rbf_kernel(X, X, sigma)
    K_yy = rbf_kernel(Y, Y, sigma)
    K_xy = rbf_kernel(X, Y, sigma)
    
    m = X.shape[0]
    n = Y.shape[0]
    
    mmd_sq = (np.sum(K_xx) - np.trace(K_xx)) / (m * (m - 1)) + \
             (np.sum(K_yy) - np.trace(K_yy)) / (n * (n - 1)) - \
             2 * np.mean(K_xy)
             
    return float(np.sqrt(max(mmd_sq, 0)))

def run_drift(ref_csv_path: str, cur_csv_path: str, client: LLMClient) -> Dict[str, Any]:
    """
    Executes a tiered drift analysis (Semantic, Distributional, and Familiarity) 
    comparing a reference dataset against current incoming data.
    """
    ref_rows: List[Dict[str, Any]] = _load_csv(ref_csv_path)
    cur_rows: List[Dict[str, Any]] = _load_csv(cur_csv_path)

    if not ref_rows or not cur_rows:
        return {"status": "error", "message": "Missing reference or current data"}

    # Semantic and Distributional Metrics
    ref_embeddings = np.array([client.embed(r.get("text", "")) for r in ref_rows])
    cur_embeddings = np.array([client.embed(r.get("text", "")) for r in cur_rows])
    
    centroid_ref = np.mean(ref_embeddings, axis=0)
    centroid_cur = np.mean(cur_embeddings, axis=0)
    semantic_drift = 1.0 - _cosine_similarity(centroid_ref, centroid_cur)
    mmd_drift = _calculate_mmd(ref_embeddings, cur_embeddings)

    # Statistical Significance (K-S Test)
    ref_scores: List[float] = [client.score(r.get("text", "")) for r in ref_rows]
    cur_scores: List[float] = [client.score(r.get("text", "")) for r in cur_rows]
    ks_stat, ks_p_value = stats.ks_2samp(ref_scores, cur_scores)

    # Label Distribution Shift (Chi-Squared Test)
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
        
        total_ref = len(ref_rows)
        total_cur = len(cur_rows)
        ref_dist_list = [ref_counts.get(l, 0) / total_ref for l in all_labels]
        cur_dist_list = [cur_counts.get(l, 0) / total_cur for l in all_labels]
        dist_shift = math.sqrt(sum((ref_dist_list[i] - cur_dist_list[i])**2 for i in range(len(all_labels))))

        observed = [cur_counts.get(l, 0) for l in all_labels]
        expected = [ref_counts.get(l, 0) * (total_cur / total_ref) for l in all_labels]
        
        if any(e == 0 for e in expected):
            expected = [e + 1e-6 for e in expected]
            expected_sum = sum(expected)
            expected = [e * (total_cur / expected_sum) for e in expected]
        _, chi2_p_value = stats.chisquare(f_obs=observed, f_exp=expected)

    # Model Familiarity Drift
    avg_ref_score = np.mean(ref_scores)
    avg_cur_score = np.mean(cur_scores)
    familiarity_drift = abs(avg_ref_score - avg_cur_score)

    # Drift Categorization
    novelty_drifted = familiarity_drift > 2.0
    semantic_drifted = semantic_drift > 0.1 or mmd_drift > 0.05
    
    drift_types = []
    if semantic_drifted:
        drift_types.append("Covariate Shift")
    if chi2_p_value is not None and chi2_p_value < 0.05:
        drift_types.append("Prior Probability Shift")
    if novelty_drifted:
        drift_types.append("Familiarity Drift")

    is_drifted = len(drift_types) > 0

    # Build investigation results
    res_metrics = {
        "tier_1_standard": {
            "semantic_drift": np.round(semantic_drift, 4),
            "mmd_drift": np.round(mmd_drift, 4),
            "ks_p_value": np.round(ks_p_value, 4),
            "distribution_shift": np.round(dist_shift, 4) if dist_shift is not None else "n/a",
            "chi2_p_value": np.round(chi2_p_value, 4) if chi2_p_value is not None else "n/a"
        },
        "tier_2_novel": {
            "familiarity_drift": np.round(familiarity_drift, 4),
            "drift_types": drift_types,
            "interpretation": "Drift detected: " + ", ".join(drift_types) if is_drifted else "Familiarity stable."
        }
    }

    explanations = {}
    explanations["ai_narrative"] = client.summarize_drift(res_metrics)
    
    # Root Cause Identification
    similarities = [_cosine_similarity(centroid_ref, emb) for emb in cur_embeddings]
    outlier_indices = np.argsort(similarities)[:3]
    explanations["semantic_outliers"] = [
        {"text": cur_rows[i].get("text", "")[:100] + "...", "similarity": float(np.round(similarities[i], 4))}
        for i in outlier_indices
    ]
    
    surprise_indices = np.argsort(cur_scores)[:3]
    explanations["familiarity_surprises"] = [
        {"text": cur_rows[i].get("text", "")[:100] + "...", "score": float(np.round(cur_scores[i], 4))}
        for i in surprise_indices
    ]

    if has_labels:
        explanations["label_shift"] = {
            l: {"ref": ref_counts.get(l, 0), "cur": cur_counts.get(l, 0), "delta": cur_counts.get(l, 0) - ref_counts.get(l, 0)}
            for l in all_labels
        }

    return {
        "module": "drift",
        "method": "Tiered Semantic Drift Suite",
        "tier_1_standard": res_metrics["tier_1_standard"],
        "tier_2_novel": res_metrics["tier_2_novel"],
        "drift_detected": bool(is_drifted),
        "explanations": explanations,
        "visualization_data": {
            "ref_scores": [float(s) for s in ref_scores],
            "cur_scores": [float(s) for s in cur_scores],
            "ref_label_counts": {k: int(v) for k, v in ref_counts.items()} if has_labels else None,
            "cur_label_counts": {k: int(v) for k, v in cur_counts.items()} if has_labels else None,
        },
        "status": "ok"
    }