import json
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.integer)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def write_report(report: Dict[str, Any], out_path: str) -> None:
    """Saves a drift investigation result dictionary to a JSON file."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    report["generated_at"] = datetime.utcnow().isoformat() + "Z"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)

def generate_charts(report: Dict[str, Any], out_dir: str, prefix: str = "") -> List[str]:
    """
    Generates distribution and label shift PNG charts for a specific drift scenario.
    Returns a list of absolute paths to the generated images.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    chart_paths = []
    viz_data = report.get("visualization_data", {})
    if not viz_data:
        return []

    sns.set_theme(style="whitegrid", palette="muted")
    file_prefix = f"{prefix}_" if prefix else ""
    
    # Familiarity distribution (KDE Plot)
    ref_scores = viz_data.get("ref_scores", [])
    cur_scores = viz_data.get("cur_scores", [])
    if ref_scores and cur_scores:
        plt.figure(figsize=(8, 4))
        sns.kdeplot(ref_scores, fill=True, label="Baseline", color="blue", alpha=0.3)
        sns.kdeplot(cur_scores, fill=True, label="Incoming", color="orange", alpha=0.3)
        plt.title(f"Familiarity Shift: {prefix.replace('_', ' ').title()}", fontsize=12)
        plt.xlabel("Log-Likelihood Score")
        plt.legend()
        
        path = out_path / f"{file_prefix}familiarity.png"
        plt.savefig(path, bbox_inches='tight', dpi=100)
        plt.close()
        chart_paths.append(str(path.absolute()))

    # Label distribution (Bar Plot)
    ref_labels = viz_data.get("ref_label_counts")
    cur_labels = viz_data.get("cur_label_counts")
    if ref_labels and cur_labels:
        data = []
        labels = sorted(set(ref_labels.keys()) | set(cur_labels.keys()))
        for l in labels:
            data.append({"Label": l, "Frequency": ref_labels.get(l, 0), "Dataset": "Ref"})
            data.append({"Label": l, "Frequency": cur_labels.get(l, 0), "Dataset": "Cur"})
        
        df = pd.DataFrame(data)
        plt.figure(figsize=(8, 4))
        sns.barplot(data=df, x="Label", y="Frequency", hue="Dataset")
        plt.title(f"Label Shift: {prefix.replace('_', ' ').title()}", fontsize=12)
        plt.ylabel("Count")
        
        path = out_path / f"{file_prefix}labels.png"
        plt.savefig(path, bbox_inches='tight', dpi=100)
        plt.close()
        chart_paths.append(str(path.absolute()))

    return chart_paths

def write_consolidated_report(all_results: Dict[str, Any], out_path: str, chart_map: Dict[str, List[str]]) -> None:
    """
    Compiles all scenario results and visual assets into a single Markdown investigation report.
    Uses relative paths for image embedding to ensure portability.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    lines = [
        f"# Sentinel Weekly Drift Investigation Report",
        f"**Consolidated at:** {datetime.utcnow().isoformat()}Z",
        f"",
        f"## Executive Overview",
        f"| Scenario | Status | Key Signals |",
        f"| :--- | :--- | :--- |"
    ]
    
    # Filtering scenario results from metadata
    scenarios = {k: v for k, v in all_results.items() if isinstance(v, dict)}
    
    for name, res in scenarios.items():
        status = "**DRIFT**" if res.get("drift_detected") else "**SAFE**"
        signals = ", ".join(res.get("tier_2_novel", {}).get("drift_types", [])) or "Stable"
        lines.append(f"| {name.replace('_', ' ').title()} | {status} | {signals} |")
    
    lines.append("")
    lines.append("---")
    
    for name, res in scenarios.items():
        pretty_name = name.replace('_', ' ').title()
        is_drifted = res.get("drift_detected", False)
        status_text = "DRIFT" if is_drifted else "SAFE"
        
        lines.append(f"## Scenario: {pretty_name} ({status_text})")
        lines.append(f"**Interpretation:** {res.get('tier_2_novel', {}).get('interpretation')}")
        lines.append("")

        # AI-powered narrative summary
        ai_narrative = res.get("explanations", {}).get("ai_narrative")
        if ai_narrative:
            lines.append("### AI Investigation Summary")
            lines.append(f"> {ai_narrative}")
            lines.append("")
        
        # Embedded visual assets
        if name in chart_map:
            lines.append("### Metrics Visualization")
            for cp in chart_map[name]:
                relative_path = f"assets/{Path(cp).name}"
                lines.append(f"![{pretty_name} Chart]({relative_path})")
            lines.append("")

        # Statistical Metrics Dashboard
        t1 = res.get("tier_1_standard", {})
        t2 = res.get("tier_2_novel", {})
        lines.append("### Core Metrics")
        lines.append("| Metric | Value | Verdict |")
        lines.append("| :--- | :--- | :--- |")
        lines.append(f"| Semantic Drift | {t1.get('semantic_drift')} | {'SHIFTED' if t1.get('semantic_drift', 0) > 0.1 else 'OK'} |")
        lines.append(f"| MMD Distance | {t1.get('mmd_drift')} | {'SHIFTED' if t1.get('mmd_drift', 0) > 0.05 else 'OK'} |")
        lines.append(f"| Familiarity Drift | {t2.get('familiarity_drift')} | {'NOVEL' if t2.get('familiarity_drift', 0) > 2.0 else 'OK'} |")
        lines.append("")

        # Root Cause Explanations and Examples
        ex = res.get("explanations", {})
        if ex.get("semantic_outliers") or ex.get("familiarity_surprises"):
            lines.append("### Root Cause Highlights")
            if "semantic_outliers" in ex:
                lines.append("**Top Semantic Deviations:**")
                for item in ex["semantic_outliers"]:
                    lines.append(f"- `{item['text']}` (Sim: {item['similarity']})")
            if "familiarity_surprises" in ex:
                lines.append("**Top Model Surprises:**")
                for item in ex["familiarity_surprises"]:
                    lines.append(f"- `{item['text']}` (Score: {item['score']})")
            lines.append("")

        lines.append("---")

    lines.append("## Mitigation Recommendations")
    lines.append("1. **Data Recalibration**: For scenarios with detected Semantic or Label Drift, re-baseline the embedding centroids.")
    lines.append("2. **OOD Inspection**: Manual review of 'Model Surprises' is recommended to identify emerging edge cases.")
    lines.append("3. **Threshold Tuning**: If false positives occur, consider increasing the MMD threshold to 0.08.")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))