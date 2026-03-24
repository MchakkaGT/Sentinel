# Sentinel — Modular LLM Validation Framework

## Project

Enterprise AI systems increasingly rely on deployed large language models (LLMs), but production validation remains underdeveloped. Existing tools such as MLflow and Evidently AI focus primarily on statistical monitoring of structured data, rather than behavioral validation of generative models. Meanwhile, benchmark suites such as StereoSet, CrowS-Pairs, BBQ, and TruthfulQA evaluate bias and robustness independently but are not typically integrated into automated pipelines.

Current validation approaches are fragmented: drift detection is handled separately from fairness auditing, and adversarial testing is usually research-oriented rather than production-ready. There is a gap between benchmark research datasets and practical enterprise validation workflows.

Our project builds a modular validation framework that integrates statistical drift detection, bias benchmarking, and adversarial robustness testing into a single automated pipeline. The goal is to provide a minimum working validation system that evaluates an LLM endpoint using non-trivial benchmark datasets and produces a structured validation report.

---

## What This Repo Contains

- Modular validation pipeline: runs drift detection, bias benchmarks, and adversarial tests together  
- Benchmark integration: adapters for datasets such as CrowS-Pairs and TruthfulQA  
- Report generation: structured JSON report combining statistical and behavioral measures  

---

## Goals

- Provide a reproducible, minimal working example for validating an LLM endpoint  
- Bridge the gap between academic benchmarks and enterprise validation workflows  
- Make it easy to extend with new benchmarks and evaluation modules  

---

## Installation

1. Clone the repository:

```
git clone https://github.com/your-org/sentinel.git
cd sentinel
```

2. Create and activate a virtual environment:

```
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```

---

## Quick Start

1. Configure your LLM endpoint and dataset paths in `configs/example.yaml`:

```
model_name: "your-llm-endpoint"

data:
  drift_ref_csv: "data/samples/drift_ref.csv"
  drift_cur_csv: "data/samples/drift_cur.csv"
  crows_pairs_jsonl: "data/samples/crows_pairs_sample.jsonl"
  truthfulqa_jsonl: "data/samples/truthfulqa_sample.jsonl"

output:
  report_json: "outputs/report.json"
```

2. Run the validation pipeline:

```
export PYTHONPATH=$PYTHONPATH:.
python scripts/run_validation.py --config configs/example.yaml
```

3. Inspect results in:

```
outputs/report.json
```

---

# Drift Detection (Semantic Dynamic Drift)

Sentinel implements a three-tiered Semantic Dynamic Drift (SDD) suite designed for LLM workflows:

## Tier 1 — Standard Drift
- Semantic centroid shift
- Kolmogorov–Smirnov statistical test

## Tier 2 — Distribution Drift
- Maximum Mean Discrepancy (MMD) with RBF kernels
- Detects subtle changes in embedding distributions

## Tier 3 — Sentinel Novelty Detection
- Familiarity drift (surprise scoring)
- Identifies out-of-distribution (OOD) inputs using model behavior

---

## Drift Categorization

The system automatically classifies drift into:

- Covariate Shift — change in input distribution  
- Prior Probability Shift — change in label distribution  
- Familiarity Drift — inputs outside model’s learned distribution  

---

## Real-World Benchmarking (AG News)

Run drift benchmarking:

```
export PYTHONPATH=$PYTHONPATH:.
python3 scripts/benchmark_drift.py
```

---

# Bias Evaluation (CrowS-Pairs)

This module evaluates fairness using the CrowS-Pairs dataset.

## Core Process
- Compare stereotypical vs anti-stereotypical sentence pairs  
- Score both using the model  
- Measure which type is preferred  

## Metrics

- Bias Score  
- Directional Breakdown (stereotypical, anti-stereotypical, ties)  
- Category-Level Analysis (gender, race, religion, etc.)  
- Confidence Margin (strength of preference)  
- Summary Insights (most biased category + interpretation)  

---

# Adversarial Robustness Testing (TruthfulQA)

This module evaluates model robustness using adversarial prompt variations.

## Prompt Variants

- Base prompt  
- Misleading assumption  
- Confident rephrasing  
- Contradictory context  
- Logical trap  
- Authoritative pressure  

---

## Metrics

- Overall Truth Accuracy  
- Robust Accuracy (adversarial only)  
- Consistency Score (stability across variants)  
- Variant-Level Accuracy  
- Failure Mode Classification:
  - hallucination  
  - refusal  
  - contradiction/confusion  
  - partial correctness  

---

## Output

All modules are combined into:

```
outputs/report.json
```

Example:

```
{
  "drift": {...},
  "bias": {
    "bias_score": 0.62,
    "most_biased_category": "gender"
  },
  "adversarial": {
    "overall_truth_accuracy": 0.58,
    "robust_accuracy": 0.41,
    "consistency_score": 0.67
  }
}
```

---

## LLM Client

Located in:

```
sentinel/api/llm_client.py
```

Current implementation is a placeholder but can be extended to real APIs.

---

## Roadmap

- Replace heuristic scoring with real model likelihood scoring  
- Add multi-model comparison  
- Expand benchmark datasets  
- Add visualization/dashboard  
- Integrate CI/CD validation  

---

## License & Contributions

Contributions are welcome. Open issues or PRs for improvements.
