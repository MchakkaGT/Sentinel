# Sentinel — Modular LLM Validation Framework

**Project**

Enterprise AI systems increasingly rely on deployed large language models (LLMs), but production validation remains underdeveloped. Existing tools such as MLflow and Evidently AI focus primarily on statistical monitoring of structured data, rather than behavioral validation of generative models. Meanwhile, benchmark suites such as StereoSet, CrowS-Pairs, BBQ, and TruthfulQA evaluate bias and robustness independently but are not typically integrated into automated pipelines.

Current validation approaches are fragmented: drift detection is handled separately from fairness auditing, and adversarial testing is usually research-oriented rather than production-ready. There is a gap between benchmark research datasets and practical enterprise validation workflows.

Our project builds a modular validation framework that integrates statistical drift detection, bias benchmarking, and adversarial robustness testing into a single automated pipeline. The goal is to provide a minimum working validation system that evaluates an LLM endpoint using non-trivial benchmark datasets and produces a structured validation report.

**What This Repo Contains**

- **Modular validation pipeline**: glue to run drift detection, bias benchmarks, and adversarial tests together.
- **Benchmark integration**: adapters to run common datasets (StereoSet, CrowS-Pairs, BBQ, TruthfulQA) against an LLM endpoint.
- **Report generation**: produce a structured validation report combining statistical and behavioral measures.

**Goals**

- Provide a reproducible, minimal working example for validating an LLM endpoint.
- Bridge the gap between academic benchmarks and enterprise validation workflows.
- Make it easy to add new benchmarks, monitoring modules, and reporting outputs.

**Installation**

1. Clone the repository:

   ```bash
   git clone https://github.com/your-org/sentinel.git
   cd sentinel
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

**Quick Start**

1. Configure your LLM endpoint and dataset locations in a YAML config file (see `configs/example.yaml`):

   ```yaml
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

   ```bash
   export PYTHONPATH=$PYTHONPATH:.
   python scripts/run_validation.py --config configs/example.yaml
   ```

3. Inspect the generated validation report at `outputs/report.json` to guide deployment or remediation steps.

---

## Drift Detection (Semantic Dynamic Drift)

Sentinel implements a three-tiered **Semantic Dynamic Drift (SDD)** suite designed specifically for LLM workflows:

- **Tier 1 (Standard)**: Semantic Centroid Shift and statistical $p$-values (Kolmogorov-Smirnov Test).
- **Tier 2 (Distribution)**: **Maximum Mean Discrepancy (MMD)** with RBF kernels to detect subtle shifts in embedding topology that centroid-based methods miss.
- **Tier 3 (Sentinel Novelty)**: **Familiarity Drift (Surprise)** measuring shifts in LLM log-likelihood/perplexity to identify data the model is "less familiar" with.

### Drift Categorization
The module automatically categorizes detected drift into actionable types:
- **Covariate Shift**: Significant change in the input text distribution.
- **Prior Probability Shift**: Significant change in label/category prevalence.
- **Familiarity Drift**: Out-of-Distribution (OOD) data that surprises the model.

### Real-World Benchmarking (AG News)
Sentinel includes built-in support for the **AG News Dataset** to simulate real-world drift scenarios. You can run the comprehensive benchmark suite to verify detection accuracy across 5 distinct drift types:

```bash
export PYTHONPATH=$PYTHONPATH:.
python3 scripts/benchmark_drift.py
```

---

## Bias Evaluation (CrowS-Pairs)

This module measures language model bias using the CrowS-Pairs dataset:

- Loads sentence pairs (stereotypical vs. anti-stereotypical) from JSONL.
- Scores both sentences with the model.
- Computes how often the model prefers the stereotypical sentence (overall and by bias type).
- Results are included in the main validation report.

To test just the bias module logic, see `sentinel/bias/test_bias_module.py`.

---

## Adversarial Robustness Testing (TruthfulQA)

This module evaluates model robustness against adversarial prompting using a sample version of the TruthfulQA benchmark.

- Loads question prompts from a TruthfulQA-style JSONL dataset.
- Generates multiple prompt variants for each question, including:
  - the original prompt (base), misleading prefix prompts, confident rephrasing prompts, contradictory contextual prompts
- Sends each variant through the model interface to collect responses.
- Applies a baseline truthfulness scoring heuristic by comparing generated outputs to reference answers.

The module reports several metrics in the validation report, including:

- **Overall truth accuracy** across all prompt variants
- **Robust accuracy** measuring correctness under adversarial prompts only
- **Variant-level accuracy** showing how different prompt perturbations affect model behavior
- **Example-level outputs** containing prompts, generated responses, and truth scores

Results are included in the main validation report under the `adversarial` module.

---

**Roadmap**

Note: This repository focuses on a small, practical baseline. See the module code and adapters for integration details and examples.

**Future Extensions**

Planned improvements include:

- Threshold-based validation alerts
- Multi-model comparative evaluation
- Continuous integration support
- Expanded benchmark coverage (StereoSet, BBQ, etc.)

Sentinel is intended as a foundation that can evolve into a full production validation layer for generative AI systems.

**License & Contributions**

Contributions welcome — please open issues or PRs with improvements, additional benchmarks, or automation steps.
