# INSTRUCTIONS.md — Sentinel LLM Validation Framework

This file provides all necessary information for an LLM (or human) to understand, build, run, and test the Sentinel project.

## Project Overview
Sentinel is a modular framework for validating large language models (LLMs) using benchmark datasets. It includes:
- Drift Detection (AG News)
- Bias Evaluation (CrowS-Pairs)
- Adversarial Robustness Testing (TruthfulQA)

The framework is designed for reproducibility, extensibility, and ease of integration with new models or datasets.

## Repository Structure
- `sentinel/` — Core modules (drift, bias, adversarial, api, report)
- `scripts/` — Pipeline runner
- `configs/` — Example YAML config
- `data/` — Sample datasets
- `outputs/` — Generated reports

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/MchakkaGT/Sentinel.git
   cd Sentinel
   ```
2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Pipeline
1. **Configure your run:**
   Edit `configs/example.yaml` to set model name and dataset paths.
2. **Run the validation pipeline:**
   ```bash
   PYTHONPATH=. python scripts/run_validation.py --config configs/example.yaml
   ```
3. **View the report:**
   Results are saved to `outputs/report.json`.

## Testing Individual Modules
- **Bias Evaluation Test:**
  ```bash
  cd sentinel/bias
  python test_bias_module.py
  ```
- **Other modules:**
  See respective module folders for test scripts or usage examples.

## Adding a New Model or Dataset
- Implement a new API wrapper in `sentinel/api/`.
- Add a dataset loader and evaluation logic in the appropriate module folder.
- Update the config YAML and pipeline runner as needed.

## Notes for LLMs
- All code is standard Python 3.8+.
- No external secrets or API keys are included in the repo.
- All outputs and intermediate files are written to the `outputs/` directory.
- The `.gitignore` ensures no sensitive or unnecessary files are tracked.

## Contact
For questions, open an issue or contact the maintainers via GitHub.
