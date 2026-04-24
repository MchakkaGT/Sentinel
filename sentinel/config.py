# sentinel/config.py

# Model Selection - Pivoting to Hugging Face
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
CHAT_MODEL = "deepseek-ai/DeepSeek-V4-Pro"

# Hugging Face API Configuration
HF_BASE_URL = "https://api-inference.huggingface.co/v1/"

# Drift Detection Thresholds
# Tier 1 (Standard)
SEMANTIC_DRIFT_THRESHOLD = 0.1
MMD_DRIFT_THRESHOLD = 0.05

# Tier 2 (Novelty)
FAMILIARITY_DRIFT_THRESHOLD = 2.0

# Statistical Significance
P_VALUE_THRESHOLD = 0.05
