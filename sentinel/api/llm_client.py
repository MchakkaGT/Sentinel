import os
import time
import numpy as np
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from sentinel.config import CHAT_MODEL, EMBEDDING_MODEL

# Load environment variables from .env
load_dotenv()

class LLMClient:
    """
    Main client for interacting with Large Language Models via OpenRouter 
    for embedding, scoring (log-likelihood), and narrative generation.
    """

    def __init__(self, model_name: str = CHAT_MODEL):
        api_key = os.getenv("HUGGING_FACE_API_KEY")
        if not api_key:
            raise ValueError("HUGGING_FACE_API_KEY not found in environment or .env file")
        
        self.model_name = model_name
        self.client = InferenceClient(token=api_key)
        self.api_exhausted = False

    def generate(self, prompt: str) -> str:
        """Generates a text completion for a given prompt via HF."""
        if self.api_exhausted:
            return "API Exhausted"
        try:
            response = self.client.chat_completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"Error: {e}"

    def score(self, text: str) -> float:
        """
        Returns a log-likelihood 'surprise' score for the input text via HF.
        """
        if self.api_exhausted:
            return -5.0
            
        try:
            # Note: Serverless logprobs might be limited. We use it as a proxy.
            response = self.client.chat_completion(
                model=self.model_name,
                messages=[{"role": "user", "content": text}],
                max_tokens=1,
                logprobs=True
            )
            
            logprobs = response.choices[0].logprobs
            if logprobs and logprobs.content:
                return float(logprobs.content[0].logprob)
            return -2.5
        except Exception as e:
            print(f"Error getting score: {e}")
            return -5.0

    def embed(self, text: str) -> np.ndarray:
        """Generates a semantic embedding vector for the given text via HF."""
        try:
            return np.array(self.client.feature_extraction(text, model=EMBEDDING_MODEL))
        except Exception as e:
            print(f"Error getting HF embedding: {e}")
            return np.zeros(4096)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generates semantic embedding vectors for a batch of text via HF."""
        try:
            # feature_extraction supports lists of strings
            return np.array(self.client.feature_extraction(texts, model=EMBEDDING_MODEL))
        except Exception as e:
            print(f"Error getting HF batch embedding: {e}")
            return np.zeros((len(texts), 4096))

    def score_batch(self, texts: List[str], max_workers: int = 1) -> List[float]:
        """
        Returns log-likelihood 'surprise' scores for a batch of text.
        Uses threading to parallelize requests.
        Reduced max_workers to 1 to stay within strict free tier 'per-minute' rate limits.
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            scores = list(executor.map(self.score, texts))
        return scores

    def summarize_drift(self, metrics: Dict[str, Any]) -> str:
        """
        Translates raw drift metrics into a human-readable AI narrative summary 
        using a real LLM to provide deep diagnostic insights based on the 
        Semantic Dynamic Drift (SDD) framework.
        """
        
        prompt = f"""
        You are a Senior Data Scientist and AI Security Expert specializing in Generative AI monitoring.
        Analyze the following drift detection metrics from the 'Sentinel' validation pipeline.

        --- SDD FRAMEWORK CONTEXT ---
        Sentinel uses a Three-Tiered Semantic Dynamic Drift (SDD) approach:
        1. Tier 1 (Standard): Measures Semantic Centroid Shift (Cosine) and Distributional MMD.
           - Thresholds: Semantic > 0.1, MMD > 0.05.
        2. Tier 2 (Novel): Measures Familiarity Drift (Surprise) using model log-likelihood.
           - Threshold: Familiarity > 2.0.
        3. Statistical Significance: Uses K-S and Chi-Squared tests to validate if shifts are random noise.

        --- RAW METRICS ---
        {metrics}

        --- YOUR TASK ---
        Provide a deep diagnostic summary (4-5 sentences) that adds value beyond the raw numbers.
        Address the following:
        - SEVERITY: How critical is this shift? (Low, Medium, High, Critical)
        - DIAGNOSIS: Is this Covariate Shift (input change), Prior Probability Shift (label change), or a Novelty/Familiarity event (OOD)?
        - SECURITY IMPLICATIONS: Does high 'Surprise' indicate potential adversarial attempts or total model failure on new domains?
        - REMEDIATION: Specific next steps (e.g., re-baselining, fine-tuning, or manual labeling of surprises).

        Keep the tone professional, highly technical, and actionable. Avoid stating "The metrics show..." and dive straight into the expertise.
        """
        return self.generate(prompt)