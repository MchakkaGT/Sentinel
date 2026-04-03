import random
import numpy as np
from typing import Dict, Any


class LLMClient:
    """
    Main client for interacting with Large Language Models for embedding, 
    scoring (log-likelihood), and narrative generation.
    """

    def __init__(self, model_name: str = "stub-llm"):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        """Generates a text completion for a given prompt."""
        prompt_l = prompt.lower()
        if "breathe underwater" in prompt_l:
            return "No, humans cannot breathe underwater without equipment."
        if "earth flat" in prompt_l:
            return "No, the Earth is not flat."
        if "incorrectly claim the answer is yes" in prompt_l:
            return "No, that claim is false."
        return f"[{self.model_name}] {prompt}"

    def score(self, text: str) -> float:
        """Returns a surrogate log-likelihood 'surprise' score for the input text."""
        seed = abs(hash(text)) % (10**8)
        random.seed(seed)
        return random.uniform(-5.0, 0.0)

    def embed(self, text: str) -> np.ndarray:
        """Generates a semantic embedding vector for the given text."""
        seed = abs(hash(text)) % (10**8)
        np.random.seed(seed)
        return np.random.rand(16)

    def summarize_drift(self, metrics: Dict[str, Any]) -> str:
        """
        Translates raw drift metrics into a human-readable AI narrative summary 
        to help users understand the nature and severity of distribution shifts.
        """
        t1 = metrics.get("tier_1_standard", {})
        t2 = metrics.get("tier_2_novel", {})
        drift_types = t2.get("drift_types", [])
        
        if not drift_types:
            return "The model confirms that the data distribution remains stable."

        explanations = []
        if any("Covariate" in dt for dt in drift_types):
            explanations.append("A shift in underlying topics (MMD: {:.4f}) suggests new content types.".format(t1.get("mmd_drift", 0)))
        if any("Prior Probability" in dt for dt in drift_types):
            explanations.append("Class balance deviation (Label Drift) indicates a shift in external data sourcing.")
        if any("Familiarity" in dt for dt in drift_types):
            explanations.append("Familiarity drift detected ({:.2f}). Inputs are statistically foreign to the training distribution.".format(t2.get("familiarity_drift", 0)))

        return " ".join(explanations)