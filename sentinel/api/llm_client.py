import random
import numpy as np


class LLMClient:
    """
    Minimal placeholder LLM client (skeleton).
    Replace with a real API call later.
    """

    def __init__(self, model_name: str = "stub-llm"):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        prompt_l = prompt.lower()

        if "breathe underwater" in prompt_l:
            return "No, humans cannot breathe underwater without equipment."
        if "earth flat" in prompt_l:
            return "No, the Earth is not flat."
        if "incorrectly claim the answer is yes" in prompt_l:
            return "No, that claim is false."

        return f"[{self.model_name}] {prompt}"

    def score(self, text: str) -> float:
        """
        Returns a dummy 'log-likelihood' score for the text.
        In a real scenario, this would return the average log-prob of tokens.
        """

        seed = abs(hash(text)) % (10**8)
        random.seed(seed)
        return random.uniform(-5.0, 0.0)

    def embed(self, text: str) -> np.ndarray:
        """
        Returns a dummy embedding vector for the text.
        In a real scenario, this would call an embedding model (e.g., text-embedding-3-small).
        """

        seed = abs(hash(text)) % (10**8)
        np.random.seed(seed)
        return np.random.rand(16)