
import numpy as np

class LLMClient:
    """
    Minimal placeholder LLM client (skeleton)
    Replace `generate()` with a real API call later (OpenAI, local model, etc.).
    """
    def __init__(self, model_name: str = "stub-llm"):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        return f"[{self.model_name}] {prompt}"

    def score(self, text: str) -> float:
        """
        Returns a dummy 'log-likelihood' score for the text.
        In a real scenario, this would return the average log-prob of tokens.
        """

        return -len(text) / 10.0

    def embed(self, text: str) -> np.ndarray:
        """
        Returns a dummy embedding vector for the text.
        In a real scenario, this would call an embedding model (e.g., text-embedding-3-small).
        """

        seed = abs(hash(text)) % (10**8)
        np.random.seed(seed)
        return np.random.rand(16)

