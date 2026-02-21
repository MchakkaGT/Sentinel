class LLMClient:
    """
    Minimal placeholder LLM client (skeleton)
    Replace `generate()` with a real API call later (OpenAI, local model, etc.).
    """
    def __init__(self, model_name: str = "stub-llm"):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        return f"[{self.model_name}] {prompt}"