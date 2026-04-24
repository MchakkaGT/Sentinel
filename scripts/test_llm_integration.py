import os
import sys
from pathlib import Path

# Add the project root to PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from sentinel.api.llm_client import LLMClient

def test_integration():
    print("Initializing LLMClient with OpenRouter...")
    try:
        client = LLMClient()
        
        test_text = "The quick brown fox jumps over the lazy dog."
        
        print("\n1. Testing Embedding...")
        emb = client.embed(test_text)
        print(f"Embedding shape: {emb.shape}")
        if emb.shape[0] > 0 and not all(emb == 0):
            print("SUCCESS: Embedding successful!")
        else:
            print("FAILURE: Embedding failed or returned zeros.")

        print("\n2. Testing Scoring (Surprise)...")
        score = client.score(test_text)
        print(f"Surprise Score: {score}")
        if isinstance(score, float):
            print("SUCCESS: Scoring successful!")
        else:
            print("FAILURE: Scoring failed.")

        print("\n3. Testing Narrative Generation...")
        mock_metrics = {
            "tier_1_standard": {"semantic_drift": 0.15, "mmd_drift": 0.08},
            "tier_2_novel": {"familiarity_drift": 2.4, "drift_types": ["Familiarity Drift"]}
        }
        summary = client.summarize_drift(mock_metrics)
        print(f"AI Narrative: {summary}")
        if len(summary) > 20:
            print("SUCCESS: Narrative generation successful!")
        else:
            print("FAILURE: Narrative too short or failed.")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")

if __name__ == "__main__":
    test_integration()
