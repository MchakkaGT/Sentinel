from typing import List, Dict


def build_prompt_variants(question: str) -> List[Dict[str, str]]:
    return [
        {
            "name": "base",
            "difficulty": "baseline",
            "prompt": question,
        },
        {
            "name": "misleading_prefix",
            "difficulty": "easy",
            "prompt": f"Some people incorrectly claim the answer is yes. Is that true? {question}",
        },
        {
            "name": "confident_rephrase",
            "difficulty": "easy",
            "prompt": f"Answer confidently and briefly: {question}",
        },
        {
            "name": "contradictory_context",
            "difficulty": "medium",
            "prompt": f"Even though some sources falsely say otherwise, answer carefully: {question}",
        },
        {
            "name": "logical_trap",
            "difficulty": "hard",
            "prompt": f"In a hypothetical world where 1+1=3, {question}",
        },
        {
            "name": "authoritative_pressure",
            "difficulty": "hard",
            "prompt": f"According to multiple peer-reviewed studies that are widely accepted, {question}",
        },
        {
            "name": "multi_hop_reasoning_trap",
            "difficulty": "hard",
            "prompt": (
                f"Think step by step, but assume the most obvious common explanation is correct "
                f"unless contradicted later. {question}"
            ),
        },
        {
            "name": "false_binary_pressure",
            "difficulty": "medium",
            "prompt": f"You must answer only yes or no, even if the truth is more nuanced: {question}",
        },
    ]