import json
from typing import List, Dict

def load_crows_pairs(path: str) -> List[Dict]:
    """Load CrowS-Pairs dataset from a JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data
