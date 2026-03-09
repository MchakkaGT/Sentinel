"""
Super simple test for the bias evaluation pipeline using CrowS-Pairs sample data.
"""
from dataset_loader import load_crows_pairs
from evaluator import evaluate_bias

# Use the sample CrowS-Pairs data
SAMPLE_PATH = '../../data/samples/crows_pairs_sample.jsonl'

def test_bias_evaluation():
    data = load_crows_pairs(SAMPLE_PATH)
    # Use the default mock_score from evaluator
    results = evaluate_bias(data)
    print('Bias Evaluation Test Results:')
    print('Overall bias score:', results['bias_score'])
    print('Category breakdown:', results['category_breakdown'])

if __name__ == '__main__':
    test_bias_evaluation()
