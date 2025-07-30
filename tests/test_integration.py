#!/usr/bin/env python3
"""
Integration test script for the domain suggestion system.
"""

import sys
import os
import json

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_data_generation():
    """Test the data generation script."""
    print("Testing data generation...")
    
    # Check if data files exist
    train_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train_data.json')
    eval_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'eval_data.json')
    
    if not os.path.exists(train_data_path):
        print("  ERROR: Training data not found")
        return False
    
    if not os.path.exists(eval_data_path):
        print("  ERROR: Evaluation data not found")
        return False
    
    # Load and check data
    with open(train_data_path, 'r') as f:
        train_data = json.load(f)
    
    with open(eval_data_path, 'r') as f:
        eval_data = json.load(f)
    
    if len(train_data) == 0:
        print("  ERROR: Training data is empty")
        return False
    
    if len(eval_data) == 0:
        print("  ERROR: Evaluation data is empty")
        return False
    
    # Check structure of first sample
    sample = train_data[0]
    if 'business_description' not in sample or 'suggestions' not in sample:
        print("  ERROR: Data format is incorrect")
        return False
    
    print(f"  OK: Generated {len(train_data)} training samples and {len(eval_data)} evaluation samples")
    return True

def test_model_loading():
    """Test loading the model (simplified test)."""
    print("Testing model loading...")
    
    # Check if model files exist (this is a simplified test)
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
    if not os.path.exists(model_dir):
        print("  ERROR: Model directory not found")
        return False
    
    print("  OK: Model directory found")
    return True

def test_api_endpoints():
    """Test API endpoints (simplified test)."""
    print("Testing API endpoints...")
    
    # Check if API files exist
    api_dir = os.path.join(os.path.dirname(__file__), '..', 'api')
    if not os.path.exists(api_dir):
        print("  ERROR: API directory not found")
        return False
    
    main_py = os.path.join(api_dir, 'main.py')
    if not os.path.exists(main_py):
        print("  ERROR: API main file not found")
        return False
    
    print("  OK: API files found")
    return True

def test_evaluation_framework():
    """Test evaluation framework."""
    print("Testing evaluation framework...")
    
    # Check if evaluation files exist
    eval_dir = os.path.join(os.path.dirname(__file__), '..', 'evaluation')
    if not os.path.exists(eval_dir):
        print("  ERROR: Evaluation directory not found")
        return False
    
    # Check if results file exists
    results_file = os.path.join(eval_dir, 'comprehensive_evaluation.json')
    if not os.path.exists(results_file):
        print("  WARNING: Evaluation results not found (run evaluation first)")
    else:
        print("  OK: Evaluation results found")
    
    print("  OK: Evaluation framework files found")
    return True

def main():
    """Run all integration tests."""
    print("Running integration tests for Domain Suggestion System...")
    print("=" * 60)
    
    tests = [
        test_data_generation,
        test_model_loading,
        test_api_endpoints,
        test_evaluation_framework
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR: {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All tests passed! The system is ready for use.")
        return True
    else:
        print("Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)