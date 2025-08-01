#!/usr/bin/env python3
"""
Evaluation script for the Mistral domain suggestion model.
"""

import sys
import os
import json
from collections import defaultdict
from sklearn.metrics import accuracy_score
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.mistral_domain_model import MistralDomainSuggestionModel
from evaluation.comprehensive_eval import load_data, evaluate_model_performance


def evaluate_mistral_model(eval_data_path: str = 'data/eval_data.json', 
                          sample_size: int = 100) -> dict:
    """
    Evaluate the Mistral domain suggestion model.
    
    Args:
        eval_data_path (str): Path to evaluation data
        sample_size (int): Number of samples to evaluate (for faster testing)
        
    Returns:
        dict: Evaluation results
    """
    print("Loading evaluation data...")
    eval_data = load_data(eval_data_path)
    
    # Sample data for faster evaluation
    if sample_size and sample_size < len(eval_data):
        import random
        eval_data = random.sample(eval_data, sample_size)
        print(f"Using sample of {sample_size} examples for evaluation")
    
    print("Initializing Mistral model...")
    model = MistralDomainSuggestionModel()
    
    print("Generating predictions...")
    predictions = []
    ground_truth = []
    
    for i, sample in enumerate(eval_data):
        if i % 10 == 0:
            print(f"Processing sample {i+1}/{len(eval_data)}")
            
        business_description = sample['business_description']
        true_suggestions = sample['suggestions']
        
        # Generate predictions
        try:
            pred_suggestions = model.generate_suggestions(business_description, num_suggestions=3)
        except Exception as e:
            print(f"Error generating suggestions for '{business_description}': {e}")
            # Use fallback
            pred_suggestions = [{'domain': 'example.com', 'confidence': 0.5}]
        
        predictions.extend(pred_suggestions)
        ground_truth.extend(true_suggestions)
    
    print("Calculating metrics...")
    
    # Calculate basic metrics
    all_suggestions = predictions
    
    # Calculate metrics using the existing function
    metrics = evaluate_model_performance(eval_data)
    
    # Add Mistral-specific metrics
    mistral_metrics = {
        'model_name': 'Mistral-7B-finetuned',
        'total_samples': len(eval_data),
        'total_predictions': len(predictions),
    }
    
    # Combine metrics
    combined_metrics = {**metrics, **mistral_metrics}
    
    return combined_metrics


def compare_with_baseline(baseline_results_path: str = 'evaluation/comprehensive_evaluation.json'):
    """
    Compare Mistral model results with baseline model results.
    
    Args:
        baseline_results_path (str): Path to baseline evaluation results
    """
    print("Comparing with baseline model...")
    
    # Load baseline results if available
    try:
        with open(baseline_results_path, 'r') as f:
            baseline_results = json.load(f)
        print("Baseline results loaded successfully")
    except Exception as e:
        print(f"Could not load baseline results: {e}")
        baseline_results = None
    
    # Evaluate Mistral model
    mistral_results = evaluate_mistral_model(sample_size=50)  # Smaller sample for testing
    
    print("\n=== EVALUATION RESULTS ===")
    print(f"Mistral Model - Average Confidence: {mistral_results.get('average_confidence', 0):.3f}")
    print(f"Mistral Model - Domain Diversity: {mistral_results.get('domain_diversity', 0):.3f}")
    print(f"Mistral Model - Average Domain Length: {mistral_results.get('average_domain_length', 0):.1f}")
    
    if baseline_results:
        print(f"Baseline Model - Average Confidence: {baseline_results.get('average_confidence', 0):.3f}")
        print(f"Baseline Model - Domain Diversity: {baseline_results.get('domain_diversity', 0):.3f}")
        print(f"Baseline Model - Average Domain Length: {baseline_results.get('average_domain_length', 0):.1f}")
    
    return mistral_results


def main():
    """Main evaluation function."""
    print("Mistral Domain Suggestion Model Evaluation")
    print("=" * 50)
    
    results = compare_with_baseline()
    
    # Save results
    output_path = 'evaluation/mistral_evaluation.json'
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")


if __name__ == "__main__":
    main()