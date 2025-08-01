#!/usr/bin/env python3
"""
Evaluation script for the TinyLlama domain suggestion model with MLflow tracking.
"""

import sys
import os
import json
from collections import defaultdict
import mlflow
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.mistral_domain_model import MistralDomainSuggestionModel


def load_data(filepath):
    """Load JSON data from file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_metrics(predictions):
    """
    Calculate evaluation metrics.
    
    Args:
        predictions (list): Model predictions
        
    Returns:
        dict: Evaluation metrics
    """
    # Calculate average confidence
    avg_confidence = sum(pred['confidence'] for pred in predictions) / len(predictions)
    
    # Calculate domain diversity (unique domains / total domains)
    unique_domains = len(set(pred['domain'] for pred in predictions))
    domain_diversity = unique_domains / len(predictions) if predictions else 0
    
    return {
        'average_confidence': avg_confidence,
        'domain_diversity': domain_diversity,
        'total_suggestions': len(predictions),
        'unique_domains': unique_domains
    }


def evaluate_model_performance(eval_data, model):
    """
    Evaluate model performance on the evaluation dataset.
    
    Args:
        eval_data (list): Evaluation data
        model: Domain suggestion model
        
    Returns:
        dict: Performance metrics
    """
    all_suggestions = []
    
    for i, sample in enumerate(eval_data):
        if i % 10 == 0:
            print(f"Processing sample {i+1}/{len(eval_data)}")
            
        business_description = sample['business_description']
        
        # Generate predictions
        try:
            suggestions = model.generate_suggestions(business_description, num_suggestions=3)
        except Exception as e:
            print(f"Error generating suggestions for '{business_description}': {e}")
            # Use fallback
            suggestions = [{'domain': 'example.com', 'confidence': 0.5}]
        
        all_suggestions.extend(suggestions)
    
    # Calculate metrics
    metrics = calculate_metrics(all_suggestions)
    
    # Additional analysis
    # Domain length analysis
    domain_lengths = [len(suggestion['domain']) for suggestion in all_suggestions]
    avg_domain_length = sum(domain_lengths) / len(domain_lengths) if domain_lengths else 0
    
    # Extension analysis
    extensions = defaultdict(int)
    for suggestion in all_suggestions:
        domain = suggestion['domain']
        if '.' in domain:
            ext = '.' + domain.split('.')[-1]
            extensions[ext] += 1
    
    # Confidence distribution
    confidence_ranges = defaultdict(int)
    for suggestion in all_suggestions:
        confidence = suggestion['confidence']
        if confidence >= 0.9:
            confidence_ranges['0.9-1.0'] += 1
        elif confidence >= 0.8:
            confidence_ranges['0.8-0.9'] += 1
        elif confidence >= 0.7:
            confidence_ranges['0.7-0.8'] += 1
        else:
            confidence_ranges['<0.7'] += 1
    
    return {
        **metrics,
        'average_domain_length': avg_domain_length,
        'extension_distribution': dict(extensions),
        'confidence_distribution': dict(confidence_ranges)
    }


def main():
    """Main evaluation function with MLflow tracking."""
    print("Running TinyLlama model evaluation with MLflow tracking...")
    
    # Start MLflow run
    mlflow.set_tracking_uri("file:///tmp/mlflow-tracking")
    mlflow.set_experiment("domain_suggestion_tinyllama_evaluation")
    
    with mlflow.start_run(run_name="tinyllama_comprehensive_evaluation"):
        # Log parameters
        mlflow.log_params({
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "evaluation_script": "tinyllama_eval_with_mlflow",
            "dataset_type": "synthetic",
            "evaluation_approach": "comprehensive_metrics",
            "num_suggestions_per_sample": 3
        })
        
        # Load evaluation data
        eval_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'eval_data.json')
        if not os.path.exists(eval_data_path):
            print(f"Error: Evaluation data not found at {eval_data_path}")
            return
        
        eval_data = load_data(eval_data_path)
        print(f"Loaded {len(eval_data)} evaluation samples.")
        
        # For faster evaluation, let's use a sample of the data
        sample_size = min(50, len(eval_data))  # Use 50 samples or all if less
        import random
        eval_data_sample = random.sample(eval_data, sample_size)
        print(f"Using sample of {sample_size} examples for evaluation")
        
        # Log dataset size as parameter
        mlflow.log_param("eval_samples", sample_size)
        
        # Initialize model
        print("Initializing TinyLlama model...")
        model = MistralDomainSuggestionModel("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        print("Model loaded successfully!")
        
        # Run evaluation
        performance_metrics = evaluate_model_performance(eval_data_sample, model)
        
        # Print results
        print("\n=== Evaluation Results ===")
        print(f"Total suggestions: {performance_metrics['total_suggestions']}")
        print(f"Unique domains: {performance_metrics['unique_domains']}")
        print(f"Average confidence: {performance_metrics['average_confidence']:.3f}")
        print(f"Domain diversity: {performance_metrics['domain_diversity']:.3f}")
        print(f"Average domain length: {performance_metrics['average_domain_length']:.2f}")
        
        print("\nExtension distribution:")
        for ext, count in performance_metrics['extension_distribution'].items():
            print(f"  {ext}: {count}")
        
        print("\nConfidence distribution:")
        for range_name, count in performance_metrics['confidence_distribution'].items():
            print(f"  {range_name}: {count}")
        
        # Log metrics to MLflow
        # Main metrics
        mlflow.log_metrics({
            "average_confidence": performance_metrics['average_confidence'],
            "domain_diversity": performance_metrics['domain_diversity'],
            "average_domain_length": performance_metrics['average_domain_length'],
            "total_suggestions": performance_metrics['total_suggestions'],
            "unique_domains": performance_metrics['unique_domains']
        })
        
        # Extension distribution metrics
        for ext, count in performance_metrics['extension_distribution'].items():
            # Clean extension name for MLflow (remove dots)
            clean_ext = ext.replace(".", "dot_")
            mlflow.log_metric(f"extension_{clean_ext}", count)
        
        # Confidence distribution metrics
        for range_name, count in performance_metrics['confidence_distribution'].items():
            # Clean range name for MLflow
            clean_range = range_name.replace("<", "lt_").replace("-", "_").replace(".", "_")
            mlflow.log_metric(f"confidence_{clean_range}", count)
        
        # Save results
        results_dir = os.path.join(os.path.dirname(__file__))
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = os.path.join(results_dir, 'tinyllama_comprehensive_evaluation.json')
        with open(results_path, 'w') as f:
            json.dump(performance_metrics, f, indent=2)
        
        # Log results file as artifact
        mlflow.log_artifact(results_path)
        
        # Create and log visualization data
        # Extension distribution dataframe
        ext_df = pd.DataFrame([
            {"extension": ext, "count": count} 
            for ext, count in performance_metrics['extension_distribution'].items()
        ])
        ext_csv_path = "/tmp/extension_distribution.csv"
        ext_df.to_csv(ext_csv_path, index=False)
        mlflow.log_artifact(ext_csv_path)
        os.remove(ext_csv_path)  # Clean up
        
        # Confidence distribution dataframe
        conf_df = pd.DataFrame([
            {"confidence_range": range_name, "count": count} 
            for range_name, count in performance_metrics['confidence_distribution'].items()
        ])
        conf_csv_path = "/tmp/confidence_distribution.csv"
        conf_df.to_csv(conf_csv_path, index=False)
        mlflow.log_artifact(conf_csv_path)
        os.remove(conf_csv_path)  # Clean up
        
        print(f"\nResults saved to {results_path}")
        print(f"\nMLflow run completed.")
        print(f"View results with: mlflow ui --backend-store-uri file:///tmp/mlflow-tracking")


if __name__ == "__main__":
    main()