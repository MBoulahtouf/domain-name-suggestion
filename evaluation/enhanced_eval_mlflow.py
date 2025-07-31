#!/usr/bin/env python3
"""
Enhanced evaluation script with MLflow tracking for the domain suggestion model.
"""

import json
import os
import sys
import mlflow
import pandas as pd
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def load_data(filepath):
    """Load JSON data from file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_metrics(predictions, ground_truth):
    """
    Calculate evaluation metrics.
    
    Args:
        predictions (list): Model predictions
        ground_truth (list): Ground truth labels
        
    Returns:
        dict: Evaluation metrics
    """
    # For this task, we'll calculate some basic metrics
    # This is a simplified approach since exact matching is difficult for generative tasks
    
    # Calculate average confidence
    avg_confidence = sum(pred['confidence'] for pred in predictions) / len(predictions)
    
    # Calculate domain diversity (unique domains / total domains)
    unique_domains = len(set(pred['domain'] for pred in predictions))
    domain_diversity = unique_domains / len(predictions)
    
    return {
        'average_confidence': avg_confidence,
        'domain_diversity': domain_diversity,
        'total_suggestions': len(predictions),
        'unique_domains': unique_domains
    }

def evaluate_model_performance(eval_data):
    """
    Evaluate model performance on the evaluation dataset.
    
    Args:
        eval_data (list): Evaluation data
        
    Returns:
        dict: Performance metrics
    """
    all_suggestions = []
    
    for sample in eval_data:
        suggestions = sample['suggestions']
        all_suggestions.extend(suggestions)
    
    # Calculate metrics
    metrics = calculate_metrics(all_suggestions, all_suggestions)  # Self-reference for this example
    
    # Additional analysis
    # Domain length analysis
    domain_lengths = [len(suggestion['domain']) for suggestion in all_suggestions]
    avg_domain_length = sum(domain_lengths) / len(domain_lengths)
    
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
    print("Running enhanced evaluation with MLflow tracking...")
    
    # Start MLflow run
    mlflow.set_tracking_uri("file:///tmp/mlflow-tracking")
    mlflow.set_experiment("domain_suggestion_enhanced_evaluation")
    
    with mlflow.start_run(run_name="enhanced_comprehensive_evaluation"):
        # Log parameters
        mlflow.log_params({
            "evaluation_script": "enhanced_eval_with_mlflow",
            "dataset_type": "synthetic",
            "evaluation_approach": "comprehensive_metrics"
        })
        
        # Load evaluation data
        eval_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'eval_data.json')
        if not os.path.exists(eval_data_path):
            print(f"Error: Evaluation data not found at {eval_data_path}")
            return
        
        eval_data = load_data(eval_data_path)
        print(f"Loaded {len(eval_data)} evaluation samples.")
        
        # Log dataset size as parameter
        mlflow.log_param("eval_samples", len(eval_data))
        
        # Run evaluation
        performance_metrics = evaluate_model_performance(eval_data)
        
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
        
        results_path = os.path.join(results_dir, 'enhanced_comprehensive_evaluation.json')
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