#!/usr/bin/env python3
"""
MLflow experiment tracking for domain suggestion model evaluation.
"""

import mlflow
import mlflow.pytorch
import json
import os
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

# Set MLflow tracking URI (using local filesystem for simplicity)
mlflow.set_tracking_uri("file:///tmp/mlflow-tracking")

class MLflowEvaluator:
    """Evaluator using MLflow for experiment tracking."""
    
    def __init__(self, experiment_name="domain_suggestion_evaluation"):
        """
        Initialize MLflow evaluator.
        
        Args:
            experiment_name (str): Name of the MLflow experiment
        """
        self.experiment_name = experiment_name
        self.run_id = None
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
    
    def start_run(self, run_name=None):
        """
        Start a new MLflow run.
        
        Args:
            run_name (str): Name for this run
        """
        active_run = mlflow.start_run(run_name=run_name)
        self.run_id = active_run.info.run_id
        print(f"Started MLflow run: {self.run_id}")
        return active_run
    
    def log_params(self, params):
        """
        Log parameters to MLflow.
        
        Args:
            params (dict): Dictionary of parameters to log
        """
        mlflow.log_params(params)
    
    def log_metrics(self, metrics):
        """
        Log metrics to MLflow.
        
        Args:
            metrics (dict): Dictionary of metrics to log
        """
        mlflow.log_metrics(metrics)
    
    def log_artifacts(self, artifacts_dir):
        """
        Log artifacts to MLflow.
        
        Args:
            artifacts_dir (str): Directory containing artifacts to log
        """
        if os.path.exists(artifacts_dir):
            mlflow.log_artifacts(artifacts_dir)
    
    def log_model(self, model, model_name="domain_suggestion_model"):
        """
        Log model to MLflow.
        
        Args:
            model: PyTorch model to log
            model_name (str): Name for the model
        """
        # Log model as PyTorch
        mlflow.pytorch.log_model(model, model_name)
    
    def log_evaluation_results(self, eval_results):
        """
        Log comprehensive evaluation results to MLflow.
        
        Args:
            eval_results (dict): Evaluation results to log
        """
        # Log main metrics
        main_metrics = {
            "average_confidence": eval_results.get("average_confidence", 0),
            "domain_diversity": eval_results.get("domain_diversity", 0),
            "average_domain_length": eval_results.get("average_domain_length", 0),
            "total_suggestions": eval_results.get("total_suggestions", 0),
            "unique_domains": eval_results.get("unique_domains", 0)
        }
        
        mlflow.log_metrics(main_metrics)
        
        # Log distribution metrics
        extension_dist = eval_results.get("extension_distribution", {})
        for ext, count in extension_dist.items():
            # Clean extension name for MLflow (remove dots)
            clean_ext = ext.replace(".", "dot_")
            mlflow.log_metric(f"extension_{clean_ext}", count)
        
        confidence_dist = eval_results.get("confidence_distribution", {})
        for range_name, count in confidence_dist.items():
            # Clean range name for MLflow
            clean_range = range_name.replace("<", "lt_").replace("-", "_").replace(".", "_")
            mlflow.log_metric(f"confidence_{clean_range}", count)
    
    def end_run(self):
        """End the current MLflow run."""
        if self.run_id:
            mlflow.end_run()
            print(f"Ended MLflow run: {self.run_id}")
            self.run_id = None

def load_evaluation_data():
    """Load evaluation data from JSON file."""
    eval_file = os.path.join(os.path.dirname(__file__), 'comprehensive_evaluation.json')
    if os.path.exists(eval_file):
        with open(eval_file, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: Evaluation data not found at {eval_file}")
        return {}

def main():
    """Main function to run MLflow evaluation tracking."""
    print("Starting MLflow evaluation tracking...")
    
    # Initialize evaluator
    evaluator = MLflowEvaluator("domain_suggestion_model_evaluation")
    
    # Start a new run
    with evaluator.start_run(run_name="baseline_evaluation"):
        # Log parameters
        params = {
            "model_type": "gpt2_baseline",
            "dataset_size": 1000,
            "train_samples": 800,
            "eval_samples": 200,
            "evaluation_approach": "comprehensive_metrics"
        }
        evaluator.log_params(params)
        
        # Load evaluation results
        eval_results = load_evaluation_data()
        
        if eval_results:
            # Log evaluation results
            evaluator.log_evaluation_results(eval_results)
            
            # Log the evaluation results file as an artifact
            eval_file = os.path.join(os.path.dirname(__file__), 'evaluation', 'comprehensive_evaluation.json')
            if os.path.exists(eval_file):
                mlflow.log_artifact(eval_file)
            
            # Log a summary markdown file
            summary_content = f"""
# Domain Suggestion Model Evaluation Summary

## Overall Metrics
- Average Confidence: {eval_results.get('average_confidence', 0):.3f}
- Domain Diversity: {eval_results.get('domain_diversity', 0):.3f}
- Average Domain Length: {eval_results.get('average_domain_length', 0):.2f}
- Total Suggestions: {eval_results.get('total_suggestions', 0)}
- Unique Domains: {eval_results.get('unique_domains', 0)}

## Extension Distribution
{chr(10).join([f"- {ext}: {count}" for ext, count in eval_results.get('extension_distribution', {}).items()])}

## Confidence Distribution
{chr(10).join([f"- {range_name}: {count}" for range_name, count in eval_results.get('confidence_distribution', {}).items()])}
"""
            
            # Write summary to file and log as artifact
            summary_file = "/tmp/evaluation_summary.md"
            with open(summary_file, "w") as f:
                f.write(summary_content)
            mlflow.log_artifact(summary_file)
            os.remove(summary_file)  # Clean up temporary file
            
            print("Logged evaluation results to MLflow:")
            print(f"  - Parameters: {len(params)}")
            print(f"  - Metrics: {5 + len(eval_results.get('extension_distribution', {})) + len(eval_results.get('confidence_distribution', {}))}")
            print(f"  - Artifacts: 2")
        else:
            print("No evaluation data found. Run evaluation/comprehensive_eval.py first.")
    
    print(f"\nMLflow tracking completed.")
    print(f"View results with: mlflow ui --backend-store-uri file:///tmp/mlflow-tracking")
    print(f"Or check the tracking directory: /tmp/mlflow-tracking")

if __name__ == "__main__":
    main()