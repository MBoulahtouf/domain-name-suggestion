#!/usr/bin/env python3
"""
Evaluation script for the Qwen domain suggestion model using an open-source LLM as judge.
"""

import sys
import os
import json
import random
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.qwen_domain_model import QwenDomainSuggestionModel


class OpenSourceLLMJudge:
    """Open-source LLM judge for evaluating domain suggestions."""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initialize the open-source LLM judge.
        
        Args:
            model_name (str): Name of the model to use as judge
        """
        print(f"Loading open-source LLM judge: {model_name}")
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Judge model loaded successfully!")
    
    def evaluate_suggestion(self, business_description: str, domain_suggestion: dict) -> dict:
        """
        Evaluate a domain suggestion using the open-source LLM judge.
        
        Args:
            business_description (str): Business description
            domain_suggestion (dict): Domain suggestion with 'domain' and 'confidence' keys
            
        Returns:
            dict: Evaluation scores for relevance, professionalism, and memorability
        """
        domain = domain_suggestion['domain']
        
        # Format prompt for evaluation
        prompt = f"""You are an expert evaluator of domain name suggestions.
        
Please evaluate the following domain name suggestion for the given business description.

Business Description: {business_description}
Domain Suggestion: {domain}

Evaluate the domain suggestion on these three dimensions:
1. Relevance: How well does the domain name relate to the business description? (1-10)
2. Professionalism: Does the domain name sound professional and appropriate for business use? (1-10)
3. Memorability: Is the domain name easy to remember and pronounce? (1-10)

Respond with ONLY a JSON object in this exact format:
{{
  "relevance_score": 7,
  "professionalism_score": 8,
  "memorability_score": 9
}}

No other text, just the JSON object."""

        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.3,
                    do_sample=False,  # Use greedy decoding for consistency
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                evaluation = json.loads(json_str)
                
                # Ensure scores are in valid range
                for key in ['relevance_score', 'professionalism_score', 'memorability_score']:
                    if key in evaluation:
                        evaluation[key] = max(1, min(10, evaluation[key]))
                
                return evaluation
            else:
                # Return default scores if JSON not found
                return {
                    "relevance_score": 5,
                    "professionalism_score": 5,
                    "memorability_score": 5
                }
                
        except Exception as e:
            print(f"Error evaluating suggestion: {e}")
            # Return default scores on error
            return {
                "relevance_score": 5,
                "professionalism_score": 5,
                "memorability_score": 5
            }


def load_data(filepath):
    """Load JSON data from file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_metrics(evaluations):
    """
    Calculate evaluation metrics from judge evaluations.
    
    Args:
        evaluations (list): List of evaluation results
        
    Returns:
        dict: Aggregated metrics
    """
    if not evaluations:
        return {}
    
    # Calculate average scores
    relevance_scores = [eval.get('relevance_score', 0) for eval in evaluations]
    professionalism_scores = [eval.get('professionalism_score', 0) for eval in evaluations]
    memorability_scores = [eval.get('memorability_score', 0) for eval in evaluations]
    
    avg_relevance = sum(relevance_scores) / len(relevance_scores)
    avg_professionalism = sum(professionalism_scores) / len(professionalism_scores)
    avg_memorability = sum(memorability_scores) / len(memorability_scores)
    
    # Calculate overall score (weighted average)
    overall_score = (
        0.5 * avg_relevance +
        0.3 * avg_professionalism +
        0.2 * avg_memorability
    ) / 10  # Normalize to 0-1 scale
    
    # Calculate other metrics
    total_suggestions = len(evaluations)
    
    return {
        'avg_relevance_score': avg_relevance,
        'avg_professionalism_score': avg_professionalism,
        'avg_memorability_score': avg_memorability,
        'avg_overall_score': overall_score,
        'total_suggestions': total_suggestions
    }


def evaluate_model_performance(eval_data, qwen_model, judge_model, sample_size=None):
    """
    Evaluate model performance on the evaluation dataset.
    
    Args:
        eval_data (list): Evaluation data
        qwen_model: Qwen domain suggestion model
        judge_model: Open-source LLM judge model
        sample_size (int): Number of samples to evaluate (None for all)
        
    Returns:
        dict: Performance metrics
    """
    if sample_size:
        eval_data = random.sample(eval_data, min(sample_size, len(eval_data)))
    
    evaluations = []
    all_suggestions = []
    
    for i, sample in enumerate(eval_data):
        print(f"Processing sample {i+1}/{len(eval_data)}")
        
        business_description = sample['business_description']
        
        # Generate suggestions with Qwen model
        try:
            suggestions = qwen_model.generate_suggestions(business_description, num_suggestions=3)
        except Exception as e:
            print(f"Error generating suggestions for '{business_description}': {e}")
            # Use fallback
            suggestions = [{'domain': 'example.com', 'confidence': 0.5}]
        
        all_suggestions.extend(suggestions)
        
        # Evaluate each suggestion with open-source LLM judge
        for suggestion in suggestions:
            try:
                evaluation = judge_model.evaluate_suggestion(business_description, suggestion)
                # Add context to evaluation
                evaluation_with_context = {
                    'business_description': business_description,
                    'domain_suggestion': suggestion['domain'],
                    'confidence': suggestion['confidence'],
                    **evaluation
                }
                evaluations.append(evaluation_with_context)
            except Exception as e:
                print(f"Error evaluating suggestion '{suggestion['domain']}': {e}")
    
    # Calculate metrics
    metrics = calculate_metrics(evaluations)
    
    # Add detailed results
    metrics['evaluations'] = evaluations
    metrics['all_suggestions'] = all_suggestions
    
    return metrics


def main():
    """Main evaluation function."""
    print("Running Qwen model evaluation with open-source LLM judge...")
    
    # Load evaluation data
    eval_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'eval_data.json')
    if not os.path.exists(eval_data_path):
        print(f"Error: Evaluation data not found at {eval_data_path}")
        # Create sample data for testing
        eval_data = [
            {"business_description": "tech startup building AI tools"},
            {"business_description": "organic bakery with fresh bread"},
            {"business_description": "fitness center with yoga classes"},
            {"business_description": "art gallery featuring local artists"},
            {"business_description": "eco-friendly cleaning service"}
        ]
        print("Using sample evaluation data for testing.")
    else:
        eval_data = load_data(eval_data_path)
        print(f"Loaded {len(eval_data)} evaluation samples.")
    
    # Use a smaller sample for faster evaluation
    sample_size = min(5, len(eval_data))
    print(f"Using sample of {sample_size} examples for evaluation")
    
    # Initialize Qwen model
    print("Initializing Qwen domain suggestion model...")
    try:
        qwen_model = QwenDomainSuggestionModel()
        print("Qwen model loaded successfully!")
        
        # Initialize open-source LLM judge
        print("Initializing open-source LLM judge...")
        judge_model = OpenSourceLLMJudge()
        print("Judge model loaded successfully!")
        
        # Run evaluation
        performance_metrics = evaluate_model_performance(eval_data, qwen_model, judge_model, sample_size)
        
        # Print results
        print("\n=== Evaluation Results ===")
        print(f"Total suggestions: {performance_metrics.get('total_suggestions', 0)}")
        print(f"Average relevance score: {performance_metrics.get('avg_relevance_score', 0):.2f}/10")
        print(f"Average professionalism score: {performance_metrics.get('avg_professionalism_score', 0):.2f}/10")
        print(f"Average memorability score: {performance_metrics.get('avg_memorability_score', 0):.2f}/10")
        print(f"Average overall score: {performance_metrics.get('avg_overall_score', 0):.3f}")
        
        # Save results
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = os.path.join(results_dir, 'qwen_with_open_source_judge_evaluation.json')
        with open(results_path, 'w') as f:
            json.dump(performance_metrics, f, indent=2)
        
        print(f"\nResults saved to {results_path}")
        
        return performance_metrics
        
    except Exception as e:
        print(f"Error initializing or running models: {e}")
        print("Make sure you have set the QWEN_API_KEY environment variable and have internet access.")
        return None


if __name__ == "__main__":
    results = main()