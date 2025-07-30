#!/usr/bin/env python3
"""
Evaluation framework using LLM-as-a-judge for domain suggestions.
"""

import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from rouge_score import rouge_scorer
import random

class LLMEvaluator:
    """Evaluator using LLM-as-a-judge approach."""
    
    def __init__(self, model_name='gpt2'):
        """
        Initialize the evaluator.
        
        Args:
            model_name (str): Name of the pre-trained model to use as judge
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Add padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        # ROUGE scorer for additional metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    def evaluate_relevance(self, business_description, domain_suggestion):
        """
        Evaluate the relevance of a domain suggestion using the LLM-as-judge approach.
        
        Args:
            business_description (str): Business description
            domain_suggestion (str): Domain suggestion to evaluate
            
        Returns:
            dict: Evaluation scores
        """
        # Create evaluation prompt
        prompt = f"""
        Evaluate the relevance of a domain name suggestion for a business.
        
        Business Description: {business_description}
        Domain Suggestion: {domain_suggestion}
        
        On a scale of 1-10, how relevant is the domain name to the business description?
        Consider factors like:
        - Does the domain reflect the business type?
        - Does it include relevant keywords?
        - Is it memorable and professional?
        
        Score:"""
        
        # Tokenize the prompt
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        # Generate response (limit to a few tokens)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=5,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and extract score
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Simple extraction of numerical score (this is a basic approach)
        score_text = generated_text.replace(prompt, "").strip()
        
        # Try to extract a numerical score
        relevance_score = 5.0  # Default score
        for word in score_text.split():
            try:
                # Try to parse a number from the response
                num = float(''.join(c for c in word if c.isdigit() or c == '.'))
                if 1 <= num <= 10:
                    relevance_score = num
                    break
            except ValueError:
                continue
        
        # Normalize to 0-1 scale
        normalized_relevance = relevance_score / 10.0
        
        # Calculate ROUGE score between domain and description
        rouge_scores = self.rouge_scorer.score(business_description.lower(), domain_suggestion.lower())
        rouge_l = rouge_scores['rougeL'].fmeasure
        
        return {
            'relevance_score': normalized_relevance,
            'rouge_l': rouge_l,
            'judge_response': score_text[:100] + "..." if len(score_text) > 100 else score_text
        }
    
    def evaluate_suggestions_batch(self, eval_data):
        """
        Evaluate a batch of domain suggestions.
        
        Args:
            eval_data (list): List of evaluation samples
            
        Returns:
            dict: Overall evaluation metrics
        """
        results = []
        
        for sample in eval_data:
            business_description = sample['business_description']
            suggestions = sample['suggestions']
            
            sample_results = []
            for suggestion in suggestions:
                domain = suggestion['domain']
                confidence = suggestion['confidence']
                
                # Evaluate relevance
                eval_result = self.evaluate_relevance(business_description, domain)
                eval_result['domain'] = domain
                eval_result['confidence'] = confidence
                
                sample_results.append(eval_result)
            
            results.append({
                'business_description': business_description,
                'evaluations': sample_results
            })
        
        # Calculate overall metrics
        total_relevance = 0
        total_rouge = 0
        total_count = 0
        
        for result in results:
            for evaluation in result['evaluations']:
                total_relevance += evaluation['relevance_score']
                total_rouge += evaluation['rouge_l']
                total_count += 1
        
        avg_relevance = total_relevance / total_count if total_count > 0 else 0
        avg_rouge = total_rouge / total_count if total_count > 0 else 0
        
        return {
            'results': results,
            'average_relevance': avg_relevance,
            'average_rouge_l': avg_rouge,
            'total_evaluations': total_count
        }

def main():
    """Main function to run evaluation."""
    print("Initializing LLM evaluator...")
    evaluator = LLMEvaluator()
    
    # Load evaluation data
    with open('../data/eval_data.json', 'r') as f:
        eval_data = json.load(f)
    
    print(f"Loaded {len(eval_data)} evaluation samples.")
    
    # For demonstration, let's evaluate a smaller subset
    subset_data = eval_data[:10]  # First 10 samples
    
    print("Running evaluation...")
    evaluation_results = evaluator.evaluate_suggestions_batch(subset_data)
    
    print(f"\nEvaluation Results:")
    print(f"Average Relevance Score: {evaluation_results['average_relevance']:.3f}")
    print(f"Average ROUGE-L Score: {evaluation_results['average_rouge_l']:.3f}")
    print(f"Total Evaluations: {evaluation_results['total_evaluations']}")
    
    # Save results
    with open('../evaluation/llm_judge_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print("\nSample detailed results:")
    for i, result in enumerate(evaluation_results['results'][:3]):
        print(f"\n{i+1}. Business: {result['business_description']}")
        for j, eval_result in enumerate(result['evaluations']):
            print(f"   {j+1}. {eval_result['domain']} - Relevance: {eval_result['relevance_score']:.2f}, "
                  f"ROUGE-L: {eval_result['rouge_l']:.2f}")

if __name__ == "__main__":
    main()