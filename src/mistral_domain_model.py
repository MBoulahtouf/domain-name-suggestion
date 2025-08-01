#!/usr/bin/env python3
"""
Mistral-based domain suggestion model implementation.
This model uses the pre-fine-tuned Mistral model for domain name generation.
"""

import json
import torch
import random
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from torch.utils.data import Dataset


class DomainSuggestionDataset(Dataset):
    """Custom dataset for domain suggestion inference."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        """
        Initialize the dataset.
        
        Args:
            data_path (str): Path to the JSON data file
            tokenizer: Tokenizer for the model
            max_length (int): Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        sample = self.data[idx]
        description = sample['business_description']
        
        # Format input text
        input_text = f"Generate a domain name for: {description}"
        
        # Tokenize input
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'business_description': description
        }


class MistralDomainSuggestionModel:
    """Domain suggestion model based on Mistral-7B fine-tuned for domain generation."""
    
    def __init__(self, model_name: str = "harshit2551/domain-name-generator-mistral7B-finetuned"):
        """
        Initialize the Mistral-based model.
        
        Args:
            model_name (str): Name of the pre-trained model to use
        """
        self.model_name = model_name
        print(f"Loading model: {model_name}")
        
        # Initialize using pipeline for easier inference
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        self.tokenizer = self.generator.tokenizer
    
    def generate_suggestions(self, business_description: str, num_suggestions: int = 3, 
                           max_length: int = 30, temperature: float = 0.7) -> List[Dict[str, Any]]:
        """
        Generate domain suggestions for a business description using the Mistral model.
        
        Args:
            business_description (str): Description of the business
            num_suggestions (int): Number of suggestions to generate
            max_length (int): Maximum length of generated domain name
            temperature (float): Sampling temperature (higher = more random)
            
        Returns:
            list: List of domain suggestions with confidence scores
        """
        # Format input prompt for the Mistral model
        prompt = f"Generate a domain name for: {business_description}"
        
        try:
            # Generate multiple suggestions
            outputs = self.generator(
                prompt,
                max_length=len(self.tokenizer.encode(prompt)) + max_length,
                num_return_sequences=num_suggestions,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            suggestions = []
            
            for output in outputs:
                # Extract the generated text
                generated_text = output['generated_text']
                
                # Post-process to extract domain-like text
                domain = self._extract_domain(generated_text, business_description)
                
                # Simple confidence score based on text quality
                confidence = self._calculate_confidence(domain, business_description)
                
                suggestions.append({
                    'domain': domain,
                    'confidence': confidence
                })
            
            return suggestions
            
        except Exception as e:
            print(f"Error generating suggestions: {e}")
            # Fallback to baseline suggestions
            return self._generate_fallback_suggestions(business_description, num_suggestions)
    
    def _extract_domain(self, generated_text: str, business_description: str) -> str:
        """
        Extract domain name from generated text.
        
        Args:
            generated_text (str): Full generated text
            business_description (str): Original business description
            
        Returns:
            str: Extracted domain name
        """
        # Remove the prompt part
        if "Generate a domain name for:" in generated_text:
            domain_part = generated_text.split("Generate a domain name for:")[-1]
        else:
            domain_part = generated_text
            
        # Simple extraction of domain-like text
        # Look for text that looks like a domain name
        import re
        
        # Try to find a domain pattern
        domain_pattern = r'\b([a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]*\.[a-zA-Z]{2,})\b'
        matches = re.findall(domain_pattern, domain_part)
        
        if matches:
            return matches[0]  # Return first match
        
        # If no domain pattern found, try to extract last alphanumeric sequence with dot
        words = domain_part.split()
        for word in reversed(words):
            if '.' in word and len(word) > 3:
                # Clean the word to make it domain-like
                cleaned = ''.join(c for c in word if c.isalnum() or c in '.-')
                if '.' in cleaned and len(cleaned) > 3:
                    return cleaned
        
        # Fallback: create a simple domain
        # Extract keywords from business description
        keywords = business_description.lower().split()[:3]
        base_name = ''.join([kw for kw in keywords if kw.isalnum()])[:15] or "business"
        
        # Add a common extension
        return f"{base_name}.com"
    
    def _calculate_confidence(self, domain: str, business_description: str) -> float:
        """
        Calculate a simple confidence score for a domain suggestion.
        
        Args:
            domain (str): Generated domain name
            business_description (str): Original business description
            
        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        # Basic checks for domain quality
        score = 0.5  # Base score
        
        # Check domain length (ideal: 5-20 characters)
        if 5 <= len(domain.split('.')[0]) <= 20:
            score += 0.2
            
        # Check if domain contains business keywords
        business_words = business_description.lower().split()
        domain_parts = domain.lower().split('.')[0]
        
        for word in business_words:
            if word[:3] in domain_parts:  # Match first 3 chars of each word
                score += 0.1
                break
                
        # Check for common good extensions
        good_extensions = ['.com', '.io', '.ai', '.co', '.org']
        for ext in good_extensions:
            if domain.endswith(ext):
                score += 0.1
                break
                
        # Check for special characters (fewer is better)
        special_chars = sum(1 for c in domain if not c.isalnum() and c != '.')
        if special_chars <= 1:
            score += 0.1
            
        return min(score, 1.0)  # Cap at 1.0
    
    def _generate_fallback_suggestions(self, business_description: str, num_suggestions: int) -> List[Dict[str, Any]]:
        """
        Generate fallback domain suggestions if the model fails.
        
        Args:
            business_description (str): Description of the business
            num_suggestions (int): Number of suggestions to generate
            
        Returns:
            list: List of fallback domain suggestions
        """
        suggestions = []
        words = business_description.lower().split()
        base_names = []
        
        # Create base names from business description
        if len(words) >= 2:
            base_names.append(''.join(words[:2])[:15])
            if len(words) >= 3:
                base_names.append(''.join(words[:3])[:15])
        
        base_names.append(''.join(words[:1])[:15] if words else "business")
        
        # Common extensions
        extensions = ['.com', '.io', '.ai', '.co', '.org', '.net']
        
        for i in range(num_suggestions):
            base = base_names[i % len(base_names)]
            ext = extensions[i % len(extensions)]
            domain = f"{base}{ext}"
            confidence = round(random.uniform(0.3, 0.7), 2)
            
            suggestions.append({
                'domain': domain,
                'confidence': confidence
            })
            
        return suggestions
    
    def batch_generate_suggestions(self, business_descriptions: List[str], 
                                 num_suggestions: int = 3) -> List[List[Dict[str, Any]]]:
        """
        Generate domain suggestions for multiple business descriptions.
        
        Args:
            business_descriptions (list): List of business descriptions
            num_suggestions (int): Number of suggestions to generate per description
            
        Returns:
            list: List of suggestion lists for each business description
        """
        results = []
        for description in business_descriptions:
            suggestions = self.generate_suggestions(description, num_suggestions)
            results.append(suggestions)
        return results


def main():
    """Main function to test the Mistral domain suggestion model."""
    print("Initializing Mistral domain suggestion model...")
    model = MistralDomainSuggestionModel()
    
    print("Testing with sample inputs...")
    
    # Test cases
    test_descriptions = [
        "organic coffee shop in downtown area",
        "tech startup that builds AI-powered project management tools",
        "eco-friendly fitness center for yoga and meditation",
        "vintage travel agency specializing in European destinations"
    ]
    
    for description in test_descriptions:
        print(f"\nBusiness: {description}")
        print("Suggestions:")
        suggestions = model.generate_suggestions(description, num_suggestions=3)
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion['domain']} (confidence: {suggestion['confidence']:.2f})")


if __name__ == "__main__":
    main()