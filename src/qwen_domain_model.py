#!/usr/bin/env python3
"""
Qwen-based domain suggestion model implementation.
This model uses the Qwen API for domain name generation.
"""

import json
import random
import re
import os
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class QwenDomainSuggestionModel:
    """Domain suggestion model based on Qwen API."""
    
    def __init__(self, model: str = "qwen-max"):
        """
        Initialize the Qwen model.
        
        Args:
            model (str): Qwen model to use
        """
        self.model = model
        self.api_key = os.getenv("QWEN_API_KEY")
        
        if not self.api_key:
            raise ValueError("QWEN_API_KEY environment variable is required")
        
        print(f"Using Qwen model: {model}")
    
    def generate_suggestions(self, business_description: str, num_suggestions: int = 3) -> List[Dict[str, Any]]:
        """
        Generate domain suggestions for a business description using the Qwen API.
        
        Args:
            business_description (str): Description of the business
            num_suggestions (int): Number of suggestions to generate
            
        Returns:
            list: List of domain suggestions with confidence scores
        """
        # Format input prompt for the Qwen model
        prompt = f"""Generate exactly {num_suggestions} domain name suggestions for a business described as: {business_description}.
        
Requirements for each domain name:
1. Should be relevant to the business description
2. Should include a common top-level domain (.com, .net, .org, .io, .ai, .co)
3. Should be between 5-20 characters (excluding the domain extension)
4. Should be memorable and professional

Respond with ONLY a JSON array in this exact format:
[
  {{"domain": "example1.com", "confidence": 0.92}},
  {{"domain": "example2.net", "confidence": 0.87}},
  {{"domain": "example3.org", "confidence": 0.83}}
]

No other text, just the JSON array."""

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an expert at generating domain names for businesses."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            response = requests.post(
                "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["output"]["text"]
                
                # Try to parse JSON from response
                try:
                    # Extract JSON array from response
                    json_start = content.find('[')
                    json_end = content.rfind(']') + 1
                    
                    if json_start != -1 and json_end != -1:
                        json_str = content[json_start:json_end]
                        suggestions = json.loads(json_str)
                        
                        # Validate and clean suggestions
                        validated_suggestions = [
                            {
                                'domain': suggestion.get('domain', '').strip(),
                                'confidence': max(0.0, min(1.0, suggestion.get('confidence', 0.5)))
                            }
                            for suggestion in suggestions
                            if suggestion.get('domain', '').strip() and '.' in suggestion.get('domain', '').strip() and len(suggestion.get('domain', '').strip()) > 3
                        ]
                        
                        # If we don't have enough suggestions, generate fallbacks
                        while len(validated_suggestions) < num_suggestions:
                            fallback = self._generate_fallback_suggestion(business_description)
                            validated_suggestions.append(fallback)
                            
                        return validated_suggestions[:num_suggestions]
                    else:
                        # Fallback if JSON not found
                        return [self._generate_fallback_suggestion(business_description) for _ in range(num_suggestions)]
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    return [self._generate_fallback_suggestion(business_description) for _ in range(num_suggestions)]
            else:
                print(f"Qwen API call failed with status {response.status_code}: {response.text}")
                # Fallback to algorithmic generation
                return [self._generate_fallback_suggestion(business_description) for _ in range(num_suggestions)]
                
        except Exception as e:
            print(f"Error generating suggestions: {e}")
            # Fallback to algorithmic generation
            return [self._generate_fallback_suggestion(business_description) for _ in range(num_suggestions)]
    
    def _generate_fallback_suggestion(self, business_description: str) -> Dict[str, Any]:
        """
        Generate a fallback domain suggestion.
        
        Args:
            business_description (str): Description of the business
            
        Returns:
            dict: Domain suggestion with confidence score
        """
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
        
        # Select a base name and extension
        base = random.choice(base_names)
        ext = random.choice(extensions)
        domain = f"{base}{ext}"
        confidence = round(random.uniform(0.3, 0.7), 2)
        
        return {
            'domain': domain,
            'confidence': confidence
        }
    
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
    """Main function to test the Qwen domain suggestion model."""
    print("Initializing Qwen domain suggestion model...")
    
    try:
        model = QwenDomainSuggestionModel()
        
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
                
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("Make sure you have set the QWEN_API_KEY environment variable.")


if __name__ == "__main__":
    main()