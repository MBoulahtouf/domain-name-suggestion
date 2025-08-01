#!/usr/bin/env python3
"""
Test script for the Mistral domain suggestion model.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.mistral_domain_model import MistralDomainSuggestionModel


def test_model():
    """Test the Mistral domain suggestion model."""
    print("Testing domain suggestion model...")
    
    # Initialize the model
    try:
        # Use a smaller, open-source model that doesn't require authentication
        model = MistralDomainSuggestionModel("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test cases
    test_cases = [
        "organic coffee shop in downtown area",
        "tech startup that builds AI-powered project management tools",
        "eco-friendly fitness center for yoga and meditation",
        "vintage travel agency specializing in European destinations",
        "local bakery specializing in gluten-free products"
    ]
    
    print("\nGenerating domain suggestions...")
    for i, description in enumerate(test_cases, 1):
        print(f"\n{i}. Business: {description}")
        try:
            suggestions = model.generate_suggestions(description, num_suggestions=3)
            print("   Suggestions:")
            for j, suggestion in enumerate(suggestions, 1):
                print(f"     {j}. {suggestion['domain']} (confidence: {suggestion['confidence']:.2f})")
        except Exception as e:
            print(f"   Error generating suggestions: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_model()