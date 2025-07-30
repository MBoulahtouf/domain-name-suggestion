#!/usr/bin/env python3
"""
FastAPI implementation for the domain suggestion service.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import re
import random

# Import our model (simplified for this example)
# In a real implementation, you would import the actual trained model
# from ..src.baseline_model import DomainSuggestionModel

app = FastAPI(title="Domain Suggestion API", version="1.0.0")

# Simplified model class for demonstration
class SimpleDomainModel:
    """A simple domain suggestion model for demonstration."""
    
    def generate_suggestions(self, business_description: str, num_suggestions: int = 3) -> List[dict]:
        """
        Generate domain suggestions for a business description.
        
        Args:
            business_description (str): Description of the business
            num_suggestions (int): Number of suggestions to generate
            
        Returns:
            List[dict]: List of domain suggestions with confidence scores
        """
        # Simple keyword extraction
        words = re.findall(r'\w+', business_description.lower())
        
        # Filter out common words
        keywords = [word for word in words if word not in [
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'
        ]]
        
        # Domain extensions
        extensions = ['.com', '.net', '.org', '.co', '.io']
        
        suggestions = []
        for i in range(num_suggestions):
            # Create a domain name from keywords
            if keywords:
                # Select 1-2 keywords for the domain
                selected_keywords = random.sample(keywords, min(len(keywords), random.randint(1, 2)))
                domain_base = ''.join(selected_keywords)
            else:
                # Fallback if no keywords found
                domain_base = 'business'
            
            # Add a random modifier sometimes
            if random.random() > 0.7:
                modifiers = ['pro', 'hub', 'zone', 'world', 'plus', 'max']
                domain_base += random.choice(modifiers)
            
            # Add extension
            extension = random.choice(extensions)
            domain = domain_base + extension
            
            # Confidence score
            confidence = round(random.uniform(0.6, 0.95), 2)
            
            suggestions.append({
                'domain': domain,
                'confidence': confidence
            })
        
        return suggestions

# Initialize the model
model = SimpleDomainModel()

# Define request and response models
class DomainRequest(BaseModel):
    business_description: str

class DomainSuggestion(BaseModel):
    domain: str
    confidence: float

class DomainResponse(BaseModel):
    suggestions: List[DomainSuggestion]
    status: str
    message: Optional[str] = None

# Safety keywords for content filtering
INAPPROPRIATE_KEYWORDS = [
    'adult', 'nude', 'explicit', 'porn', 'sex', 'xxx', 'casino', 'gambling',
    'weapon', 'drug', 'viagra', 'pharmacy', 'illegal'
]

def is_inappropriate(content: str) -> bool:
    """
    Check if content contains inappropriate keywords.
    
    Args:
        content (str): Content to check
        
    Returns:
        bool: True if content is inappropriate, False otherwise
    """
    content_lower = content.lower()
    return any(keyword in content_lower for keyword in INAPPROPRIATE_KEYWORDS)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Domain Suggestion API",
        "version": "1.0.0",
        "description": "Generate domain name suggestions based on business descriptions"
    }

@app.post("/suggest", response_model=DomainResponse)
async def suggest_domains(request: DomainRequest):
    """
    Generate domain suggestions for a business description.
    
    Args:
        request (DomainRequest): Business description
        
    Returns:
        DomainResponse: Domain suggestions with confidence scores
    """
    # Check for inappropriate content
    if is_inappropriate(request.business_description):
        return DomainResponse(
            suggestions=[],
            status="blocked",
            message="Request contains inappropriate content"
        )
    
    try:
        # Generate suggestions
        suggestions = model.generate_suggestions(request.business_description)
        
        return DomainResponse(
            suggestions=suggestions,
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating suggestions: {str(e)}")

# Example usage endpoints
@app.post("/suggest/example", response_model=DomainResponse)
async def suggest_domains_example():
    """Example endpoint with a sample business description."""
    example_request = DomainRequest(business_description="organic coffee shop in downtown area")
    return await suggest_domains(example_request)

@app.post("/suggest/safety_example", response_model=DomainResponse)
async def suggest_domains_safety_example():
    """Example endpoint with inappropriate content."""
    example_request = DomainRequest(business_description="adult content website with explicit nude content")
    return await suggest_domains(example_request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)