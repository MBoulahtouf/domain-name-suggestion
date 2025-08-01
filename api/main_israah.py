#!/usr/bin/env python3
"""
FastAPI implementation for the domain suggestion service using the IsraaH model.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import threading
import time

app = FastAPI(title="Domain Suggestion API", version="1.0.0")

# Global variables for model and tokenizer
model = None
tokenizer = None
model_lock = threading.Lock()

# Safety keywords for content filtering
INAPPROPRIATE_KEYWORDS = [
    'adult', 'nude', 'explicit', 'porn', 'sex', 'xxx', 'casino', 'gambling',
    'weapon', 'drug', 'viagra', 'pharmacy', 'illegal', 'hate', 'racist',
    'offensive', 'profanity', 'obscene', 'vulgar', 'indecent', 'violence'
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

def initialize_model():
    """
    Initialize the IsraaH domain name suggestion model.
    """
    global model, tokenizer
    
    model_id = "IsraaH/domain-name-suggestion-generator"
    
    try:
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def generate_domain_suggestions_with_model(business_description: str, num_suggestions: int = 5) -> List[dict]:
    """
    Generate domain suggestions using the IsraaH model.
    
    Args:
        business_description (str): Description of the business
        num_suggestions (int): Number of suggestions to generate
        
    Returns:
        List[dict]: List of domain suggestions with confidence scores
    """
    global model, tokenizer
    
    if model is None or tokenizer is None:
        # Fallback to simple generation if model not loaded
        return generate_simple_suggestions(business_description, num_suggestions)
    
    try:
        # Create a conversation with the business description
        messages = [
            {
                "role": "user", 
                "content": f"Generate {num_suggestions} domain name suggestions for: {business_description}. "
                          f"Only return domain names with common extensions like .com, .net, .org, .io. "
                          f"Format each suggestion as 'domain.com (confidence: 0.XX)' on a new line."
            }
        ]
        
        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        # Decode the response
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        # Parse the response into structured format
        suggestions = parse_suggestions(response)
        
        # Filter out any inappropriate suggestions
        filtered_suggestions = []
        for suggestion in suggestions:
            if not is_inappropriate(suggestion.get('domain', '')):
                filtered_suggestions.append(suggestion)
        
        return filtered_suggestions
        
    except Exception as e:
        print(f"Error generating suggestions with model: {e}")
        # Fallback to simple generation
        return generate_simple_suggestions(business_description, num_suggestions)

def parse_suggestions(response_text):
    """
    Parse the model response into structured suggestions.
    """
    suggestions = []
    lines = response_text.strip().split('\n')
    
    for line in lines:
        # Look for patterns like "domain.com (confidence: 0.95)"
        match = re.search(r'([\w-]+\.[\w.]+)(?:\s+\(confidence:\s*(0\.\d+)\))?', line)
        if match:
            domain = match.group(1)
            confidence = float(match.group(2)) if match.group(2) else 0.85  # Default confidence
            suggestions.append({
                "domain": domain,
                "confidence": confidence
            })
    
    return suggestions

def generate_simple_suggestions(business_description: str, num_suggestions: int = 3) -> List[dict]:
    """
    Simple fallback domain suggestion generator.
    
    Args:
        business_description (str): Description of the business
        num_suggestions (int): Number of suggestions to generate
        
    Returns:
        List[dict]: List of domain suggestions with confidence scores
    """
    import random
    
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

# Define request and response models
class DomainRequest(BaseModel):
    business_description: str
    num_suggestions: Optional[int] = 5

class DomainSuggestion(BaseModel):
    domain: str
    confidence: float

class DomainResponse(BaseModel):
    suggestions: List[DomainSuggestion]
    status: str
    message: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """
    Initialize the model when the API starts up.
    """
    print("Initializing model on startup...")
    # Initialize model in background to not block startup
    threading.Thread(target=initialize_model, daemon=True).start()

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Domain Suggestion API",
        "version": "1.0.0",
        "description": "Generate domain name suggestions based on business descriptions using the IsraaH model"
    }

@app.post("/suggest", response_model=DomainResponse)
async def suggest_domains(request: DomainRequest):
    """
    Generate domain suggestions for a business description using the IsraaH model.
    
    Args:
        request (DomainRequest): Business description and number of suggestions
        
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
        suggestions = generate_domain_suggestions_with_model(
            request.business_description, 
            request.num_suggestions
        )
        
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