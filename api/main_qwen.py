#!/usr/bin/env python3
"""
FastAPI implementation for the domain suggestion service using the Qwen model.
"""

import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.qwen_domain_model import QwenDomainSuggestionModel


# Pydantic models for request and response
class DomainSuggestionRequest(BaseModel):
    business_description: str


class DomainSuggestion(BaseModel):
    domain: str
    confidence: float


class DomainSuggestionResponse(BaseModel):
    suggestions: List[DomainSuggestion]
    status: str
    message: str = None


# Initialize FastAPI app
app = FastAPI(
    title="Domain Suggestion API",
    description="API for generating domain name suggestions based on business descriptions using Qwen",
    version="1.0.0"
)

# Initialize the Qwen model
model = None


@app.on_event("startup")
async def startup_event():
    """Initialize the model when the app starts."""
    global model
    try:
        model = QwenDomainSuggestionModel()
        print("Qwen domain suggestion model loaded successfully!")
    except Exception as e:
        print(f"Error loading Qwen model: {e}")
        model = None


@app.get("/", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Domain suggestion API is running"}


@app.post("/suggest", response_model=DomainSuggestionResponse, tags=["Domain Suggestion"])
async def suggest_domains(request: DomainSuggestionRequest):
    """
    Generate domain name suggestions based on business descriptions using the Qwen model.
    
    Args:
        request: Business description for which to generate domain suggestions
        
    Returns:
        Domain suggestions with confidence scores
    """
    global model
    
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not initialized. Please check server logs."
        )
    
    business_description = request.business_description.strip()
    
    # Safety check for inappropriate content
    inappropriate_keywords = ["adult", "explicit", "nude", "porn", "casino", "gambling"]
    if any(keyword in business_description.lower() for keyword in inappropriate_keywords):
        return DomainSuggestionResponse(
            suggestions=[],
            status="blocked",
            message="Request contains inappropriate content"
        )
    
    # Handle empty input
    if not business_description:
        return DomainSuggestionResponse(
            suggestions=[],
            status="error",
            message="Business description cannot be empty"
        )
    
    try:
        # Generate domain suggestions
        suggestions = model.generate_suggestions(business_description, num_suggestions=3)
        
        # Convert to response format
        formatted_suggestions = [
            DomainSuggestion(domain=suggestion["domain"], confidence=suggestion["confidence"])
            for suggestion in suggestions
        ]
        
        return DomainSuggestionResponse(
            suggestions=formatted_suggestions,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating domain suggestions: {str(e)}"
        )


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "main_qwen:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "false").lower() == "true"
    )