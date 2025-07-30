#!/usr/bin/env python3
"""
Script to generate synthetic data for domain name suggestions.
"""

import json
import random
from typing import List, Dict

# Sample business types and keywords
BUSINESS_TYPES = [
    "coffee shop", "bakery", "restaurant", "bookstore", "boutique",
    "fitness center", "spa", "salon", "law firm", "medical clinic",
    "tech startup", "consulting firm", "photography studio", "art gallery",
    "music store", "pet store", "hardware store", "grocery store",
    "car repair shop", "travel agency"
]

BUSINESS_ADJECTIVES = [
    "organic", "premium", "deluxe", "express", "quick", "fresh",
    "natural", "urban", "modern", "classic", "vintage", "eco-friendly",
    "luxury", "affordable", "local", "family-owned", "artisanal"
]

LOCATIONS = [
    "downtown", "midtown", "uptown", "suburban", "riverside", "hillside",
    " lakeside", "city center", "main street", "historic district",
    "business district", "shopping mall", "airport", "beachfront"
]

DOMAIN_EXTENSIONS = [".com", ".net", ".org", ".co", ".io", ".ai"]

def generate_business_description() -> str:
    """Generate a random business description."""
    business_type = random.choice(BUSINESS_TYPES)
    
    # 50% chance to add an adjective
    if random.random() > 0.5:
        adjective = random.choice(BUSINESS_ADJECTIVES)
        business_type = f"{adjective} {business_type}"
    
    # 30% chance to add a location
    location_desc = ""
    if random.random() > 0.7:
        location = random.choice(LOCATIONS)
        location_desc = f" in {location}"
    
    return f"{business_type}{location_desc}"

def generate_domain_name(description: str) -> str:
    """Generate a domain name based on the business description."""
    # Simplified approach: extract keywords and combine
    words = description.lower().replace(",", "").replace(".", "").split()
    
    # Remove common words that shouldn't be in domain names
    filtered_words = [word for word in words if word not in ["in", "the", "and", "with", "for"]]
    
    # Take 1-3 words for the domain name
    num_words = min(random.randint(1, 3), len(filtered_words))
    selected_words = random.sample(filtered_words, num_words)
    
    # Join words and add extension
    domain_base = "".join(selected_words)
    extension = random.choice(DOMAIN_EXTENSIONS)
    
    return f"{domain_base}{extension}"

def generate_sample_data(num_samples: int = 1000) -> List[Dict]:
    """Generate sample data for training/evaluation."""
    data = []
    
    for _ in range(num_samples):
        description = generate_business_description()
        # Generate 1-5 domain suggestions for each description
        num_suggestions = random.randint(1, 5)
        suggestions = []
        
        for _ in range(num_suggestions):
            domain = generate_domain_name(description)
            # Confidence score between 0.5 and 0.95
            confidence = round(random.uniform(0.5, 0.95), 2)
            suggestions.append({
                "domain": domain,
                "confidence": confidence
            })
        
        data.append({
            "business_description": description,
            "suggestions": suggestions
        })
    
    return data

def main():
    """Generate and save synthetic data."""
    print("Generating synthetic data...")
    
    # Generate training data
    train_data = generate_sample_data(800)
    with open("data/train_data.json", "w") as f:
        json.dump(train_data, f, indent=2)
    
    # Generate evaluation data
    eval_data = generate_sample_data(200)
    with open("data/eval_data.json", "w") as f:
        json.dump(eval_data, f, indent=2)
    
    print(f"Generated {len(train_data)} training samples and {len(eval_data)} evaluation samples.")
    print("Data saved to data/train_data.json and data/eval_data.json")

if __name__ == "__main__":
    main()