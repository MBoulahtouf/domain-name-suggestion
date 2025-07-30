# Domain Suggestion LLM

This project implements a fine-tuned LLM for domain name suggestions based on business descriptions.

## Project Structure

- `data/`: Contains datasets for training and evaluation
- `notebooks/`: Jupyter notebooks for experiments and analysis
- `src/`: Source code for the model and utilities
- `tests/`: Unit tests for the code
- `evaluation/`: Evaluation framework and results
- `api/`: API implementation for serving the model

## Assignment Requirements

1. Create a synthetic dataset for domain name suggestions
2. Develop a baseline model and an improved model
3. Implement an LLM-as-a-judge evaluation framework
4. Create a technical report documenting the approach and results
5. (Optional) Deploy an API endpoint for the model

## API Specifications

### Input
```json
{
  "business_description": "organic coffee shop in downtown area"
}
```

### Output
```json
{
  "suggestions": [
    {"domain": "organicbeanscafe.com", "confidence": 0.92},
    {"domain": "downtowncoffee.org", "confidence": 0.87},
    {"domain": "freshbreworganic.net", "confidence": 0.83}
  ],
  "status": "success"
}
```

### Safety Example

For inappropriate content, the API should block the request:

#### Input
```json
{
  "business_description": "adult content website with explicit nude content"
}
```

#### Output
```json
{
  "suggestions": [],
  "status": "blocked",
  "message": "Request contains inappropriate content"
}
```