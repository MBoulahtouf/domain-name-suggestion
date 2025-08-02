# Using Qwen for Domain Name Generation

This guide explains how to use the Qwen API for domain name generation with an open-source LLM as the judge.

## Model Architecture

1. **Domain Name Generator**: Qwen API (third-party model)
2. **LLM-as-a-Judge**: Open-source LLM (e.g., TinyLlama, Mistral)

## Setup Instructions

1. **Get Qwen API Key**:
   - Sign up at [Qwen's website](https://dashscope.aliyuncs.com/)
   - Obtain your API key

2. **Set Environment Variable**:
   ```bash
   export QWEN_API_KEY=your_api_key_here
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r api/requirements_qwen.txt
   ```

## Running the System

### 1. Using the Python Module

```python
from src.qwen_domain_model import QwenDomainSuggestionModel

# Initialize the model
model = QwenDomainSuggestionModel()

# Generate domain suggestions
suggestions = model.generate_suggestions("organic coffee shop in downtown area", num_suggestions=3)
for suggestion in suggestions:
    print(f"{suggestion['domain']} (confidence: {suggestion['confidence']:.2f})")
```

### 2. Using the API

Start the API server:
```bash
uvicorn api.main_qwen:app --host 0.0.0.0 --port 8000
```

Make a request:
```bash
curl -X POST http://localhost:8000/suggest \
  -H "Content-Type: application/json" \
  -d '{"business_description": "organic coffee shop in downtown area"}'
```

### 3. Running Evaluation

Evaluate the Qwen model with an open-source LLM as judge:
```bash
python evaluation/evaluate_qwen_with_open_source_judge.py
```

## API Specifications

### Request
```json
{
  "business_description": "organic coffee shop in downtown area"
}
```

### Response
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

Request with inappropriate content:
```json
{
  "business_description": "adult content website with explicit nude content"
}
```

Response:
```json
{
  "suggestions": [],
  "status": "blocked",
  "message": "Request contains inappropriate content"
}
```

## Advantages of This Approach

1. **High-Quality Generation**: Qwen is a powerful language model that produces high-quality domain suggestions
2. **Cost-Effective Evaluation**: Using an open-source LLM as judge reduces costs compared to using Qwen for evaluation
3. **Flexibility**: Easy to swap different open-source models as the judge
4. **Safety**: Built-in content filtering for inappropriate requests

## Limitations

1. **API Dependency**: Requires a stable internet connection and valid API key
2. **Cost**: API calls may incur costs depending on usage
3. **Rate Limiting**: May be subject to rate limits imposed by the API provider