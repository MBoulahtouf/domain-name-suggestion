# Domain Suggestion LLM - Quick Start Guide

This document provides instructions on how to set up and run the Domain Suggestion LLM system.

## Project Structure

```
domain_suggestion_llm/
├── data/                 # Generated datasets
├── notebooks/            # Jupyter notebooks for analysis and demos
├── src/                  # Source code for data generation and models
├── tests/                # Integration tests
├── evaluation/           # Evaluation framework and results
├── api/                  # API implementation
├── README.md             # Project overview
├── requirements.txt      # Project dependencies
└── technical_report.md   # Technical report
```

## Setup Instructions

1. **Create a virtual environment and install dependencies:**
   ```bash
   cd domain_suggestion_llm
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Generate synthetic data:**
   ```bash
   python src/generate_data.py
   ```
   This will create training and evaluation datasets in the `data/` directory.

3. **Run the evaluation:**
   ```bash
   python evaluation/comprehensive_eval.py
   ```
   This will evaluate the synthetic dataset and save results in `evaluation/comprehensive_evaluation.json`.

## Running the API

1. **Navigate to the API directory:**
   ```bash
   cd api
   ```

2. **Install API dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API server:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
   
   Or use the provided script:
   ```bash
   ./run_api.sh
   ```

4. **Access the API:**
   - API documentation: http://localhost:8000/docs
   - Main endpoint: http://localhost:8000/suggest

## API Usage Examples

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

For inappropriate content, the API will block the request:

#### Request
```json
{
  "business_description": "adult content website with explicit nude content"
}
```

#### Response
```json
{
  "suggestions": [],
  "status": "blocked",
  "message": "Request contains inappropriate content"
}
```

## Jupyter Notebooks

The project includes Jupyter notebooks for data analysis and API demonstration:

1. `notebooks/01_data_analysis.ipynb` - Data analysis and visualization
2. `notebooks/02_api_demo.ipynb` - API usage demonstration

To run the notebooks:
```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Start Jupyter
jupyter notebook
```

## Testing

Run the integration tests to verify that all components are working correctly:
```bash
python tests/test_integration.py
```

## Technical Report

A comprehensive technical report is available in `technical_report.md`, which includes:
- Project overview and problem statement
- Dataset creation methodology
- Model development approach
- Evaluation framework and results
- Safety considerations
- API implementation details
- Conclusion and future improvements