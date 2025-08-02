# Domain Suggestion LLM

This project implements a fine-tuned LLM for domain name suggestions based on business descriptions.

## Project Structure

- `data/`: Contains datasets for training and evaluation
- `notebooks/`: Jupyter notebooks for experiments and analysis
- `src/`: Source code for the model and utilities
- `tests/`: Unit tests for the code
- `evaluation/`: Evaluation framework and results
- `api/`: API implementation for serving the model
- `mlflow_data/`: MLflow tracking database and artifacts

## Notebooks

1. `01_data_analysis.ipynb`: Analysis of the training and evaluation datasets
2. `02_api_demo.ipynb`: Demonstration of the API functionality
3. `03_mlflow_demo.ipynb`: Demonstration of MLflow integration for experiment tracking
4. `04_mistral_7b_colab.ipynb`: Implementation of Mistral 7B model running on Google Colab with GPU (general-purpose approach)
5. `04_israah_domain_suggestion_model.ipynb`: Implementation of the specialized `IsraaH/domain-name-suggestion-generator` model (direct model loading)
6. `05_israah_domain_suggestion_model_chat_template.ipynb`: Implementation of the specialized `IsraaH/domain-name-suggestion-generator` model using chat template approach
7. `06_domain_suggestion_evaluation_and_api.ipynb`: Systematic evaluation, edge case testing, and API implementation for the domain suggestion model
8. `07_mistral_peft_domain_suggestion.ipynb`: Implementation of Mistral 7B model with PEFT (Parameter-Efficient Fine-Tuning) for domain name generation
9. `08_qwen_domain_suggestion.ipynb`: Implementation using Qwen API for domain name generation with open-source LLM as judge

## Assignment Requirements

1. Create a synthetic dataset for domain name suggestions
2. Develop a baseline model and an improved model
3. Implement an LLM-as-a-Judge evaluation framework
4. Create a technical report documenting the approach and results
5. (Optional) Deploy an API endpoint for the model

## MLflow Integration

This project uses MLflow for experiment tracking and model versioning. 

### Setup

1. Start the MLflow tracking server:
```bash
mlflow server --backend-store-uri sqlite:///mlflow_data/mlflow.db --default-artifact-root ./mlflow_data/mlruns --host 0.0.0.0 --port 5001 &
```

2. Run experiments using the provided scripts or notebooks:
```bash
# Run the domain suggestion experiment
python domain_suggestion_mlflow.py

# Or use the Jupyter notebook
jupyter notebook domain_suggestion_experiment.ipynb
```

3. View experiment results in the MLflow UI at http://localhost:5001

### Using MLflow in Your Project

For your domain suggestion project, you should:

1. **Track Model Versions**: Log each iteration of your fine-tuned LLM with MLflow
2. **Log Evaluation Metrics**: Use the LLM-as-a-Judge framework to evaluate domain suggestions and log the results
3. **Compare Experiments**: Use the MLflow UI to compare different model versions and hyperparameters
4. **Manage Datasets**: Log information about your synthetic datasets
5. **Document Edge Cases**: Log discovered edge cases and how your model handles them

See `MLFLOW_USAGE.md` for more detailed information on using MLflow with this project.

## API Implementation

The project includes two API implementations:

1. **Baseline API** (`api/main.py`): Simple rule-based implementation for demonstration
2. **IsraaH Model API** (`api/main_israah.py`): Implementation using the specialized `IsraaH/domain-name-suggestion-generator` model

### Running the IsraaH Model API

#### Option 1: Using the run script
```bash
./run_api_israah.sh
```

#### Option 2: Manual setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r api/requirements_israah.txt

# Run the API
uvicorn api.main_israah:app --host 0.0.0.0 --port 8000
```

#### Option 3: Using Docker
```bash
# Build the Docker image
docker build -t domain-suggestion-api -f api/Dockerfile.israah .

# Run the container
docker run -p 8000:8000 domain-suggestion-api
```

### API Endpoints

- `GET /`: Root endpoint with API information
- `POST /suggest`: Generate domain suggestions for a business description
- `POST /suggest/example`: Example endpoint with a sample business description
- `POST /suggest/safety_example`: Example endpoint with inappropriate content

### API Specifications

#### Input
```json
{
  "business_description": "organic coffee shop in downtown area",
  "num_suggestions": 5
}
```

#### Output
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

#### Safety Example

For inappropriate content, the API will block the request:

##### Input
```json
{
  "business_description": "adult content website with explicit nude content"
}
```

##### Output
```json
{
  "suggestions": [],
  "status": "blocked",
  "message": "Request contains inappropriate content"
}
```

## Model Information

The IsraaH domain name suggestion model is a specialized LLM already fine-tuned for generating domain name suggestions based on business descriptions.

Model details:
- Repository: https://huggingface.co/IsraaH/domain-name-suggestion-generator
- Architecture: LLaMA
- Task: Text generation
- Language: English
- License: Apache 2.0

## Evaluation

The model has been systematically evaluated using the provided test dataset and edge cases. Results are available in the `evaluation/` directory.

### Running All Experiments

To run all experiments and generate a consolidated evaluation report:

1. Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

2. Run the consolidated evaluation script:
```bash
python consolidated_evaluation.py
```

This will:
- Evaluate all implemented models (IsraaH, Mistral, and algorithmic baseline)
- Generate a comparison report with metrics for each model
- Save detailed results to `evaluation/consolidated_evaluation_report.json`

3. View experiment results in MLflow:
```bash
./view_mlflow_ui.sh
```
Then open http://localhost:5001 in your browser.