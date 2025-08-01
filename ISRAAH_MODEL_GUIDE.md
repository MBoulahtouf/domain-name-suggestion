# Using the IsraaH Domain Name Suggestion Model

This guide explains how to use the specialized `IsraaH/domain-name-suggestion-generator` model with the notebooks provided.

## About the Model

The `IsraaH/domain-name-suggestion-generator` is a specialized model already fine-tuned for generating domain name suggestions based on business descriptions. This eliminates the need for fine-tuning and provides better results out of the box compared to general-purpose models.

Model details:
- Architecture: LLaMA
- Task: Text generation
- Language: English
- License: Apache 2.0

## Available Notebooks

1. `04_israah_domain_suggestion_model.ipynb`: Direct model loading approach
2. `05_israah_domain_suggestion_model_chat_template.ipynb`: Chat template approach (recommended)
3. `06_domain_suggestion_evaluation_and_api.ipynb`: Systematic evaluation, edge case testing, and API implementation

## Recommended Workflow for Assignment

For your specific assignment requirements, we recommend using the `06_domain_suggestion_evaluation_and_api.ipynb` notebook which is designed specifically for:

1. **Systematic Evaluation**: Running the model on test data and analyzing results
2. **Edge Case Discovery**: Testing with inappropriate content and unusual inputs
3. **Safety Filtering**: Implementing content filtering for inappropriate requests
4. **Model Improvement**: Iterating on prompts to improve output quality
5. **API Implementation**: Creating a deployable API endpoint

## Steps to Run (Evaluation Notebook)

1. Upload the `06_domain_suggestion_evaluation_and_api.ipynb` notebook to Google Colab
2. Make sure to select a GPU runtime:
   - Go to Runtime -> Change runtime type -> Hardware accelerator -> GPU
3. Run the cells in order:
   - First cell installs the required dependencies
   - Second cell loads the specialized model
   - Third cell defines the domain suggestion function with safety filtering
   - Fourth cell sets up the systematic evaluation framework
   - Fifth cell defines edge case testing
   - Sixth cell runs evaluation and edge case testing
   - Seventh cell implements model improvement cycles
   - Eighth cell shows API implementation
   - Ninth cell exports results for further analysis
   - Tenth cell provides conclusion and next steps

## Expected Results

The notebook will:
1. Evaluate the model on your test dataset
2. Test edge cases including inappropriate content
3. Implement safety filtering to block inappropriate requests
4. Show how to improve the model through prompt engineering
5. Provide a complete API implementation ready for deployment

## Key Features for Assignment Requirements

### Systematic Evaluation
- Runs the model on your test data
- Analyzes performance metrics
- Compares generated suggestions with expected results

### Edge Case Discovery
- Tests with inappropriate content to ensure safety filtering works
- Tests with unusual inputs (empty strings, very long descriptions, etc.)
- Validates model robustness

### Safety Filtering
- Implements keyword-based filtering for inappropriate content
- Blocks requests containing inappropriate business descriptions
- Filters out inappropriate domain suggestions

### Model Improvement Cycles
- Demonstrates prompt engineering techniques
- Shows how to iterate on prompts to improve output quality
- Provides a framework for A/B testing different approaches

### API Deployment
- Complete Flask API implementation
- Health check endpoint
- Domain suggestion endpoint with proper error handling
- Ready for cloud deployment

## Customization for Your Assignment

You can customize the notebook for your specific requirements:

1. **Evaluation Data**: Modify the evaluation to use your specific test dataset
2. **Safety Keywords**: Update the `INAPPROPRIATE_KEYWORDS` list with terms relevant to your use case
3. **Prompt Engineering**: Experiment with different prompts to improve suggestion quality
4. **API Endpoints**: Add additional endpoints or modify existing ones for your needs
5. **Metrics**: Add additional evaluation metrics that are important for your assignment

## Deployment Instructions

To deploy the API:

1. Save the API code to a file (e.g., `api.py`)
2. Install required dependencies: `pip install flask torch transformers accelerate`
3. Run the API: `python api.py`
4. Test with curl or requests:
   ```bash
   curl -X POST http://localhost:8000/suggest \
     -H "Content-Type: application/json" \
     -d '{"business_description": "coffee shop"}'
   ```

For cloud deployment, you can use platforms like:
- AWS EC2 or Lambda
- Google Cloud Run or Compute Engine
- Azure App Service or Functions
- Heroku
- Railway or Render

## Troubleshooting

1. If you get a "CUDA out of memory" error:
   - Reduce the `max_new_tokens` parameter
   - Use a smaller model variant if available
   - Restart the runtime and try again

2. If the model loading fails:
   - Check your internet connection
   - Make sure you have enough disk space
   - Restart the runtime and try again

3. If generation is slow:
   - The first generation is always slower due to compilation
   - Subsequent generations should be faster