# Deploying the Qwen Domain Suggestion API

This guide explains how to deploy the Qwen domain suggestion API using various methods.

## Prerequisites

Before deploying, ensure you have:
1. Python 3.7 or higher
2. pip package manager
3. Docker (for containerized deployment)
4. A Qwen API key

## Local Deployment

### Option 1: Using the Run Script (Recommended)

1. Make the script executable:
   ```bash
   chmod +x run_api_qwen.sh
   ```

2. Set your Qwen API key:
   ```bash
   export QWEN_API_KEY=your_api_key_here
   ```

3. Run the API:
   ```bash
   ./run_api_qwen.sh
   ```

The script will automatically:
- Create a virtual environment
- Install dependencies
- Start the API server

### Option 2: Manual Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv venv_qwen
   source venv_qwen/bin/activate  # On Windows: venv_qwen\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r api/requirements_qwen.txt
   ```

3. Set your Qwen API key:
   ```bash
   export QWEN_API_KEY=your_api_key_here
   ```

4. Run the API:
   ```bash
   uvicorn api.main_qwen:app --host 0.0.0.0 --port 8000
   ```

## Docker Deployment

### Building the Docker Image

1. Build the Docker image:
   ```bash
   docker build -t domain-suggestion-api-qwen -f api/Dockerfile.qwen .
   ```

2. Run the container (make sure to pass the API key):
   ```bash
   docker run -e QWEN_API_KEY=your_api_key_here -p 8000:8000 domain-suggestion-api-qwen
   ```

## Cloud Deployment

### Deploying to AWS EC2

1. Launch an EC2 instance with Ubuntu
2. SSH into the instance
3. Install dependencies:
   ```bash
   sudo apt update
   sudo apt install python3-pip git -y
   ```
4. Clone your repository:
   ```bash
   git clone <your-repository-url>
   cd <repository-directory>
   ```
5. Set your API key:
   ```bash
   export QWEN_API_KEY=your_api_key_here
   ```
6. Run the API:
   ```bash
   ./run_api_qwen.sh
   ```

### Deploying to Google Cloud Run

1. Build and push the Docker image to Google Container Registry:
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT-ID/domain-suggestion-api-qwen
   ```

2. Deploy to Cloud Run (make sure to set the API key as an environment variable):
   ```bash
   gcloud run deploy --image gcr.io/PROJECT-ID/domain-suggestion-api-qwen --platform managed --set-env-vars QWEN_API_KEY=your_api_key_here
   ```

### Deploying to Heroku

1. Create a Heroku app:
   ```bash
   heroku create your-app-name
   ```

2. Set the stack to container:
   ```bash
   heroku stack:set container
   ```

3. Set the API key as a config var:
   ```bash
   heroku config:set QWEN_API_KEY=your_api_key_here
   ```

4. Deploy:
   ```bash
   git push heroku main
   ```

## API Usage

Once deployed, you can interact with the API using curl or any HTTP client:

### Generate Domain Suggestions
```bash
curl -X POST http://localhost:8000/suggest \
  -H "Content-Type: application/json" \
  -d '{"business_description": "organic coffee shop in downtown area"}'
```

### Check API Health
```bash
curl http://localhost:8000/
```

### Access API Documentation
Open your browser and navigate to:
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)

## Environment Variables

The API supports the following environment variables:

- `PORT`: Port to run the API on (default: 8000)
- `HOST`: Host to bind the API to (default: 0.0.0.0)
- `QWEN_API_KEY`: Your Qwen API key (required)

Example with environment variables:
```bash
PORT=8080 HOST=127.0.0.1 QWEN_API_KEY=your_api_key_here uvicorn api.main_qwen:app --host $HOST --port $PORT
```

## Monitoring and Logging

The API includes basic logging. For production deployments, consider:

1. Setting up centralized logging (e.g., ELK stack, Datadog)
2. Implementing health checks
3. Adding monitoring for response times and error rates
4. Setting up alerts for critical issues

## Performance Considerations

1. **API Latency**: API calls to Qwen may have variable latency depending on network conditions and Qwen's load.

2. **Rate Limiting**: Be aware of Qwen's rate limits and implement appropriate handling in your application.

3. **Fallback Mechanisms**: The implementation includes fallback mechanisms for when the API is unavailable.

4. **Caching**: For frequently requested business descriptions, consider implementing caching.

## Security Considerations

1. **Content Filtering**: The API includes built-in content filtering for inappropriate content.

2. **Input Validation**: All inputs are validated using Pydantic models.

3. **API Keys**: Keep your Qwen API key secure and do not commit it to version control.

4. **HTTPS**: Always use HTTPS in production environments.

## Troubleshooting

### API Key Issues

If the API fails to start or generate suggestions:
1. Verify your Qwen API key is set correctly
2. Check that your API key has the necessary permissions
3. Ensure your API key hasn't expired

### Network Issues

If you encounter network errors:
1. Check your internet connection
2. Verify that you can reach the Qwen API endpoint
3. Check if there are any firewall rules blocking the connection

### Slow Response Times

If responses are slow:
1. Network latency to the Qwen API
2. Qwen's current load
3. Complexity of the business description

### API Not Starting

If the API fails to start:
1. Check that all dependencies are installed
2. Verify the Python version is compatible
3. Check for syntax errors in the code
4. Ensure the required ports are available