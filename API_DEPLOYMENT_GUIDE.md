# Deploying the IsraaH Domain Suggestion API

This guide explains how to deploy the IsraaH domain suggestion API using various methods.

## Prerequisites

Before deploying, ensure you have:
1. Python 3.7 or higher
2. pip package manager
3. Docker (for containerized deployment)
4. A machine with sufficient RAM (at least 8GB recommended)

## Local Deployment

### Option 1: Using the Run Script (Recommended)

1. Make the script executable:
   ```bash
   chmod +x run_api_israah.sh
   ```

2. Run the API:
   ```bash
   ./run_api_israah.sh
   ```

The script will automatically:
- Create a virtual environment
- Install dependencies
- Start the API server

### Option 2: Manual Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r api/requirements_israah.txt
   ```

3. Run the API:
   ```bash
   uvicorn api.main_israah:app --host 0.0.0.0 --port 8000
   ```

## Docker Deployment

### Building the Docker Image

1. Build the Docker image:
   ```bash
   docker build -t domain-suggestion-api -f api/Dockerfile.israah .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 domain-suggestion-api
   ```

### Using Docker Compose

If you have docker-compose installed:
```bash
docker-compose up --build
```

## Cloud Deployment

### Deploying to AWS EC2

1. Launch an EC2 instance with Ubuntu and at least 8GB RAM
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
5. Run the API:
   ```bash
   ./run_api_israah.sh
   ```

### Deploying to Google Cloud Run

1. Build and push the Docker image to Google Container Registry:
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT-ID/domain-suggestion-api
   ```

2. Deploy to Cloud Run:
   ```bash
   gcloud run deploy --image gcr.io/PROJECT-ID/domain-suggestion-api --platform managed
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

3. Deploy:
   ```bash
   git push heroku main
   ```

## API Usage

Once deployed, you can interact with the API using curl or any HTTP client:

### Generate Domain Suggestions
```bash
curl -X POST http://localhost:8000/suggest \
  -H "Content-Type: application/json" \
  -d '{"business_description": "organic coffee shop in downtown area", "num_suggestions": 5}'
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

Example with environment variables:
```bash
PORT=8080 HOST=127.0.0.1 uvicorn api.main_israah:app --host $HOST --port $PORT
```

## Monitoring and Logging

The API includes basic logging. For production deployments, consider:

1. Setting up centralized logging (e.g., ELK stack, Datadog)
2. Implementing health checks
3. Adding monitoring for response times and error rates
4. Setting up alerts for critical issues

## Performance Considerations

1. **Memory Usage**: The IsraaH model requires significant RAM. Ensure your deployment environment has at least 8GB RAM.

2. **Cold Starts**: The first request after startup may be slow as the model loads. Consider warming up the API with a test request.

3. **Concurrency**: The API uses threading to handle the model loading, but the model itself may not be highly concurrent. For high-traffic applications, consider:
   - Using multiple instances behind a load balancer
   - Implementing request queuing
   - Caching frequent requests

4. **Rate Limiting**: For production use, implement rate limiting to prevent abuse.

## Security Considerations

1. **Content Filtering**: The API includes built-in content filtering for inappropriate content.

2. **Input Validation**: All inputs are validated using Pydantic models.

3. **API Keys**: For production, consider adding API key authentication.

4. **HTTPS**: Always use HTTPS in production environments.

## Troubleshooting

### Model Loading Issues

If the model fails to load:
1. Check internet connectivity
2. Verify sufficient disk space is available
3. Ensure adequate RAM is available
4. Check the model repository is accessible

### Slow Response Times

If responses are slow:
1. The first request is always slower due to model compilation
2. Subsequent requests should be faster
3. Consider the complexity of the business description
4. Check if the system is under heavy load

### Memory Errors

If you encounter memory errors:
1. Ensure the system has at least 8GB RAM
2. Close other applications to free up memory
3. Consider using a smaller model variant if available

### API Not Starting

If the API fails to start:
1. Check that all dependencies are installed
2. Verify the Python version is compatible
3. Check for syntax errors in the code
4. Ensure the required ports are available