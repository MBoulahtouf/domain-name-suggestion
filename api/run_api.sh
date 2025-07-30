#!/bin/bash
# Script to start the Domain Suggestion API

echo "Starting Domain Suggestion API..."

# Check if virtual environment exists, if not create it
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Start the API
echo "Starting API server on http://localhost:8000"
echo "API Documentation available at http://localhost:8000/docs"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload