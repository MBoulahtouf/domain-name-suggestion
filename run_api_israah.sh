#!/bin/bash
# Script to run the IsraaH model API locally

echo "Starting Domain Suggestion API with IsraaH model..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r api/requirements_israah.txt
else
    echo "Using existing virtual environment..."
    source venv/bin/activate
fi

# Run the API
echo "Starting API server..."
echo "API will be available at http://localhost:8000"
echo "Documentation will be available at http://localhost:8000/docs"
uvicorn api.main_israah:app --host 0.0.0.0 --port 8000 --reload