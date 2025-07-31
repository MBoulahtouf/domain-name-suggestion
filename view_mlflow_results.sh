#!/bin/bash
# Script to start MLflow UI for viewing experiment results

echo "Starting MLflow UI..."
echo "Open your browser and go to: http://localhost:5000"

# Start MLflow UI
mlflow ui --backend-store-uri file:///tmp/mlflow-tracking --port 5000