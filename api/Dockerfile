# Dockerfile for the API service
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy API requirements
COPY api/requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy API files
COPY api/ .

# Expose port
EXPOSE 8000

# Create a non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]