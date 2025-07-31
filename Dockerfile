# Dockerfile for the main domain suggestion project
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Expose port for API (even though this image primarily runs scripts)
EXPOSE 8000

# Create a non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Default command (can be overridden)
CMD ["/bin/bash"]