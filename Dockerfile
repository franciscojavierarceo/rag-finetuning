# Multi-stage Docker build for RAG Embedding Fine-tuning
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies using uv (much faster!)
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /workspace/tensorboard_logs \
             /workspace/feature_repo/data \
             /workspace/fine_tuned_kubeflow_embeddings

# Set environment variables for dynamic paths
ENV PROJECT_DIR=/workspace
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Default command
CMD ["python", "kubeflow_embedding_training.py"]

# --- Production stage ---
FROM base as production

# Copy only necessary files for production
COPY kubeflow_embedding_training.py .
COPY feature_repo/ ./feature_repo/
COPY requirements.txt .

# Set proper permissions
RUN chmod +x kubeflow_embedding_training.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; import sentence_transformers; print('Dependencies OK')" || exit 1

# Run as non-root user for security
RUN groupadd -r trainer && useradd -r -g trainer trainer
RUN chown -R trainer:trainer /workspace
USER trainer

# Expose port for TensorBoard (optional)
EXPOSE 6006

CMD ["python", "kubeflow_embedding_training.py"]