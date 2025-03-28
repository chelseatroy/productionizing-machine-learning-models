#!/bin/bash

set -e

echo "Creating dataset cache folder: ./hf_cache"
mkdir -p hf_cache

echo "Downloading IMDB dataset into hf_cache..."
python3 -c "
from datasets import load_dataset
load_dataset('imdb', cache_dir='./hf_cache')
"
echo "Dataset downloaded."

echo "🛠️ Updating Dockerfiles..."

# Patch dashboard Dockerfile
cat > dashboard/Dockerfile <<EOF
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

COPY dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard/ .
COPY hf_cache /root/.cache/huggingface/datasets

EXPOSE 8501
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
EOF

# Patch metaflow Dockerfile
cat > metaflow_pipeline/Dockerfile <<EOF
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

COPY metaflow_pipeline/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY metaflow_pipeline/ .
COPY hf_cache /root/.cache/huggingface/datasets
EOF

echo "Dockerfiles updated."

echo "Rebuilding Docker images..."
docker compose build

echo "Build complete. Run with: docker compose up"
