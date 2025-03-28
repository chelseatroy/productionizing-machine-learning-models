Write-Host "📦 Creating hf_cache directory..."
New-Item -ItemType Directory -Force -Path "hf_cache"

Write-Host "📥 Downloading IMDB dataset with Python..."
python -c "from datasets import load_dataset; load_dataset('imdb', cache_dir='./hf_cache')"

Write-Host "🛠 Writing Dockerfiles..."

Set-Content -Path "dashboard\Dockerfile" -Value @"
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*
COPY dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY dashboard/ .
COPY hf_cache /root/.cache/huggingface/datasets
EXPOSE 8501
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
"@

Set-Content -Path "metaflow_pipeline\Dockerfile" -Value @"
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*
COPY metaflow_pipeline/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY metaflow_pipeline/ .
COPY hf_cache /root/.cache/huggingface/datasets
"@

Write-Host "🔨 Building Docker images..."
docker compose build

Write-Host "✅ Build complete. Run the project with: docker compose up"
