FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your inference code
COPY inference/inference_server.py .

EXPOSE 8000

CMD ["uvicorn", "inference_server:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app"]