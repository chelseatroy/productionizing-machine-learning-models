# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install system-level dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Hugging Face cache into image
COPY ../hf_cache /root/.cache/huggingface/datasets

# Copy your source code into the container
COPY dashboard/ .

# Expose the default Streamlit port
EXPOSE 8501

# Default command to run the dashboard
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
