import streamlit as st
import boto3
import joblib
import tempfile
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG ---
S3_BUCKET = "productionizing-ml-models-bucket"
FILES = {
    "model": "model.joblib",
    "vectorizer": "vectorizer.joblib",
    "metrics": "metrics.json",
    "confusion": "confusion_matrix.png"
}

st.set_page_config(page_title="📚 Sentiment Dashboard", layout="centered")
st.title("📚 Book Review Sentiment Dashboard")

# --- Download files from S3 ---
@st.cache_resource
def download_from_s3():
    s3 = boto3.client("s3")
    local_files = {}
    for name, key in FILES.items():
        tmp_path = os.path.join(tempfile.gettempdir(), key)
        s3.download_file(S3_BUCKET, key, tmp_path)
        local_files[name] = tmp_path
    return local_files

# Load everything
files = download_from_s3()
model = joblib.load(files["model"])
vectorizer = joblib.load(files["vectorizer"])

# --- Metrics Section ---
st.subheader("📊 Evaluation Metrics")
with open(files["metrics"]) as f:
    metrics = json.load(f)
st.json(metrics)

# --- Confusion Matrix ---
st.subheader("📉 Confusion Matrix")
try:
    st.image(files["confusion"], caption="Validation Set Confusion Matrix")
except Exception as e:
    st.warning(f"Could not load confusion matrix image: {e}")

# --- Live Prediction ---
st.subheader("🧪 Try it yourself!")
review = st.text_area("Enter a book review:")

if review:
    try:
        prediction = model.predict(vectorizer.transform([review]))
        label = "👍 Positive" if prediction[0] == 1 else "👎 Negative"
        st.markdown(f"### Prediction: {label}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
