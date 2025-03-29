import streamlit as st
import boto3
import joblib
import json
import os
import tempfile
import matplotlib.pyplot as plt
from botocore.exceptions import ClientError

# --- CONFIG ---
S3_BUCKET = "productionizing-ml-models-bucket"

st.set_page_config(page_title="📚 Sentiment Model Comparator", layout="wide")
st.title("📚 Book Review Sentiment Dashboard")
st.markdown("Compare multiple model versions, side-by-side.")

s3 = boto3.client("s3")

# --- Helpers ---
def list_versions():
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix="models/")
    folders = set()
    for item in response.get("Contents", []):
        parts = item["Key"].split("/")
        if len(parts) >= 2:
            folders.add(parts[1])
    return sorted(folders, reverse=True)

def download_file_from_s3(version: str, filename: str):
    key = f"models/{version}/{filename}"
    tmp_path = os.path.join(tempfile.gettempdir(), f"{version}_{filename}")
    try:
        s3.download_file(S3_BUCKET, key, tmp_path)
        return tmp_path
    except ClientError as e:
        st.error(f"Missing file in S3: {key}")
        return None

def load_model_set(version):
    model = joblib.load(download_file_from_s3(version, "model.joblib"))
    vectorizer = joblib.load(download_file_from_s3(version, "vectorizer.joblib"))
    with open(download_file_from_s3(version, "metrics.json")) as f:
        metrics = json.load(f)
    matrix_path = download_file_from_s3(version, "confusion_matrix.png")
    return model, vectorizer, metrics, matrix_path

# --- UI: Select Versions ---
available_versions = list_versions()

if len(available_versions) < 2:
    st.warning("Need at least 2 model versions to compare.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    version_a = st.selectbox("Model A Version", available_versions, key="a")
with col2:
    version_b = st.selectbox("Model B Version", available_versions, key="b", index=1 if len(available_versions) > 1 else 0)

# --- Load both model versions ---
model_a, vec_a, metrics_a, cm_a = load_model_set(version_a)
model_b, vec_b, metrics_b, cm_b = load_model_set(version_b)

# --- Metrics ---
st.subheader("📊 Evaluation Metrics")
met_col1, met_col2 = st.columns(2)
with met_col1:
    st.markdown(f"#### Version A: `{version_a}`")
    st.json(metrics_a)
with met_col2:
    st.markdown(f"#### Version B: `{version_b}`")
    st.json(metrics_b)

# --- Confusion Matrices ---
st.subheader("📉 Confusion Matrices")
conf_col1, conf_col2 = st.columns(2)
with conf_col1:
    if cm_a:
        st.image(cm_a, caption=f"Confusion Matrix A - {version_a}")
with conf_col2:
    if cm_b:
        st.image(cm_b, caption=f"Confusion Matrix B - {version_b}")

# --- A/B Prediction ---
st.subheader("🧪 A/B Prediction Test")
text_input = st.text_area("Enter a book review to test both models:")

if text_input:
    pred_a = model_a.predict(vec_a.transform([text_input]))[0]
    pred_b = model_b.predict(vec_b.transform([text_input]))[0]

    pred_label = lambda p: "👍 Positive" if p == 1 else "👎 Negative"
    ab_col1, ab_col2 = st.columns(2)
    with ab_col1:
        st.markdown(f"### Model A Prediction: {pred_label(pred_a)}")
    with ab_col2:
        st.markdown(f"### Model B Prediction: {pred_label(pred_b)}")
