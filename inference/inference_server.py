from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os
import pickle

app = FastAPI(title="Book Review Sentiment API")
app.current_model_version = os.getenv("MODEL_VERSION", "latest")

s3_bucket = os.getenv("S3_BUCKET")

class ReviewInput(BaseModel):
    text: str

# Load model on startup
@app.on_event("startup")
def load_model():
    load_model_from_cloud(app.current_model_version)

def load_model_from_cloud(version):
    global model, vectorizer
    print(f"Inference server currently fetching MODEL VERSION: {version}")
    s3 = boto3.client("s3")
    s3.download_file(
        "productionizing-ml-models-bucket",
        f"models/{version}/vectorizer.pkl",
        "downloaded_vectorizer.pkl"
    )
    s3.download_file(
        "productionizing-ml-models-bucket",
        f"models/{version}/idf.pkl",
        "downloaded_idf.pkl"
    )
    s3.download_file(
        "productionizing-ml-models-bucket",
        f"models/{version}/model.joblib",
        "downloaded_model.joblib"
    )
    vectorizer = pickle.load(open("downloaded_vectorizer.pkl", "rb"))
    idf = pickle.load(open("downloaded_idf.pkl", "rb"))
    vectorizer.idf_ = idf
    model = joblib.load("downloaded_model.joblib")


@app.get("/versions")
def list_model_versions():
    try:
        s3 = boto3.client("s3")
        response = s3.list_objects_v2(Bucket=s3_bucket, Prefix="models/")
        folders = set()
        for item in response.get("Contents", []):
            parts = item["Key"].split("/")
            if len(parts) >= 2:
                folders.add(parts[1])
        return {"versions": sorted(folders, reverse=True)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict_sentiment(review: ReviewInput):
    try:
        vectorized = vectorizer.transform([review.text])
        prediction = model.predict(vectorized)[0]
        return {
            "prediction": "positive" if prediction == 1 else "negative",
            "raw": int(prediction),
            "version": app.current_model_version
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload-model/{version}")
def reload_model(version: str):
    try:
        load_model_from_cloud(version)
        app.current_model_version = version
        return {"message": f"Model reloaded: {version}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")
