from metaflow import FlowSpec, step, Parameter
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import boto3
import os
import uuid
from datasets import load_dataset

class SentimentAnalysisFlow(FlowSpec):

    s3_bucket = Parameter("s3-bucket", help="S3 bucket to store the model")

    @step
    def start(self):
        print("Fetching book reviews from IMDb...")
        self.raw_data = load_dataset("imdb", split="train", cache_dir="/root/.cache/huggingface/datasets").shuffle(seed=42).select(range(2000))
        self.test_data = load_dataset("imdb", split="test", cache_dir="/root/.cache/huggingface/datasets").shuffle(seed=42).select(range(500))
        self.next(self.prepare_data)

    @step
    def prepare_data(self):
        print("Preparing and cleaning data...")
        texts = [x['text'] for x in self.raw_data]
        labels = [x['label'] for x in self.raw_data]

        X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.X_train = self.vectorizer.fit_transform(X_train)
        self.X_val = self.vectorizer.transform(X_val)
        self.y_train = y_train
        self.y_val = y_val
        self.X_test = self.vectorizer.transform([x['text'] for x in self.test_data])
        self.y_test = [x['label'] for x in self.test_data]
        self.next(self.train)

    @step
    def train(self):
        print("Training model...")
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(self.X_train, self.y_train)
        self.next(self.validate)

    @step
    def validate(self):
        print("Validating model...")
        y_pred = self.model.predict(self.X_val)

        print("Classification Report:")
        print(classification_report(self.y_val, y_pred))

        cm = confusion_matrix(self.y_val, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("confusion_matrix.png")
        self.model_path = "model.joblib"
        joblib.dump(self.model, self.model_path)

        # Save vectorizer
        joblib.dump(self.vectorizer, "vectorizer.joblib")

        # Save metrics
        from sklearn.metrics import accuracy_score, f1_score
        import json

        metrics = {
            "accuracy": accuracy_score(self.y_val, y_pred),
            "f1_score": f1_score(self.y_val, y_pred),
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f)

        self.next(self.push)

    @step
    def push(self):
        print("Uploading model to S3...")
        s3 = boto3.client('s3')
        s3.upload_file(self.model_path, self.s3_bucket, self.model_path)
        self.model_url = f"https://{self.s3_bucket}.s3.amazonaws.com/{self.model_path}"

        for fname in ["model.joblib", "vectorizer.joblib", "confusion_matrix.png", "metrics.json"]:
            s3.upload_file(fname, self.s3_bucket, fname)

        self.next(self.end)

    @step
    def end(self):
        print("Flow completed!")
        print(f"Model available at: {self.model_url}")

if __name__ == '__main__':
    SentimentAnalysisFlow()

