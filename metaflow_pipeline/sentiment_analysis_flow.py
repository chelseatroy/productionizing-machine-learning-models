from metaflow import FlowSpec, step, Parameter
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import boto3
import os
import uuid
import json
import datetime

class SentimentAnalysisFlow(FlowSpec):
    s3_bucket = Parameter("s3_bucket", help="S3 bucket to store the model")

    @step
    def start(self):
        print("Fetching book reviews from IMDb...")
        dataset = load_dataset("imdb")  # Using IMDb as a proxy for book reviews
        self.raw_data = dataset['train'].shuffle(seed=42).select(range(2000))  # Limit for faster training
        self.test_data = dataset['test'].shuffle(seed=42).select(range(500))
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

        # Generate version ID
        self.version_id = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")

        # File names
        self.model_path = "model.joblib"
        self.vectorizer_path = "vectorizer.joblib"
        self.metrics_path = "metrics.json"
        self.confusion_path = "confusion_matrix.png"

        # Save artifacts
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)

        metrics = {
            "accuracy": accuracy_score(self.y_val, y_pred),
            "f1_score": f1_score(self.y_val, y_pred)
        }
        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f)

        plt.savefig(self.confusion_path)
        self.next(self.push)

    @step
    def push(self):
        print("Uploading model and artifacts to S3...")
        s3 = boto3.client('s3')
        version_folder = f"models/{self.version_id}"

        for fname in [self.model_path, self.vectorizer_path, self.metrics_path, self.confusion_path]:
            s3.upload_file(fname, self.s3_bucket, f"{version_folder}/{fname}")

        self.model_url = f"https://{self.s3_bucket}.s3.amazonaws.com/{version_folder}/{self.model_path}"
        self.next(self.end)

    @step
    def end(self):
        print("Flow completed!")
        print(f"Model available at: {self.model_url}")

if __name__ == '__main__':
    SentimentAnalysisFlow()
