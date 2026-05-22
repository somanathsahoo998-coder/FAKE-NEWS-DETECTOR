# FAKE-NEWS-DETECTOR
An AI/ML based web application to detect fake news using Natural Language Processing and Machine Learning.
⚙️ How It Works

STEP-1 — Paste Content
Users paste any news article, headline, or social media content into the text box.

STEP-2 — AI Analysis
The system analyzes the content using machine learning and NLP techniques to detect:
Sensational words
Reliability patterns
Logical inconsistencies
Fake news indicators

 STEP 3 — Get Verdict
The application provides:
Real or Fake prediction
Credibility analysis
Red flag detection

-------------------------------------CODE: ----------------------------------------------------
 !pip install flask flask-cors pyngrok scikit-learn pandas joblib
!pip install datasets

Requirement already satisfied: flask in /usr/local/lib/python3.12/dist-packages (3.1.3)
Collecting flask-cors
  Downloading flask_cors-6.0.2-py3-none-any.whl.metadata (5.3 kB)
Collecting pyngrok
  Downloading pyngrok-8.1.2-py3-none-any.whl.metadata (8.6 kB)
Requirement already satisfied: scikit-learn in /usr/local/lib/python3.12/dist-packages (1.6.1)
Requirement already satisfied: pandas in /usr/local/lib/python3.12/dist-packages (2.2.2)
Requirement already satisfied: joblib in /usr/local/lib/python3.12/dist-packages (1.5.3)
Requirement already satisfied: blinker>=1.9.0 in /usr/local/lib/python3.12/dist-packages (from flask) (1.9.0)
Requirement already satisfied: click>=8.1.3 in /usr/local/lib/python3.12/dist-packages (from flask) (8.3.3)
Requirement already satisfied: itsdangerous>=2.2.0 in /usr/local/lib/python3.12/dist-packages (from flask) (2.2.0)
Requirement already satisfied: jinja2>=3.1.2 in /usr/local/lib/python3.12/dist-packages (from flask) (3.1.6)
Requirement already satisfied: markupsafe>=2.1.1 in /usr/local/lib/python3.12/dist-packages (from flask) (3.0.3)
Requirement already satisfied: werkzeug>=3.1.0 in /usr/local/lib/python3.12/dist-packages (from flask) (3.1.8)
Requirement already satisfied: PyYAML>=5.1 in /usr/local/lib/python3.12/dist-packages (from pyngrok) (6.0.3)
Requirement already satisfied: numpy>=1.19.5 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (2.0.2)
Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (1.16.3)
Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (3.6.0)
Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.12/dist-packages (from pandas) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.12/dist-packages (from pandas) (2025.2)
Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/dist-packages (from pandas) (2026.1)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)
Downloading flask_cors-6.0.2-py3-none-any.whl (13 kB)
Downloading pyngrok-8.1.2-py3-none-any.whl (25 kB)
Installing collected packages: pyngrok, flask-cors
...
Requirement already satisfied: annotated-doc>=0.0.2 in /usr/local/lib/python3.12/dist-packages (from typer->huggingface-hub>=0.24.0->datasets) (0.0.4)
Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.12/dist-packages (from rich>=12.3.0->typer->huggingface-hub>=0.24.0->datasets) (4.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.12/dist-packages (from rich>=12.3.0->typer->huggingface-hub>=0.24.0->datasets) (2.20.0)
Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.12/dist-packages (from markdown-it-py>=2.2.0->rich>=12.3.0->typer->huggingface-hub>=0.24.0->datasets) (0.1.2)
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...

from google.colab import drive
drive.mount('/content/drive')

import shutil
shutil.copy('/content/drive/MyDrive/Fake.csv', '/content/Fake.csv')
shutil.copy('/content/drive/MyDrive/True.csv', '/content/True.csv')
shutil.copy('/content/drive/MyDrive/True1.csv', '/content/True1.csv')
shutil.copy('/content/drive/MyDrive/Fake1.csv', '/content/Fake1.csv')
shutil.copy('/content/drive/MyDrive/fake_news_dataset.csv', '/content/fake_news_dataset.csv')
print("Files copied!")

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Dataset 1: News CSVs
fake = pd.read_csv('Fake.csv')
real = pd.read_csv('True.csv')
fake['label'] = 1
real['label'] = 0
news_df = pd.concat([fake, real]).sample(frac=1).reset_index(drop=True)
news_df['text'] = news_df['title'] + ' ' + news_df['text']
news_df = news_df[['text', 'label']]
print(f"News dataset: {len(news_df)} rows")

# Dataset 1b: fake_news_dataset.csv
fake_news = pd.read_csv('fake_news_dataset.csv')
fake_news = fake_news[['text', 'label']].dropna()
# label is already 0/1
print(f"fake_news_dataset: {len(fake_news)} rows")

# Dataset 1c: Fake1.csv and True1.csv
fake1 = pd.read_csv('Fake1.csv')
true1 = pd.read_csv('True1.csv')
fake1['label'] = 1
true1['label'] = 0
extra_df = pd.concat([fake1, true1]).reset_index(drop=True)
extra_df['text'] = extra_df['title'] + ' ' + extra_df['text']
extra_df = extra_df[['text', 'label']]
print(f"Fake1 + True1 dataset: {len(extra_df)} rows")

# Dataset 2: FEVER
fever_data = load_dataset("copenlu/fever_gold_evidence")
fever_df = pd.DataFrame({
    'text':  fever_data['train']['claim'],
    'label': fever_data['train']['label']
})
fever_df = fever_df[fever_df['label'].isin([0, 1])].reset_index(drop=True)
# 0 = SUPPORTS (Real), 1 = REFUTES (Fake) — already numeric
print(f"FEVER dataset: {len(fever_df)} rows")
fever_df = pd.DataFrame({
    'text':  fever_data['train']['claim'],
    'label': fever_data['train']['label']
})
fever_df = fever_df[fever_df['label'].isin(['SUPPORTS', 'REFUTES'])].reset_index(drop=True)
fever_df['label'] = fever_df['label'].map({'SUPPORTS': 0, 'REFUTES': 1})
print(f"FEVER dataset: {len(fever_df)} rows")


# Dataset 3: Erfan English
erfan_data = load_dataset("ErfanMoosaviMonazzah/fake-news-detection-dataset-English")
erfan_df = pd.DataFrame({
    'text':  erfan_data['train']['text'],
    'label': erfan_data['train']['label']
})
print(f"Erfan dataset: {len(erfan_df)} rows")

# Combine ALL datasets
combined_df = pd.concat([news_df, fever_df, erfan_df, fake_news, extra_df]).sample(frac=1).reset_index(drop=True)
print(f"Final combined total: {len(combined_df)} rows")

# Combine both
combined_df = pd.concat([news_df, fever_df, erfan_df]).sample(frac=1).reset_index(drop=True)
print(f"Combined total: {len(combined_df)} rows")

# Train
X_train, X_test, y_train, y_test = train_test_split(
    combined_df['text'], combined_df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=15000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)
print(f"\nAccuracy: {accuracy_score(y_test, preds):.2%}")
print(classification_report(y_test, preds, target_names=['Real', 'Fake']))

joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("\nModel saved!")

News dataset: 44898 rows
fake_news_dataset: 4000 rows
Fake1 + True1 dataset: 44898 rows
/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:93: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

README.md: 0.00B [00:00, ?B/s]train.jsonl:   0%|          | 0.00/96.9M [00:00<?, ?B/s]valid.jsonl: 0.00B [00:00, ?B/s]test.jsonl: 0.00B [00:00, ?B/s]Generating train split:   0%|          | 0/228277 [00:00<?, ? examples/s]Generating validation split:   0%|          | 0/15935 [00:00<?, ? examples/s]Generating test split:   0%|          | 0/16039 [00:00<?, ? examples/s]

FEVER dataset: 0 rows
FEVER dataset: 161897 rows
README.md:   0%|          | 0.00/487 [00:00<?, ?B/s]train.tsv:   0%|          | 0.00/78.4M [00:00<?, ?B/s]validation.tsv:   0%|          | 0.00/15.5M [00:00<?, ?B/s]test.tsv:   0%|          | 0.00/22.0M [00:00<?, ?B/s]Generating train split:   0%|          | 0/30000 [00:00<?, ? examples/s]Generating validation split:   0%|          | 0/6000 [00:00<?, ? examples/s]Generating test split:   0%|          | 0/8267 [00:00<?, ? examples/s]
Erfan dataset: 30000 rows
Final combined total: 285693 rows
Combined total: 236795 rows

Accuracy: 75.20%
              precision    recall  f1-score   support

        Real       0.77      0.88      0.82     30258
        Fake       0.71      0.52      0.60     17101

    accuracy                           0.75     47359
   macro avg       0.74      0.70      0.71     47359
weighted avg       0.75      0.75      0.74     47359


Model saved!

from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
import joblib, threading

app = Flask(__name__)
CORS(app)

model      = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data       = request.get_json()
    text       = data.get('text', '')
    vec        = vectorizer.transform([text])
    pred       = model.predict(vec)[0]
    proba      = model.predict_proba(vec)[0]
    label      = 'Fake' if pred == 1 else 'Real'
    confidence = round(float(max(proba)) * 100, 1)
    return jsonify({'label': label, 'confidence': confidence, 'is_fake': bool(pred == 1)})

from pyngrok import ngrok
ngrok.set_auth_token("3E4zhq1U793RIUpiFFTSUJ9kliy_3YexrMsA2mC5jstW7savs")
threading.Thread(target=lambda: app.run(port=5000)).start()
public_url = ngrok.connect(5000)
print("Your API URL:", public_url)

from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
import joblib, threading

app = Flask(__name__)
CORS(app)

model      = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data       = request.get_json()
    text       = data.get('text', '')
    vec        = vectorizer.transform([text])
    pred       = model.predict(vec)[0]
    proba      = model.predict_proba(vec)[0]
    label      = 'Fake' if pred == 1 else 'Real'
    confidence = round(float(max(proba)) * 100, 1)
    return jsonify({'label': label, 'confidence': confidence, 'is_fake': bool(pred == 1)})

from pyngrok import ngrok
ngrok.set_auth_token("3E4zhq1U793RIUpiFFTSUJ9kliy_3YexrMsA2mC5jstW7savs")
threading.Thread(target=lambda: app.run(port=5000)).start()
public_url = ngrok.connect(5000)
print("Your API URL:", public_url)

 * Serving Flask app '__main__'
 * Debug mode: off
Address already in use
Port 5000 is in use by another program. Either identify and stop that program, or start the server with a different port.
Your API URL: NgrokTunnel: "https://dollhouse-frisbee-parched.ngrok-free.dev" -> "http://localhost:5000"

from pyngrok import ngrok
ngrok.kill()
print("Server stopped!")

Server stopped!

