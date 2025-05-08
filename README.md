# 🧠 BERT Sentiment Analysis Pipeline (TensorFlow + Hugging Face)

This project demonstrates a **modular, production-ready sentiment analysis pipeline** built with:

- 🤗 Hugging Face Transformers  
- 🧠 TensorFlow 2  
- 📊 Evaluation metrics (Precision, Recall, F1)  
- 🧪 CLI interface for custom training  
- 🪵 Logging for traceability  
- 🚀 Deployable via FastAPI or Docker

---

## 🎯 Purpose

This project was built to:

1. **Showcase real-world NLP pipeline design** for interviews or portfolio use.
2. Provide a **clean, modular codebase** for training a BERT model on CSV data.
3. Demonstrate **parallel preprocessing**, evaluation, and deployment readiness.
4. Serve as a foundation for building AI APIs or services (e.g., SaaS insights, chatbots).

---

## 🏁 Workflow Overview

CSV → Preprocessing → Tokenization (parallel) → Model Training → Evaluation → Save Model → Deploy


### 🔧 Steps

1. **Load CSV**: Read text + label data from a file.
2. **Split & Format**: Split into train/test and convert to Hugging Face Datasets.
3. **Tokenize**: Use `BertTokenizerFast` with parallelism (`num_proc=2`).
4. **Convert to TF Dataset**: Pad inputs, shuffle and batch.
5. **Train**: Fine-tune `bert-base-uncased` using `TFBertForSequenceClassification`.
6. **Evaluate**: Generate Precision, Recall, F1-score.
7. **Deploy**: Use FastAPI or Docker to expose prediction endpoint.

---

## 📁 Project Structure

📦 bert_sentiment_pipeline/
├── bert_sentiment_pipeline.py # Main modular script
├── sample_sentiment.csv # Example data file
├── app.py # FastAPI app for deployment
├── requirements.txt # All required Python packages
└── README.md # Project documentation


---

## 📊 Sample Data Format

```csv
text,label
"I love this product!",1
"This is the worst!",0

label = 1 → Positive

label = 0 → Negative
```

How to Run the Project
1. Install requirements
```
pip install -r requirements.txt
```

2. Train the model

python bert_sentiment_pipeline.py --csv_path sample_sentiment.csv --epochs 3

3. Serve predictions (via FastAPI)

uvicorn app:app --reload

Test via browser at:
http://localhost:8000/docs

📈 Evaluation Example

Classification Report:

              precision    recall  f1-score   support

     negative     0.89      0.85      0.87        20
     positive     0.88      0.90      0.89        20

    accuracy                         0.88        40

🐳 Optional: Docker Deployment

docker build -t bert-sentiment-api .
docker run -p 8000:8000 bert-sentiment-api

✨ Outcome

    🔥 Trained sentiment classifier using BERT with TensorFlow

    📦 CLI-compatible modular script

    📊 Evaluation-ready with metrics

    🚀 Ready to deploy via FastAPI or Docker

🧑‍💼 Ideal For

    AI interview preparation

    NLP learning projects

    SaaS feedback analysis

    Chatbot or frontend API integration