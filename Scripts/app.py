from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import TFBertForSequenceClassification, BertTokenizerFast
import tensorflow as tf

app = FastAPI()

# Load model and tokenizer
model = TFBertForSequenceClassification.from_pretrained("./bert_sentiment_model")
tokenizer = BertTokenizerFast.from_pretrained("./bert_sentiment_model")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(input: TextInput):
    tokens = tokenizer(input.text, return_tensors="tf", truncation=True, padding=True)
    logits = model(tokens).logits
    pred = tf.argmax(logits, axis=1).numpy()[0]
    label = "positive" if pred == 1 else "negative"
    return {"label": label, "score": float(tf.nn.softmax(logits)[0][pred])}
