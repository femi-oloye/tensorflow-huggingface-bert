import pandas as pd
import tensorflow as tf
from datasets import Dataset
from transformers import BertTokenizerFast, TFBertForSequenceClassification, DataCollatorWithPadding
from transformers import create_optimizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import argparse
import logging

# 🪵 Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 📥 Load CSV data
def load_data(csv_path):
    logger.info(f"Loading dataset from {csv_path}")
    return pd.read_csv(csv_path)

# ✂️ Split and format data
def preprocess_data(df):
    logger.info("Splitting data into train and test sets")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
    )
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})
    return train_dataset, test_dataset

# 🧼 Tokenize text
def tokenize_data(train_dataset, test_dataset, tokenizer, max_length):
    logger.info("Tokenizing datasets")
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, max_length=max_length)

    train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=2, remove_columns=["text"])
    test_dataset = test_dataset.map(tokenize_function, batched=True, num_proc=2, remove_columns=["text"])
    return train_dataset, test_dataset

# 🔄 Prepare TensorFlow datasets
def prepare_tf_datasets(train_dataset, test_dataset, tokenizer, batch_size):
    logger.info("Preparing TensorFlow datasets")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

    tf_train = train_dataset.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["label"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    tf_test = test_dataset.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["label"],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    return tf_train, tf_test

# 🧠 Train BERT model
def build_and_train_model(tf_train, tf_test, num_train_steps, num_warmup_steps, model_name, epochs):
    logger.info("Loading and training BERT model")
    model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    optimizer, schedule = create_optimizer(
        init_lr=5e-5,
        num_warmup_steps=num_warmup_steps,
        num_train_steps=num_train_steps
    )
    model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=["accuracy"])
    model.fit(tf_train, validation_data=tf_test, epochs=epochs)
    return model

# 📊 Evaluate model
def evaluate_model(model, tf_test):
    logger.info("Evaluating model performance")
    y_true = []
    y_pred = []
    for batch in tf_test:
        inputs = {k: v for k, v in batch.items() if k != "label"}
        logits = model.predict(inputs).logits
        preds = tf.math.argmax(logits, axis=1).numpy()
        labels = batch["label"].numpy()
        y_pred.extend(preds)
        y_true.extend(labels)

    report = classification_report(y_true, y_pred, digits=4)
    print("\nClassification Report:\n", report)

# 💾 Save model
def save_model(model, tokenizer, output_dir):
    logger.info(f"Saving model and tokenizer to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# 🚀 Main pipeline
def main(args):
    df = load_data(args.csv_path)
    train_dataset, test_dataset = preprocess_data(df)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    train_dataset, test_dataset = tokenize_data(train_dataset, test_dataset, tokenizer, args.max_length)
    tf_train, tf_test = prepare_tf_datasets(train_dataset, test_dataset, tokenizer, args.batch_size)

    num_train_steps = len(tf_train) * args.epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    model = build_and_train_model(tf_train, tf_test, num_train_steps, num_warmup_steps, args.model_name, args.epochs)
    evaluate_model(model, tf_test)
    save_model(model, tokenizer, args.output_dir)
    logger.info("✅ Pipeline complete.")

# 🧾 CLI Setup
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT on sentiment data from CSV")
    parser.add_argument("--csv_path", type=str, default="sample_sentiment.csv", help="Path to CSV file")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Hugging Face model name")
    parser.add_argument("--max_length", type=int, default=128, help="Max token length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--output_dir", type=str, default="./bert_sentiment_model", help="Directory to save model")
    args = parser.parse_args()

    main(args)
