import os
import pandas as pd
import gradio as gr
import torch
from transformers import BertForSequenceClassification, BertTokenizer

# === STEP 1: Load datasets ===
fake_df = pd.read_csv(r"D:\Fake_news_detection_BERT\archive (2)\Fake.csv")
real_df = pd.read_csv(r"D:\Fake_news_detection_BERT\archive (2)\True.csv")

fake_df["label"] = 0
real_df["label"] = 1

df = pd.concat([fake_df, real_df], ignore_index=True)

# === STEP 2: Check column names ===
print("Columns:", df.columns)
text_column = "text"  # Change if your CSV uses a different column name

# === STEP 3: Load trained model if exists, else fallback ===
model_path = r"D:\Fake_news_detection_BERT\results"
if os.path.exists(model_path) and os.path.isdir(model_path):
    print(f"Loading trained model from {model_path}")
    model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
else:
    print("Trained model not found, loading fallback model.")
    model_name = "distilbert-base-uncased"  # smaller and faster
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = BertTokenizer.from_pretrained(model_name)

model.eval()

# === STEP 4: Prediction function ===
label_map = {1: "Real", 0: "Fake"}

def predict_news(text):
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**encoding)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    return label_map.get(predicted_class_id, "Unknown")

# === STEP 5: Gradio Interface ===
interface = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(lines=4, placeholder="Enter news text here..."),
    outputs="text",
    title="Fake News Detector",
    description="Enter news text to check if it is Fake or Real using a BERT-based model."
)

if __name__ == "__main__":
    interface.launch()
