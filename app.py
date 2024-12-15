import warnings
warnings.filterwarnings('ignore')

import os
import json
import random
import re
import pandas as pd  # Add this import for pandas
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Flask App Initialization
app = Flask(__name__)

# Path to the dataset
json_file = os.path.join(os.path.dirname(__file__), "dataset_chatbot.json")

# Load JSON data
def load_json_file(filename):
    with open(filename) as f:
        return json.load(f)

intents = load_json_file(json_file)

# Extract info and preprocess data
def extract_json_info(json_file):
    patterns, tags = [], []
    for intent in json_file['intents']:
        if 'tag' in intent:
            patterns.extend(intent['patterns'])
            tags.extend([intent['tag']] * len(intent['patterns']))
    return pd.DataFrame({'Pattern': patterns, 'Tag': tags})

df = extract_json_info(intents)

# Preprocess pattern text
def preprocess_pattern(pattern):
    pattern = pattern.lower()
    pattern = re.sub(r'[^\w\s]', '', pattern)  # Remove punctuation
    pattern = re.sub(r'\s+', ' ', pattern).strip()  # Remove extra spaces
    return pattern

df['Pattern'] = df['Pattern'].apply(preprocess_pattern)

# Prepare labels
labels = df['Tag'].unique().tolist()
label2id = {label: id for id, label in enumerate(labels)}
id2label = {id: label for label, id in label2id.items()}
df['labels'] = df['Tag'].map(lambda x: label2id[x.strip()])

# Split data
X = list(df['Pattern'])
y = list(df['labels'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Load BERT model and tokenizer
model_name = "indolem/indobert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label2id))

# Transform the data
train_encoding = tokenizer(X_train, truncation=True, padding=True, return_tensors="pt")
test_encoding = tokenizer(X_test, truncation=True, padding=True, return_tensors="pt")

# Define Dataset class
class ChatbotDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ChatbotDataset(train_encoding, y_train)
test_dataset = ChatbotDataset(test_encoding, y_test)

# Define evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {'Accuracy': acc, 'F1': f1, 'Precision': precision, 'Recall': recall}

# Training arguments
training_args = TrainingArguments(
    output_dir='./output',
    do_train=True,
    do_eval=True,
    num_train_epochs=70,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,  # Gradient clipping
    logging_strategy='steps',
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    load_best_model_at_end=True,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
model_path = "chatbot"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

# Load trained model
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Predict intent
def predict_intent(text):
    tokens = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    outputs = model(**tokens)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions.item()

# Flask endpoints
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"response": "I didn't receive any message. Please try again."})

    # Predict intent using the BERT model
    intent_id = predict_intent(user_message)
    intent_label = id2label[intent_id]

    # Fetch the response from the dataset based on predicted intent
    response_list = []
    for intent in intents['intents']:
        if intent['tag'] == intent_label:
            response_list = intent['responses']
            break
    
    if response_list:
        # Return a random response from the matched intent
        bot_response = random.choice(response_list)
    else:
        bot_response = "I'm not sure how to respond to that."

    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
