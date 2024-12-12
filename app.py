import warnings
warnings.filterwarnings('ignore')

import json
import os
import numpy as np
import pandas as pd
import random
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from flask import Flask, request, jsonify, render_template
import torch

# Load the data
def load_json_file(filename):
    with open(filename, encoding='utf-8') as f:
        return json.load(f)

json_file = "C:/Users/Azui-Cz1/Desktop/T/dataset_chatbot.json"  # Update with your actual path
intents = load_json_file(json_file)

# Validate JSON structure
if not all('tag' in intent and 'patterns' in intent for intent in intents['intents']):
    raise ValueError("Invalid JSON structure")

# Extract Info from the JSON data file and Store it in a dataframe
df = pd.DataFrame([(pattern.lower(), intent['tag']) for intent in intents['intents'] for pattern in intent['patterns']], 
                  columns=['Pattern', 'Tag'])

# Prepare labels
labels = df['Tag'].unique().tolist()
label2id = {label: id for id, label in enumerate(labels)}
id2label = {v: k for k, v in label2id.items()}
df['labels'] = df['Tag'].map(label2id)

# Split the data into train and test
X = list(df['Pattern'])
y = list(df['labels'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Load BERT Pretrained model and Tokenizer
model_name = "indolem/indobert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# Transform the data into numerical format
train_encoding = tokenizer(X_train, truncation=True, padding=True, max_length=512)
test_encoding = tokenizer(X_test, truncation=True, padding=True, max_length=512)

# Build Data Loader
class DataLoader(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataloader = DataLoader(train_encoding, y_train)
test_dataloader = DataLoader(test_encoding, y_test)

# Define Evaluation Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {'Accuracy': acc, 'F1': f1, 'Precision': precision, 'Recall': recall}

# Define Training Arguments
training_args = TrainingArguments(
    output_dir='./output',
    learning_rate=5e-5,
    do_train=True,
    do_eval=True,
    fp16=True,
    num_train_epochs=1,  # Increased epochs for better training
    per_device_train_batch_size=32,  # Adjusted batch size
    per_device_eval_batch_size=16,
    logging_strategy='steps',
    logging_steps=50,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataloader,
    eval_dataset=test_dataloader,
    compute_metrics=compute_metrics,
)

print("Training the model...")
trainer.train()

# Ensure model path exists
model_path = "chatbot"
os.makedirs(model_path, exist_ok=True)

# Save model and tokenizer using multiple methods for reliability
try:
    # Method 1: Save with save_pretrained
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Method 2: Save state dict
    torch.save(model.state_dict(), os.path.join(model_path, 'model_state.pth'))
    
    print(f"Model saved successfully in {model_path}")
except Exception as e:
    print(f"Error saving model: {e}")

# Load the model
print("Loading model...")
try:
    # Try loading with save_pretrained method
    model = BertForSequenceClassification.from_pretrained(
        model_path, 
        id2label=id2label, 
        label2id=label2id
    )
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    # Create pipeline
    chatbot = pipeline("text-classification", model=model, tokenizer=tokenizer)
except Exception as e:
    print(f"Error loading model: {e}")

# Flask Application for Chatbot
app = Flask(__name__)

# Route Halaman Utama
@app.route("/")
def index():
    return render_template("index.html")

# Fungsi utama untuk mendapatkan respons dari chatbot
def get_chatbot_response(user_input, chatbot, intents, id2label):
    try:
        prediction = chatbot(user_input)[0]
        label = prediction['label']  # Formatnya seperti "LABEL_0"
        tag = id2label.get(int(label.split('_')[-1]), None)  # Ambil tag berdasarkan ID label

        if tag:
            for intent in intents['intents']:
                if intent['tag'] == tag:
                    return random.choice(intent['responses']) if intent['responses'] else "I'm not sure how to respond to that."
        return "I'm not sure how to respond to that."
    except Exception as e:
        print(f"Error in get_chatbot_response: {e}")
        return "Sorry, I'm experiencing some difficulties."

# Flask route
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    response = get_chatbot_response(user_input, chatbot, intents, id2label)
    return jsonify({"response": response})

# Mode interaktif CLI
def interactive_chat(chatbot, intents, id2label):
    print("Chatbot: Hi! I am your virtual assistant. Feel free to ask, and I'll do my best to provide you with answers.")
    print("Type 'quit' to exit the chat\n\n")

    user_input = input(":User  ").strip()
    while user_input != 'quit':
        if not user_input:
            print("Chatbot: Please enter a valid query.")
        else:
            response = get_chatbot_response(user_input, chatbot, intents, id2label)
            print(f"Chatbot: {response}\n\n")
        user_input = input(":User  ").strip()

# Pilih mode: Flask API atau CLI
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        interactive_chat(chatbot, intents, id2label)
    else:
        app.run(debug=True)