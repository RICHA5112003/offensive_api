import torch
from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, jsonify
import requests
from io import BytesIO

class IndicBERTClassifier(torch.nn.Module):
    def __init__(self, model_name="ai4bharat/indic-bert", num_classes=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        dropped = self.dropout(pooled_output)
        return self.classifier(dropped)

# Load model from Hugging Face
MODEL_URL = "https://huggingface.co/richa051122/Offensiveclassification/resolve/main/best_indicbert_model.pt"
model = IndicBERTClassifier()
model.load_state_dict(torch.load(BytesIO(requests.get(MODEL_URL).content), map_location=torch.device("cpu")))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")

app = Flask(__name__)

@app.route("/")
def home():
    return "IndicBERT Offensive Comment Classifier is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs, dim=1).item()
    label = "OFFENSIVE" if prediction == 1 else "NOT_OFFENSIVE"
    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
