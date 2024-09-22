import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained(r'path/to/model')
tokenizer = BertTokenizer.from_pretrained(r'path/to/model')

# Load your test dataset
##If cloning then place replace the path of datasets
test_data = pd.read_csv('test.csv')

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class  # Return 0 or 1

# Predict sentiments for the test dataset
test_data['predicted_sentiment'] = test_data['combined'].apply(predict_sentiment)

# Evaluate model performance
true_labels = test_data['sentiment']  # Assuming sentiment is in 0 and 1
predicted_labels = test_data['predicted_sentiment']

# Compute evaluation metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

# Display evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
