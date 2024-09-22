

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
import streamlit as st

# Load your historical dataset for training
historical_df = pd.read_csv('training.csv')

# Load your scraped dataset for testing
scraped_df = pd.read_csv('cleaned_file.csv')
historical_df['Text'] = historical_df['Sentiment'].str.lower()  # <-- Preprocess historical dataset
scraped_df['combined'] = scraped_df['combined'].str.lower()


# Map sentiments for historical data (e.g., 1 for increase, 0 for decrease)
historical_df['Sentiment'] = historical_df['Sentiment'].map({'increase': 1, 'decrease': 0})



# Split the historical data into training and test sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    historical_df['Text'].tolist(), historical_df['Sentiment'].tolist(), test_size=0.2
)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the historical data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# Create a custom dataset class
class StockDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create the datasets
train_dataset = StockDataset(train_encodings, train_labels)
val_dataset = StockDataset(val_encodings, val_labels)

# Load BERT model for classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Set up DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Function to train the model
def train(model, train_loader, optimizer, device):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        inputs = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Train for several epochs
epochs = 3
for epoch in range(epochs):
    train(model, train_loader, optimizer, device)
    print(f'Epoch {epoch + 1} complete')

# Save the trained model
model.save_pretrained(r'C:\Users\Asus\Desktop\Capx\Model_And_Tokenizer')
tokenizer.save_pretrained(r'C:\Users\Asus\Desktop\Capx\Model_And_Tokenizer')

# Streamlit interface
st.title("Stock Sentiment Prediction")

# Text input for user to enter a company statement
user_input = st.text_input("Enter a statement about a company's stock:")

if st.button('Predict'):
    if user_input:
        # Preprocess and tokenize the user's input
        input_encodings = tokenizer(user_input.lower(), truncation=True, padding=True, max_length=512, return_tensors="pt")
        
        # Move input to device
        input_encodings = {key: val.to(device) for key, val in input_encodings.items()}
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            outputs = model(**input_encodings)
        
        # Get the predicted sentiment (increase or decrease)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
        sentiment = "Increase" if predicted_label == 1 else "Decrease"
        
        # Display the prediction
        st.write(f"Predicted sentiment: {sentiment}")

# Button to predict sentiment for the entire scraped dataset
if st.button('Predict on Scraped Data'):
    # Preprocess and tokenize the scraped data
    scraped_encodings = tokenizer(scraped_df['combined'].tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt")

    # Move to device
    scraped_encodings = {key: val.to(device) for key, val in scraped_encodings.items()}

    # Make predictions
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(scraped_encodings['input_ids'].shape[0]):
            input_data = {key: val[i].unsqueeze(0) for key, val in scraped_encodings.items()}
            outputs = model(**input_data)
            predicted_label = torch.argmax(outputs.logits, dim=1).item()
            predictions.append("Increase" if predicted_label == 1 else "Decrease")
    
    # Save the predictions back to the scraped DataFrame
    scraped_df['Prediction'] = predictions
    scraped_df.to_csv('scraped_data_with_predictions.csv', index=False)
    st.write("Predictions saved to 'scraped_data_with_predictions.csv'.")

