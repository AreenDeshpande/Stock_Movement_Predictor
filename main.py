import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained(r'C:\Users\Asus\Desktop\Capx\Model_And_Tokenizer')
tokenizer = BertTokenizer.from_pretrained(r'C:\Users\Asus\Desktop\Capx\Model_And_Tokenizer')

# Load your scraped dataset (make sure to provide the correct path)
scraped_data = pd.read_csv(r'C:\Users\Asus\Desktop\Capx\datasets\scraped_data.csv') 

# Function to predict sentiment
def predict_outcome(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return 'Rising' if predicted_class == 1 else 'Falling'

# Streamlit app layout
st.title('Stock Sentiment Predictor')

# Display the scraped data
if not scraped_data.empty:
    st.write("Scraped Data:")
    st.dataframe(scraped_data[['Title', 'Text', 'Upvotes', 'Comments']])  

# Input field for company name
company_name = st.text_input("Enter the company name to predict stock sentiment:")

# Button to predict outcome for the specific company
if st.button("Predict Outcome"):
    if 'combined' in scraped_data.columns:
        # Filter the data by the company name entered by the user
        filtered_data = scraped_data[scraped_data['combined'].str.contains(company_name, case=False, na=False)]

        if not filtered_data.empty:
            # Predict sentiment only for rows related to the entered company
            filtered_data['predicted_outcome'] = filtered_data['combined'].apply(predict_outcome)
            st.write(filtered_data[['Title', 'Text', 'Upvotes', 'Comments', 'predicted_outcome']])
            
            # Calculate the final summary
            rising_count = (filtered_data['predicted_outcome'] == 'Rising').sum()
            falling_count = (filtered_data['predicted_outcome'] == 'Falling').sum()
            
            # Display the summary
            st.write(f"Summary for {company_name}:")
            st.write(f"Total 'Rising' predictions: {rising_count}")
            st.write(f"Total 'Falling' predictions: {falling_count}")
            
            # Final stock prediction based on majority of positive or negative outcomes
            if rising_count > falling_count:
                st.success(f"Overall Prediction: The stock for {company_name} is predicted to **increase** based on the sentiment analysis.")
            elif falling_count > rising_count:
                st.error(f"Overall Prediction: The stock for {company_name} is predicted to **decrease** based on the sentiment analysis.")
            else:
                st.info(f"Overall Prediction: The stock for {company_name} shows a **neutral** trend based on equal positive and negative outcomes.")
        else:
            st.write(f"No data found for the company: {company_name}")
    else:
        st.write("The scraped data does not contain a 'combined' column.")
