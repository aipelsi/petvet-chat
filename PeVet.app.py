import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the tokenizer and model from Hugging Face Hub
model_name = 'havocy28/VetBERTDx'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Streamlit user input
st.title("PetVet Chatbot")
st.write("This is a petvet chatbot that classifies veterinary diagnostic text.")

text_input = st.text_area("Enter pet details here", "Hx: 7 yo canine with history of vomiting intermittently since yesterday. No other concerns. Still eating and drinking normally. cPL negative.")

if st.button('Classify'):
    # Encode the text and prepare inputs for the model
    inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Predict and compute softmax to get probabilities
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)

    # Retrieve label mapping from model's configuration
    label_map = model.config.id2label

    # Combine labels and probabilities, and sort by probability in descending order
    sorted_probs = sorted(((prob.item(), label_map[idx]) for idx, prob in enumerate(probabilities[0])), reverse=True, key=lambda x: x[0])

    # Display sorted probabilities and labels
    st.write("Classification Results:")
    for prob, label in sorted_probs:
        st.write(f"{label}: {prob:.4f}")
