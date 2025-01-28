import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import openai  # Import OpenAI for VetGPT integration

# Load OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai_api_key"]

# Load the tokenizer and model from Hugging Face Hub
model_name = 'havocy28/VetBERTDx'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Function to diagnose pet health using the model
def diagnose_pet_health(user_input):
    """Diagnose pet health using the Hugging Face model."""
    # Encode the text and prepare inputs for the model
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Predict and compute softmax to get probabilities
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)

    # Retrieve label mapping from model's configuration
    label_map = model.config.id2label

    # Combine labels and probabilities, and sort by probability in descending order
    diagnosis = {label_map[idx]: prob.item() for idx, prob in enumerate(probabilities[0])}
    diagnosis = {k: v for k, v in sorted(diagnosis.items(), key=lambda item: item[1], reverse=True)}
    
    return diagnosis

# Function to get GPT-4 response based on the top classifications
def get_vetGPT_response(top_label, second_label, user_input):
    """Fetch detailed advice from GPT-4 based on diagnosis."""
    prompt = (
        f"You are a veterinary assistant named VetGPT. A pet owner described their pet's symptoms as:\n\n"
        f"'{user_input}'\n\n"
        f"The AI model classified the condition as:\n"
        f"1️⃣ **{top_label}** (most likely)\n"
        f"2️⃣ **{second_label}** (second most likely)\n\n"
        f"Explain what the top condition means, possible causes, recommended next steps, and whether the pet should see a vet."
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful veterinary assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response["choices"][0]["message"]["content"]

# Streamlit app
st.title("Pet Vet Chatbot 🐾")
st.write("Welcome to the Pet Vet Chatbot! Describe your pet's symptoms, and I'll help diagnose the issue.")

# Input text box for user to describe symptoms
user_input = st.text_area("Describe your pet's symptoms (e.g., vomiting, lethargy, etc.):")

if st.button("Diagnose"):
    if user_input.strip() == "":
        st.warning("Please describe your pet's symptoms.")
    else:
        # Get diagnosis from the model
        diagnosis = diagnose_pet_health(user_input)

        # Display diagnosis results
        st.subheader("Diagnosis Results:")
        labels = list(diagnosis.keys())  # Extract labels
        if len(labels) < 2:
            st.error("Not enough classifications to proceed. Please refine the input.")
        else:
            top_label, second_label = labels[0], labels[1]  # Top two classifications

            # Display probabilities
            st.write("**Model Diagnosis Results:**")
            for label, prob in diagnosis.items():
                st.write(f"- **{label}**: {prob:.2%} probability")

            # Fetch VetGPT's response based on the top classifications
            st.subheader(f"VetGPT's Advice on **{top_label}**")
            try:
                vetGPT_response = get_vetGPT_response(top_label, second_label, user_input)
                st.write(vetGPT_response)
            except Exception as e:
                st.error(f"An error occurred while fetching GPT-4 response: {str(e)}")


