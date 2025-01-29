import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import openai  # Import OpenAI for VetGPT integration

# Fetch OpenAI API key from Streamlit Secrets
if "openai_api_key" not in st.secrets:
    st.error("❌ OpenAI API key is missing! Please add it to Streamlit Secrets.")
else:
    openai.api_key = st.secrets["openai_api_key"]

# Load the tokenizer and model from Hugging Face Hub
model_name = 'havocy28/VetBERTDx'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Function to diagnose pet health using the model
def diagnose_pet_health(user_input):
    """Diagnose pet health using the Hugging Face model."""
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)
    label_map = model.config.id2label
    diagnosis = {label_map[idx]: prob.item() for idx, prob in enumerate(probabilities[0])}
    return {k: v for k, v in sorted(diagnosis.items(), key=lambda item: item[1], reverse=True)}

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
    elif "openai_api_key" not in st.secrets:
        st.error("OpenAI API key is missing! Add it to Streamlit Secrets.")
    else:
        diagnosis = diagnose_pet_health(user_input)
        st.subheader("Diagnosis Results:")
        labels = list(diagnosis.keys())
        if len(labels) < 2:
            st.error("Not enough classifications to proceed. Please refine the input.")
        else:
            top_label, second_label = labels[0], labels[1]
            for label, prob in diagnosis.items():
                st.write(f"- **{label}**: {prob:.2%} probability")
            st.subheader(f"VetGPT's Advice on **{top_label}**")
            try:
                vetGPT_response = get_vetGPT_response(top_label, second_label, user_input)
                st.write(vetGPT_response)
            except Exception as e:
                st.error(f"An error occurred while fetching GPT-4 response: {str(e)}")



