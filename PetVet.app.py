import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import openai  # Import OpenAI for VetGPT integration

# Fetch OpenAI API key from Streamlit Secrets
if "openai_api_key" not in st.secrets:
    st.error("❌ OpenAI API key is missing! Please add it to Streamlit Secrets.")
else:
    client = openai.OpenAI(api_key=st.secrets["openai_api_key"])  # Use new client-based approach

# Load the tokenizer and model from Hugging Face Hub
model_name = 'havocy28/VetBERTDx'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Function to diagnose pet health using the model
def diagnose_pet_health(user_input):
    """Diagnose pet health using the Hugging Face model and return the top two classifications, excluding 'repeat'."""
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)
    label_map = model.config.id2label
    diagnosis = {label_map[idx]: prob.item() for idx, prob in enumerate(probabilities[0])}

    # Remove 'repeat' from the diagnosis list
    filtered_diagnosis = [(label, prob) for label, prob in sorted(diagnosis.items(), key=lambda item: item[1], reverse=True) if label.lower() != "repeat"]

    return filtered_diagnosis[:2]  # Return only the top two classifications (excluding 'repeat')

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

    response = client.chat.completions.create(  # Updated for OpenAI v1.0+
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful veterinary assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Streamlit app
st.title("Pet Vet Chatbot 🐾")
st.write("Welcome to the Pet Vet Chatbot! Describe your pet's symptoms, and I'll provide veterinary advice.")

# Input text box for user to describe symptoms
user_input = st.text_area("Describe your pet's symptoms (e.g., vomiting, lethargy, etc.):")

if st.button("Get Advice"):
    if user_input.strip() == "":
        st.warning("Please describe your pet's symptoms.")
    elif "openai_api_key" not in st.secrets:
        st.error("OpenAI API key is missing! Add it to Streamlit Secrets.")
    else:
        # Get the top two diagnosis results from the model
        diagnosis = diagnose_pet_health(user_input)
        
        if len(diagnosis) < 2:
            st.error("Not enough valid classifications to proceed. Please refine the input.")
        else:
            # Extract top two classifications
            (top_label, _), (second_label, _) = diagnosis
            
            # Fetch VetGPT's response based on the top classifications
            st.subheader(f"VetGPT's Advice on Your Pet's Condition")
            try:
                vetGPT_response = get_vetGPT_response(top_label, second_label, user_input)
                st.write(vetGPT_response)
            except Exception as e:
                st.error(f"An error occurred while fetching GPT-4 response: {str(e)}")

