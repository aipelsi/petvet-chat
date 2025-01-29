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
def get_vetGPT_response(conditions, user_input):
    """Fetch a compassionate, easy-to-understand veterinary response from GPT-4 based on potential conditions."""
    
    # Extract the top two likely conditions
    top_conditions = ", ".join([condition[0] for condition in conditions])

    prompt = (
        f"You are a compassionate veterinary assistant named VetGPT. A pet owner described their pet's symptoms as:\n\n"
        f"'{user_input}'\n\n"
        f"Based on the AI model's assessment, the symptoms could indicate conditions such as **{top_conditions}**.\n\n"
        f"Please provide a **thoughtful, easy-to-understand** response explaining what this might mean, possible causes, and simple care steps "
        f"the owner can take at home. Be **empathetic** and **clear**, but remind them that this is **not a medical diagnosis** and they "
        f"should consult a veterinarian for proper care."
    )

    response = client.chat.completions.create(  # Updated for OpenAI v1.0+
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a friendly and empathetic veterinary assistant, offering triage-based advice."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Streamlit app
st.title("Pet Vet Chatbot 🐾")
st.write("Welcome to the Pet Vet Chatbot! Describe your pet's symptoms, and I'll provide basic guidance and triage advice.")

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
            # Fetch VetGPT's response based on the possible conditions
            st.subheader(f"VetGPT's Guidance for Your Pet")
            try:
                vetGPT_response = get_vetGPT_response(diagnosis, user_input)
                st.write(vetGPT_response)
                st.markdown("⚠️ **Disclaimer:** This chatbot does **not** provide medical advice. It is only for **triage purposes** and general guidance. If your pet is unwell, please consult a **licensed veterinarian immediately**.")
            except Exception as e:
                st.error(f"An error occurred while fetching GPT-4 response: {str(e)}")


