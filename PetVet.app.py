import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import openai  

# Fetch OpenAI API key from Streamlit Secrets
if "openai_api_key" not in st.secrets:
    st.error("❌ OpenAI API key is missing! Please add it to Streamlit Secrets.")
else:
    client = openai.OpenAI(api_key=st.secrets["openai_api_key"])

# Load the tokenizer and model from Hugging Face Hub
model_name = 'havocy28/VetBERTDx'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# ✅ Apply Black Background with White Text and Proper Button Styling
st.markdown("""
    <style>
    .stApp {
        background-color: white!important; /* Black background */
        color: black !important;
    }
    .main-container {
        background: white;
        padding: 40px;
        border-radius: 12px;
        box-shadow: 0px 0px 20px rgba(255, 255, 255, 0.3);
        max-width: 800px;
        margin: auto;
        color: white !important;
    }
    h1, h2, h3, p, .stMarkdown {
        color: white !important;
        text-align: center;
    }
    .stTextArea, .stTextInput {
        color: black !important;
        background: white !important;
    }
    .stButton>button {
        color: black !important;
        background: white !important;
        border: 2px solid white !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        transition: 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background: lightgray !important;
        color: black !important;
        border: 2px solid lightgray !important;
    }
    .stRadio {
        text-align: center;
    }
    </style>
    <div class="main-container">
    """, unsafe_allow_html=True)

# ✅ Centered Title & Description
st.markdown("""
    <h1>🐾 Pet Vet Chatbot</h1>
    <p style="font-style: italic;">Your AI-powered veterinary assistant</p>
    """, unsafe_allow_html=True)

# Function to diagnose pet health using the model
def diagnose_pet_health(user_input):
    """Diagnose pet health using the Hugging Face model and return the top two classifications, excluding 'repeat'."""
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)
    label_map = model.config.id2label
    diagnosis = {label_map[idx]: prob.item() for idx, prob in enumerate(probabilities[0])}
    filtered_diagnosis = [(label, prob) for label, prob in sorted(diagnosis.items(), key=lambda item: item[1], reverse=True) if label.lower() != "repeat"]
    return filtered_diagnosis[:2]  # Return only the top two classifications

# Function to get GPT-4 response based on the top classifications
def get_vetGPT_response(conditions, user_input):
    """Fetch a compassionate, easy-to-understand veterinary response from GPT-4 based on potential conditions."""
    top_conditions = ", ".join([condition[0] for condition in conditions])
    prompt = (
        f"You are a veterinary assistant named VetGPT. A pet owner described their pet's symptoms as:\n\n"
        f"'{user_input}'\n\n"
        f"The AI model suggests it could be **{top_conditions}**.\n\n"
        f"Explain what this might mean, possible causes, simple care steps, and whether the pet should see a vet."
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a friendly and empathetic veterinary assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Initialize Streamlit App
st.write("Describe your pet's symptoms, and I'll provide initial guidance.")

# Select Mode: Diagnosis or Chatbot
mode = st.radio("Choose a mode:", ["🐕 AI Triage Line & Advice", "💬 Chat with VetGPT"])

if mode == "🐕 AI Triage Line & Advice":
    user_input = st.text_area("Describe your pet's symptoms (e.g., vomiting, lethargy, etc.):")
    if st.button("Get Advice"):
        if user_input.strip() == "":
            st.warning("Please describe your pet's symptoms.")
        else:
            diagnosis = diagnose_pet_health(user_input)
            if len(diagnosis) < 2:
                st.error("Not enough classifications. Please refine the input.")
            else:
                st.subheader(f"VetGPT's Guidance for Your Pet")
                try:
                    vetGPT_response = get_vetGPT_response(diagnosis, user_input)
                    st.write(vetGPT_response)
                    st.markdown("⚠️ **This is for triage only. Always consult a licensed veterinarian.**")
                except Exception as e:
                    st.error(f"Error fetching GPT-4 response: {str(e)}")

elif mode == "💬 Chat with VetGPT":
    st.subheader("💬 Chat with VetGPT")
    st.write("Ask anything about pet care and symptoms!")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_chat_input = st.chat_input("Type your message here...")
    if user_chat_input:
        st.session_state["messages"].append({"role": "user", "content": user_chat_input})
        response = client.chat.completions.create(
            model="gpt-4",
            messages=st.session_state["messages"]
        )
        bot_reply = response.choices[0].message.content
        st.session_state["messages"].append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.write(bot_reply)

# ✅ Close the Padded Content Box
st.markdown("</div>", unsafe_allow_html=True)







