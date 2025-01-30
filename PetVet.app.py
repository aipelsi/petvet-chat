import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import openai  
import base64  # For encoding the image

# Fetch OpenAI API key from Streamlit Secrets
if "openai_api_key" not in st.secrets:
    st.error("❌ OpenAI API key is missing! Please add it to Streamlit Secrets.")
else:
    client = openai.OpenAI(api_key=st.secrets["openai_api_key"])

# Load the tokenizer and model from Hugging Face Hub
model_name = 'havocy28/VetBERTDx'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

#  Convert Local Image ("petvet.jpg") to Base64 for Streamlit Background
def get_base64_image(image_path):
    """Convert a local image file to a base64 string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

#  Use "petvet.jpg" as Background Image
image_filename = "petvet.jpg" 
background_base64 = get_base64_image(image_filename)

# Apply Full-Screen Background Image
st.markdown(f"""
    <style>
    .stApp {{
        background: url("data:image/jpg;base64,{background_base64}") no-repeat center center fixed;
        background-size: cover;
        width: 100vw;
        height: 100vh;
        margin: 0;
        padding: 10;
        overflow: -10;
    }}
    .stMarkdown, .stTextArea, .stButton, .stTitle, .stRadio {{
        color: white !important;
    }}
    </style>
    """, unsafe_allow_html=True)

#  Centered Title & Description with a Transparent Overlay for Readability
st.markdown("""
    <div style="text-align: center; background: rgba(0, 0, 0, 0.5); padding: 10px; border-radius: 10px;">
        <h1 style="color: white;">🐾 Pet Vet Chatbot</h1>
        <p style="color: lightgrey; font-style: italic;">Your AI-powered veterinary assistant</p>
    </div>
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
st.title("🐾 Pet Vet Chatbot - AI-Powered Veterinary Assistant")
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



