

                🐾 Pet Vet Chatbot - AI-Powered Veterinary Assistant
Welcome to Pet Vet Chatbot, an AI-powered veterinary assistant that helps pet owners understand their pets' symptoms and get initial guidance based on VetBERT classification and GPT-4 responses.

This application leverages deep learning models for symptom analysis and OpenAI's GPT-4 for providing expert insights. The language classification model was adopted from The Domain Adaptation and Instance Selection for Disease Syndrome Classification over Veterinary Clinical Notes (Hur et al., BioNLP 2020). With the use of OpenAI's assistant GPT, we further humanized the suggested diagnosis or possible conditions for pets.

⚠️ Disclaimer: This chatbot is not a substitute for professional veterinary care. It should only be used for triage or an initial opinion until immediate care can be provided by a licensed veterinarian.

                            How It Works
The Pet Vet Chatbot follows a two-stage AI method to analyze and explain pet health conditions:

1️⃣ VetBERTDx Model (Classification)
The VetBERTDx model is a domain-adapted BERT-based model trained on veterinary clinical notes.
It classifies input symptoms into possible conditions or syndromes.
We filter out irrelevant classifications, such as "repeat", and extract the top two most likely conditions.
2️⃣ GPT-4 Vet Assistant (Humanized Explanation)
The top two VetBERT classifications are not displayed directly to the user.
Instead, these classifications are sent to GPT-4, which generates a human-friendly response explaining:
What the condition means
Possible causes
Next steps for the pet owner
Whether immediate veterinary care is needed
This two-stage approach ensures that the raw AI output is transformed into an intuitive and meaningful veterinary insight.

🐕 Example Usage
User Input:

"My dog has been vomiting for two days and seems very tired."

GPT-4 Advice:

"The AI model suggests that this could be Gastroenteritis or Pancreatitis. Gastroenteritis is commonly caused by dietary indiscretion, infections, or toxins. If your pet is lethargic, dehydrated, or vomiting persistently, consult a veterinarian immediately."

🤝 Contributing
If you’d like to improve this project, feel free to fork the repo, create a feature branch, and submit a pull request!

