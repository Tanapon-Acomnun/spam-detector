import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model
MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Prediction function
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    label = "Spam" if torch.argmax(probs)==1 else "Not Spam"
    confidence = float(torch.max(probs))

    return label, confidence

# UI
st.set_page_config(page_title="Spam Classifier", page_icon="📧")

st.title("📧 Spam Email Detector")

user_input = st.text_area("Enter your email text here:", height=300)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        label, confidence = predict(user_input)

        if label == "Spam":
            st.error(f"🚨 {label}")
            st.write(f"Confidence: {confidence:.2%}")
            st.progress(confidence)
        else:
            st.success(f"✅ {label}")
            st.write(f"Confidence: {confidence:.2%}")
            st.progress(confidence)
