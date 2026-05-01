import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# LOAD MODEL
# =========================
MODEL_NAME = "SoftSkinz/spam-detector-model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# =========================
# PREDICTION FUNCTION (FIXED)
# =========================
def preprocess_input(text):
    text = text.replace("Subject:", "")
    text = text.replace("Body:", "")
    return text.strip()

def predict(text, threshold=0.7):
    text = preprocess_input(text)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    spam_prob = probs[0][1].item()
    ham_prob = probs[0][0].item()

    # Threshold-based decision
    if spam_prob > threshold:
        label = "Spam"
        confidence = spam_prob
    else:
        label = "Not Spam"
        confidence = ham_prob

    return label, confidence, spam_prob


# =========================
# UI CONFIG
# =========================
st.set_page_config(page_title="Spam Classifier", page_icon="📧")

st.title("📧 Spam Email Detector")

# Threshold control
threshold = st.slider(
    "Spam Sensitivity (Threshold)",
    min_value=0.5,
    max_value=0.9,
    value=0.7,
    step=0.05
)

# Input box
user_input = st.text_area(
    "Enter your email text here:",
    height=300
)

# =========================
# PREDICTION BUTTON
# =========================
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        label, confidence, spam_prob = predict(user_input, threshold)

        # Show raw probability
        st.write(f"📊 Spam Probability: {spam_prob:.2%}")

        # Result display
        if label == "Spam":
            st.error(f"🚨 {label}")
            st.write(f"Confidence: {confidence:.2%}")
            st.progress(spam_prob)
            st.caption("⚠️ This message shows spam-like patterns.")
        else:
            st.success(f"✅ {label}")
            st.write(f"Confidence: {confidence:.2%}")
            st.progress(1 - spam_prob)
            st.caption("👍 This message looks safe.")
