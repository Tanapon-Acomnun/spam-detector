import re
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# CONFIG
# =========================
MODEL_NAME = "SoftSkinz/spam-detector-model"

MAX_LEN = 128
TEMPERATURE = 2.0

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    return tokenizer, model, device

tokenizer, model, device = load_model()

# =========================
# PREPROCESSING
# =========================
def preprocess_input(text):
    # Remove email-style prefixes
    text = text.replace("Subject:", "")
    text = text.replace("Body:", "")

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()

# =========================
# PREDICTION
# =========================
def predict(text, threshold=0.6):
    text = preprocess_input(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # =========================
    # TEMPERATURE SCALING
    # =========================
    logits = outputs.logits / TEMPERATURE

    probs = torch.softmax(logits, dim=1)

    spam_prob = probs[0][1].item()
    ham_prob = probs[0][0].item()

    # =========================
    # SMARTER LABELING
    # =========================
    if spam_prob >= threshold:
        label = "Spam"
    else:
        label = "Not Spam"

    return {
        "label": label,
        "spam_prob": spam_prob,
        "ham_prob": ham_prob
    }

# =========================
# UI
# =========================
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Spam Email Detector")

st.caption(
    "AI-powered email spam classification using BERT."
)

# =========================
# THRESHOLD CONTROL
# =========================
threshold = st.slider(
    "Spam Sensitivity",
    min_value=0.30,
    max_value=0.90,
    value=0.60,
    step=0.05
)

# =========================
# INPUT
# =========================
user_input = st.text_area(
    "Paste email content:",
    height=300,
    placeholder="Paste an email here..."
)

# =========================
# PREDICT BUTTON
# =========================
if st.button("Predict"):

    if not user_input.strip():
        st.warning("Please enter email text.")
        st.stop()

    result = predict(user_input, threshold)

    spam_prob = result["spam_prob"]
    ham_prob = result["ham_prob"]
    label = result["label"]

    # =========================
    # RAW PROBABILITY
    # =========================
    st.subheader("Prediction")

    st.write(f"Spam Probability: {spam_prob:.2%}")

    # =========================
    # PROGRESS BAR
    # =========================
    st.progress(float(spam_prob))

    # =========================
    # INTERPRETATION
    # =========================
    if spam_prob < 0.40:
        confidence_text = "Likely Safe"
        color = "success"

    elif spam_prob < 0.60:
        confidence_text = "Uncertain / Mixed Signals"
        color = "warning"

    elif spam_prob < 0.85:
        confidence_text = "Suspicious"
        color = "warning"

    else:
        confidence_text = "Highly Likely Spam"
        color = "error"

    # =========================
    # RESULT DISPLAY
    # =========================
    if label == "Spam":
        st.error(f"🚨 {label}")
    else:
        st.success(f"✅ {label}")

    st.write(f"Assessment: {confidence_text}")

    # =========================
    # DEBUG INFO
    # =========================
    with st.expander("Technical Details"):
        st.write(f"Ham Probability: {ham_prob:.4f}")
        st.write(f"Spam Probability: {spam_prob:.4f}")
        st.write(f"Threshold: {threshold}")
        st.write(f"Temperature: {TEMPERATURE}")
        st.write(f"Max Length: {MAX_LEN}")

    # =========================
    # WARNING MESSAGE
    # =========================
    if label == "Spam":
        st.caption(
            "⚠️ This message contains patterns commonly associated with spam or phishing emails."
        )
    else:
        st.caption(
            "👍 This message appears relatively safe."
        )
