import streamlit as st
import numpy as np
import joblib
from pathlib import Path
import sys
import os
import time
import pandas as pd

# Ensure we can import from parent directory
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))
os.chdir(str(parent_dir))

from src.utils.preprocessing import preprocess_tweet
from src.models.bert_wrapper import BertWrapper

# -------------------------------------------------
# Streamlit Layout & Styling
# -------------------------------------------------
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #1f4037, #99f2c8);
        color: white;
    }
    .stButton>button {
        background-color: #FF5733;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Sentiment Analysis — LR, LSTM, GRU, BERT")
st.markdown("---")

MODEL_CHOICES = ["LogisticRegression", "LSTM", "GRU", "BERT"]
model_choice = st.sidebar.selectbox("⚙️ Select Model", MODEL_CHOICES)

# -------------------------------------------------
# Cached Loaders
# -------------------------------------------------
@st.cache_resource
def load_lr(path="models/lr/pipeline.joblib"):
    return joblib.load(path)

@st.cache_resource
def load_lstm(path="models/lstm"):
    from tensorflow.keras.models import load_model as _load_model
    tok_path = Path(path) / "tokenizer.joblib"
    model_path = Path(path) / "model_final.keras"
    tok = joblib.load(tok_path)
    mdl = _load_model(str(model_path))
    return tok, mdl

@st.cache_resource
def load_gru(path="models/gru"):
    from tensorflow.keras.models import load_model as _load_model
    tok_path = Path(path) / "tokenizer.joblib"
    model_path = Path(path) / "model_final.keras"
    tok = joblib.load(tok_path)
    mdl = _load_model(str(model_path))
    return tok, mdl

@st.cache_resource
def load_bert(path="models/bert"):
    return BertWrapper(path)

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
def lstm_predict(tokenizer, model, text, max_len=80):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    return model.predict(seq)[0]

def gru_predict(tokenizer, model, text, max_len=80):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    return model.predict(seq)[0]

# -------------------------------------------------
# Main UI
# -------------------------------------------------
st.write(f"🔍 Using Model: **{model_choice}**")
text = st.text_area("✏️ Enter text to analyze:", height=140)

if st.button("🚀 Predict"):
    if not text.strip():
        st.warning("⚠️ Please enter some text.")
        st.stop()

    with st.spinner("✨ Analyzing sentiment..."):
        time.sleep(1.5)  # simulate loading

    cleaned_text = preprocess_tweet(text)
    labels = ["negative", "neutral", "positive"]

    try:
        if model_choice == "LogisticRegression":
            model = load_lr()
            probs = model.predict_proba([cleaned_text])[0]
        elif model_choice == "LSTM":
            tokenizer, model = load_lstm()
            probs = lstm_predict(tokenizer, model, cleaned_text, max_len=80)
        elif model_choice == "GRU":
            tokenizer, model = load_gru()
            probs = gru_predict(tokenizer, model, cleaned_text, max_len=80)
        else:
            bert = load_bert()
            probs = bert.predict_proba([cleaned_text])[0]
    except Exception as e:
        st.error(f"❌ Model error: {e}")
        st.stop()

    pred_idx = int(np.argmax(probs))

    # 🎉 Fun animations
    if labels[pred_idx] == "positive":
        st.balloons()
    elif labels[pred_idx] == "negative":
        st.snow()

    st.subheader(f"🎯 Prediction: **{labels[pred_idx].upper()}**")
    st.metric(label="Confidence", value=f"{float(probs[pred_idx]):.2f}")

    # 📊 Probability visualization
    df = pd.DataFrame({"Class": labels, "Probability": [float(p) for p in probs]})
    st.bar_chart(df.set_index("Class"))