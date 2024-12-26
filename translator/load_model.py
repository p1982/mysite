import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define paths
TRANSLATOR_DIR = os.path.join(os.getcwd(), "translator/models/")
MODEL_PATH = os.path.join(TRANSLATOR_DIR, "translation_model.keras")
INPUT_TOKENIZER_PATH = os.path.join(TRANSLATOR_DIR, "input_tokenizer.pkl")
OUTPUT_TOKENIZER_PATH = os.path.join(TRANSLATOR_DIR, "output_tokenizer.pkl")

# Load model and tokenizers
print("[INFO] Loading model and tokenizers...")
model = load_model(MODEL_PATH)
with open(INPUT_TOKENIZER_PATH, "rb") as f:
    input_tokenizer = pickle.load(f)
with open(OUTPUT_TOKENIZER_PATH, "rb") as f:
    output_tokenizer = pickle.load(f)

print("[DEBUG] Model and tokenizers loaded successfully.")

# Function to preprocess input text
def preprocess_input(text):
    tokenized = input_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(tokenized, maxlen=5, padding="post")
    return padded

# Function to apply temperature scaling to logits
def apply_temperature(logits, temperature=1.0):
    logits = np.log(logits + 1e-9) / temperature
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

# Translate function
def translate(word):
    try:
        print("[DEBUG] Starting translation process...")
        input_data = preprocess_input(word)
        print(f"[DEBUG] Preprocessed input: {input_data}")

        # Predict probabilities
        prediction = model.predict(input_data)
        print(f"[DEBUG] Prediction raw output: {prediction}")

        # Apply temperature scaling
        temperature = 0.8
        prediction = apply_temperature(prediction, temperature=temperature)
        print(f"[DEBUG] Prediction with temperature applied: {prediction}")

        # Decode token IDs
        predicted_token_ids = np.argmax(prediction, axis=-1)
        print(f"[DEBUG] Predicted Token IDs: {predicted_token_ids}")

        # Decode text
        decoded_words = []
        for token_id in predicted_token_ids.flatten():
            word = output_tokenizer.index_word.get(token_id, "[Unknown]")
            decoded_words.append(word)
            print(f"Token ID: {token_id}, Word: {word}")

        decoded_text = " ".join(word for word in decoded_words if word != "[Unknown]" and word != "<pad>")
        print(f"[DEBUG] Decoded Text: {decoded_text}")

        return decoded_text if decoded_text else "Translation failed"
    except Exception as e:
        print(f"[ERROR] Translation failed: {e}")
        return "Translation Error"

# Test the translation function
word = "hello world"
print(f"Translated '{word}': {translate(word)}")
