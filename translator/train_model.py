import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Prepare the dataset
texts = [
    "Hello", "Good morning", "Good evening", "Good night", "How are you?", "I'm fine, thank you", "Please", "Thank you",
    "You're welcome", "Excuse me", "I'm sorry", "Yes", "No", "Maybe", "I don't know", "What is your name?",
    "My name is ...", "Nice to meet you", "Goodbye", "See you later", "Where is the restroom?", "How do I get to the airport?",
    "Can you help me?", "I need a taxi", "How much does it cost?", "Left", "Right", "Straight ahead", "Is it far?",
    "Where is the nearest hotel?", "I'm hungry", "I'm thirsty", "Can I have the menu, please?", "I would like ...",
    "What do you recommend?", "Check, please", "This is delicious", "I am vegetarian", "Do you have any vegan options?",
    "No sugar, please", "How much is this?", "Do you have this in a different size?", "I'm just looking",
    "Can I pay with a card?", "Do you accept cash?", "Where is the fitting room?", "It’s too expensive", "Do you have any discounts?",
    "Can I get a receipt?", "Thank you for your help", "Help!", "Call the police", "I need a doctor", "I'm lost",
    "Is there a hospital nearby?", "I don't feel well", "I need medicine", "Where is the pharmacy?", "I have an allergy",
    "Can you call an ambulance?"
]

translations = [
    "Привіт", "Доброго ранку", "Доброго вечора", "Добраніч", "Як справи?", "У мене все добре, дякую", "Будь ласка", "Дякую",
    "Будь ласка", "Перепрошую", "Вибачте", "Так", "Ні", "Можливо", "Я не знаю", "Як вас звати?", "Мене звати ...",
    "Приємно познайомитися", "До побачення", "До зустрічі", "Де туалет?", "Як доїхати до аеропорту?", "Ви можете мені допомогти?",
    "Мені потрібне таксі", "Скільки це коштує?", "Ліворуч", "Праворуч", "Прямо", "Це далеко?", "Де найближчий готель?",
    "Я голодний", "Я хочу пити", "Можна меню, будь ласка?", "Я хотів би ...", "Що ви порадите?", "Рахунок, будь ласка",
    "Це смачно", "Я вегетаріанець", "У вас є веганські страви?", "Без цукру, будь ласка", "Скільки це коштує?",
    "У вас є це іншого розміру?", "Я просто дивлюся", "Можна оплатити карткою?", "Ви приймаєте готівку?",
    "Де примірочна?", "Це надто дорого", "У вас є знижки?", "Можна отримати чек?", "Дякую за допомогу", "Допоможіть!",
    "Викличте поліцію", "Мені потрібен лікар", "Я загубився", "Чи є тут поблизу лікарня?", "Мені недобре", "Мені потрібні ліки",
    "Де аптека?", "У мене алергія", "Можете викликати швидку допомогу?"
]

# Debugging function for tokenization
def debug_tokenization(input_tokenizer, output_tokenizer):
    print("[DEBUG] Input Tokenizer Vocabulary:", input_tokenizer.word_index)
    print("[DEBUG] Output Tokenizer Vocabulary:", output_tokenizer.word_index)

# Function to train and save the model
def train_and_save_model():
    # Normalize the texts
    texts_normalized = [text.lower().strip() for text in texts]
    translations_normalized = [translation.lower().strip() for translation in translations]

    # Tokenizer for input and output
    input_tokenizer = Tokenizer(num_words=10000, filters="", oov_token="<OOV>")
    output_tokenizer = Tokenizer(num_words=10000, filters="", oov_token="<OOV>")

    # Fit the tokenizers
    input_tokenizer.fit_on_texts(texts_normalized)
    output_tokenizer.fit_on_texts(translations_normalized)

    # Debug tokenizers
    debug_tokenization(input_tokenizer, output_tokenizer)

    # Convert texts to sequences
    X_train = input_tokenizer.texts_to_sequences(texts_normalized)
    y_train = output_tokenizer.texts_to_sequences(translations_normalized)

    # Debug tokenized sequences
    print("[DEBUG] Tokenized Input Sequences:", X_train)
    print("[DEBUG] Tokenized Output Sequences:", y_train)

    # Pad the sequences to ensure uniform length
    X_train = pad_sequences(X_train, maxlen=5, padding="post")
    y_train = pad_sequences(y_train, maxlen=5, padding="post")

    # Define the model
    model = Sequential([
        Embedding(input_dim=10000, output_dim=64, input_length=5),
        LSTM(64, return_sequences=True),
        Dense(10000, activation="softmax"),
    ])

    # Compile the model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Reshape y_train to add the required dimension for sparse_categorical_crossentropy
    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)

    # Train the model
    print("[INFO] Training the model...")
    model.fit(X_train, y_train, epochs=10)

    # Define the translator models directory
    translator_dir = os.path.join(os.getcwd(), "translator/models/")

    # Ensure the directory exists
    os.makedirs(translator_dir, exist_ok=True)

    # Save the model in `.keras` format
    print("[INFO] Saving the model and tokenizers...")
    model.save(os.path.join(translator_dir, "translation_model.keras"))

    # Save the tokenizers
    with open(os.path.join(translator_dir, "input_tokenizer.pkl"), "wb") as f:
        pickle.dump(input_tokenizer, f)

    with open(os.path.join(translator_dir, "output_tokenizer.pkl"), "wb") as f:
        pickle.dump(output_tokenizer, f)

    print("[INFO] Model and tokenizers saved successfully in 'translator/models/'.")

# Main execution block
if __name__ == "__main__":
    train_and_save_model()
