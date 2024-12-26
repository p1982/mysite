import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
# Загружаем предобученную модель MobileNetV2
model = MobileNetV2(weights="imagenet")

# Сохраняем модель в формате .h5
model.save('./face_detector/models/models.h5')
print("Модель успешно сохранена в face_detector/models/models.h5")

# # Define the input shapes for both branches
# sequence_input = Input(shape=(20,), name="sequence_input")  # For sequence-based tasks
# numeric_input = Input(shape=(5,), name="numeric_input")     # For numeric input

# # Define the sequence processing branch
# x1 = Embedding(10000, 256, input_length=20)(sequence_input)
# x1 = LSTM(256, return_sequences=False)(x1)  # Return the last output

# # Define the numeric input processing branch
# x2 = Dense(10, activation="relu")(numeric_input)

# # Combine the two branches
# combined = Dense(10, activation="relu")(x1)
# output = Dense(1, activation="sigmoid")(combined)

# # Create the model
# model = Model(inputs=[sequence_input, numeric_input], outputs=output)

# # Compile the model (this step is required before saving)
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# # # Define the path to save the model
# # model_dir = os.path.join(os.path.dirname(__file__), "./translator/models")
# # if not os.path.exists(model_dir):
# #     os.makedirs(model_dir)
# # model_path = os.path.join(model_dir, "translation_model.h5")

# # Save the model
# model.save('./translator/models/models.h5')
# print(f"Модель успешно сохранена в")



# Define the input shapes for both branches
sequence_input = Input(shape=(20,), name="sequence_input")  # For sequence-based tasks
numeric_input = Input(shape=(5,), name="numeric_input")     # For numeric input

# Define the sequence processing branch
x1 = Embedding(10000, 256, input_length=20)(sequence_input)
x1 = LSTM(256, return_sequences=False)(x1)  # Return the last output

# Define the numeric input processing branch
x2 = Dense(10, activation="relu")(numeric_input)

# Combine the two branches
combined = Dense(10, activation="relu")(x1)
output = Dense(1, activation="sigmoid")(combined)

# Create the model
model = Model(inputs=[sequence_input, numeric_input], outputs=output)

# Compile the model (this step is required before saving)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Define the path to save the model in Keras format
model_dir = "./translator/models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, "translation_model.keras")

# Save the model in Keras format
model.save(model_path)
print(f"Model successfully saved in Keras format at: {model_path}")
