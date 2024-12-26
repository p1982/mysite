import os
from tensorflow.keras.models import load_model

# Define the path to the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models.h5")

# Load the model
model = None
try:
    model = load_model(MODEL_PATH)
    print("Модель успешно загружена.")
except OSError:
    print(f"Файл модели не найден по пути {MODEL_PATH}. Пожалуйста, добавьте файл models.h5.")
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
