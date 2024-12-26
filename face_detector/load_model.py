import os
from tensorflow.keras.models import load_model

# Указываем путь к модели
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/models.h5")

def load_pretrained_model():
    try:
        # Загружаем модель
        model = load_model(MODEL_PATH)
        print("Модель успешно загружена.")
        return model
    except OSError:
        print(f"Файл модели не найден по пути {MODEL_PATH}. Пожалуйста, добавьте файл models.h5.")
        return None
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return None
