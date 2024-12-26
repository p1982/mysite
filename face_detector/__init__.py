from .load_model import load_pretrained_model

# Загружаем предобученную модель
model = load_pretrained_model()

if model is None:
    print("Продолжение работы без загруженной модели.")
