import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class IntentClassifier:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(IntentClassifier, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path: str = None, device: str = None):
        """
        Инициализация модели (происходит только один раз).

        Args:
            model_path: Путь к локальной папке с сохранённой моделью
            device: Устройство ('cpu' или 'cuda')
        """
        if not self._initialized:
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            self.model_path = model_path or "model_files/rubert_tiny_intent"

            # Загружаем токенизатор и модель
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model = self.model.to(self.device)
            self.model.eval()

            # Получаем маппинг меток
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id

            self._initialized = True
            print(f"IntentClassifier загружен на {self.device}")

    def predict(self, text: str) -> dict:
        """
        Классификация интента пользователя по тексту.

        Args:
            text: Текст на русском языке

        Returns:
            Словарь с предсказанным интентом и вероятностями
        """
        # Токенизация
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt"
        )

        # Переносим на устройство
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Предсказание
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        # Получаем предсказанный класс
        predicted_class_id = np.argmax(probabilities)
        predicted_intent = self.id2label[predicted_class_id]
        confidence = probabilities[predicted_class_id]

        # Формируем словарь вероятностей по всем интентам
        all_probabilities = {
            self.id2label[i]: float(probabilities[i])
            for i in range(len(probabilities))
        }

        result = {
            "predicted_intent": predicted_intent,
            "confidence": float(confidence),
            "probabilities": all_probabilities
        }

        return result


# глобальный экземпляр
intent_classifier = None


def get_intent_classifier(model_path: str = None, device: str = None) -> IntentClassifier:
    """
    Функция для получения глобального экземпляра классификатора интентов.

    Args:
        model_path: Путь к локальной папке с моделью
        device: Устройство ('cpu' или 'cuda')

    Returns:
        Экземпляр IntentClassifier
    """
    global intent_classifier
    if intent_classifier is None:
        intent_classifier = IntentClassifier(model_path, device)
    return intent_classifier
