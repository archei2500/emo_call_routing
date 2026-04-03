import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SemanticEmotionClassifier:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SemanticEmotionClassifier, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = None, device: str = None):
        """
        Инициализация модели (происходит только один раз).

        Args:
            model_name: Имя модели на Hugging Face Hub
            device: Устройство ('cpu' или 'cuda')
        """
        if not self._initialized:
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            self.model_name = model_name or "cointegrated/rubert-tiny2-cedr-emotion-detection"

            # Загружаем токенизатор и модель
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()

            # Маппинг меток (из конфига модели)
            # 0: 'no_emotion', 1: 'joy', 2: 'sadness', 3: 'surprise', 4: 'fear', 5: 'anger'
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id

            self._initialized = True
            print(f"SemanticEmotionClassifier загружен на {self.device}")
            print(f"Доступные эмоции: {list(self.id2label.values())}")

    def predict(self, text: str) -> dict:
        """
        Определение эмоций в тексте.

        Args:
            text: Текст на русском языке

        Returns:
            Словарь с предсказанными эмоциями и вероятностями
        """
        # Токенизация
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        # Переносим на устройство
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Предсказание
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            # Multi-label классификация -> sigmoid
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]

        # Формируем словарь вероятностей для всех эмоций
        all_probabilities = {
            self.id2label[str(i)]: float(probabilities[i])
            for i in range(len(probabilities))
        }

        # Выбираем эмоции с вероятностью > 0.5 (включая neutral/no_emotion)
        threshold = 0.5
        detected_emotions = [
            {"emotion": self.id2label[str(i)], "probability": float(probabilities[i])}
            for i in range(len(probabilities))
            if probabilities[i] > threshold
        ]

        # Основная эмоция - с максимальной вероятностью (включая neutral)
        primary_emotion = max(all_probabilities, key=all_probabilities.get)
        primary_confidence = all_probabilities[primary_emotion]

        result = {
            "primary_emotion": primary_emotion,  # может быть 'no_emotion' (нейтральное)
            "primary_confidence": primary_confidence,
            "detected_emotions": detected_emotions,  # все эмоции выше порога
            "probabilities": all_probabilities
        }

        return result


# глобальный экземпляр
semantic_emotion_classifier = None


def get_semantic_emotion_classifier(model_name: str = None, device: str = None) -> SemanticEmotionClassifier:
    """
    Функция для получения глобального экземпляра классификатора эмоций.

    Args:
        model_name: Имя модели на Hugging Face Hub
        device: Устройство ('cpu' или 'cuda')

    Returns:
        Экземпляр SemanticEmotionClassifier
    """
    global semantic_emotion_classifier
    if semantic_emotion_classifier is None:
        semantic_emotion_classifier = SemanticEmotionClassifier(model_name, device)
    return semantic_emotion_classifier