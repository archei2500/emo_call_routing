import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor


class AdultChildDetector:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AdultChildDetector, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path: str = None, device: str = None):
        """
        Инициализация модели (происходит только один раз).

        Args:
            model_path: Путь к локальной папке с моделью или имя модели в Hugging Face Hub
            device: Устройство ('cpu' или 'cuda')
        """
        if not self._initialized:
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            self.model_path = model_path or "bookbot/wav2vec2-adult-child-cls"

            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_path)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_path)
            self.model = self.model.to(self.device)
            self.model.eval()

            self.class_labels = ["adult", "child"]

            self._initialized = True
            print(f"AdultChildDetector загружен на {self.device}")

    def predict(self, audio_array: np.ndarray, sampling_rate: int = 16000) -> dict:
        """
        Предсказание категории (ребёнок/взрослый) из аудио.

        Args:
            audio_array: NumPy массив с аудио данными
            sampling_rate: Частота дискретизации аудио

        Returns:
            Словарь с предсказаниями
        """
        if sampling_rate != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)

        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=0)

        # Подготовка входных данных
        inputs = self.processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Предсказание
        with torch.no_grad():
            logits = self.model(**inputs).logits

        probabilities = torch.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        predicted_label = self.class_labels[predicted_class_id]
        confidence = probabilities[0][predicted_class_id].item()

        probs_cpu = probabilities.cpu().numpy()[0]

        result = {
            "predicted": predicted_label,
            "confidence": confidence,
            "probabilities": {
                "adult": float(probs_cpu[0]),
                "child": float(probs_cpu[1])
            }
        }

        return result

    def predict_from_file(self, file_path: str, target_rate: int = 16000) -> dict:
        """
        Предсказание по аудиофайлу.

        Args:
            file_path: Путь к аудиофайлу
            target_rate: Целевая частота дискретизации

        Returns:
            Словарь с предсказаниями
        """
        sig, sr = librosa.load(file_path, sr=target_rate, mono=True, dtype=np.float32)
        return self.predict(sig, target_rate)


# глобальный экземпляр
detector = None


def get_adult_child_detector(model_path: str = None, device: str = None) -> AdultChildDetector:
    """
    Функция для получения глобального экземпляра детектора.

    Args:
        model_path: Путь к модели (если None, будет использована модель по умолчанию)
        device: Устройство ('cpu' или 'cuda')

    Returns:
        Экземпляр AdultChildDetector
    """
    global detector
    if detector is None:
        detector = AdultChildDetector(model_path, device)
    return detector