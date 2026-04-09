import torch
import gigaam
import tempfile
import soundfile as sf
import numpy as np


class EmotionClassifier:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmotionClassifier, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = None, model_dir: str = None, device: str = None):
        """
        Инициализация модели (происходит только один раз).

        Args:
            model_name: Имя модели ('emo' или другая)
            model_dir: Путь к локальной папке с моделью (если None, скачивается из интернета в кэш)
            device: Устройство ('cpu' или 'cuda')
        """
        if not self._initialized:
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            self.model_name = model_name or "emo"

            # Загрузка модели: если передан model_dir, загружаем локально, иначе из интернета
            if model_dir is not None:
                self.model = gigaam.load_model(self.model_name, model_dir=model_dir)
            else:
                self.model = gigaam.load_model(self.model_name)

            # Переносим на нужное устройство
            self.model = self.model.to(self.device)

            self._initialized = True
            print(f"EmotionClassifier загружен на {self.device}")

    def predict(self, audio_array) -> dict:
        """
        Предсказание эмоций из аудио.

        Args:
            audio_array: numpy массив с аудио данными ИЛИ путь к файлу (str)

        Returns:
            Словарь с вероятностями эмоций (angry, sad, neutral, positive)
        """
        # Если передан numpy-массив, сохраняем во временный файл
        if isinstance(audio_array, np.ndarray):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_path = tmp_file.name
            # Сохраняем массив как wav-файл (предполагается частота 16000 Гц)
            sf.write(temp_path, audio_array, 16000)
            result = self.model.get_probs(temp_path)
            # Удаляем временный файл
            import os
            os.unlink(temp_path)
        else:
            # Если передан путь к файлу (строка)
            result = self.model.get_probs(audio_array)

        emotion2prob = result  # get_probs уже возвращает словарь

        result = {
            "probabilities": emotion2prob,
            "predicted": max(emotion2prob, key=emotion2prob.get)
        }

        return result

    def predict_from_file(self, file_path: str) -> dict:
        """
        Предсказание эмоций по аудиофайлу.

        Args:
            file_path: Путь к аудиофайлу

        Returns:
            Словарь с вероятностями эмоций
        """
        return self.predict(file_path)


# глобальный экземпляр
emotion_classifier = None


def get_emotion_classifier(model_name: str = None, model_dir: str = None, device: str = None) -> EmotionClassifier:
    """
    Функция для получения глобального экземпляра классификатора эмоций.

    Args:
        model_name: Имя модели ('emo' или другая)
        model_dir: Путь к локальной папке с моделью (если None, скачивается из интернета)
        device: Устройство ('cpu' или 'cuda')

    Returns:
        Экземпляр EmotionClassifier
    """
    global emotion_classifier
    if emotion_classifier is None:
        emotion_classifier = EmotionClassifier(model_name, model_dir, device)
    return emotion_classifier
