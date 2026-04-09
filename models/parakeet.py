import torch
import librosa
import numpy as np
import onnx_asr


class Parakeet:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Parakeet, cls).__new__(cls)
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

            # Если передан локальный путь, загружаем оттуда, иначе с HuggingFace
            if model_path is not None:
                self.model = onnx_asr.load_model(model_path)
            else:
                self.model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")

            self._initialized = True

    def transcribe(self, audio_array: np.ndarray, sampling_rate: int = 16000) -> str:
        """
        Распознавание речи из аудио.

        Args:
            audio_array: NumPy массив с аудио данными
            sampling_rate: Частота дискретизации аудио

        Returns:
            Строка с распознанным текстом
        """
        if audio_array.ndim > 2:
            audio_array = audio_array.squeeze(0)

        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        result = self.model.recognize(audio_array, sample_rate=sampling_rate)

        return result

    def transcribe_from_file(self, file_path: str, target_rate: int = 16000) -> str:
        """
        Предсказание по аудиофайлу.

        Args:
            file_path: Путь к аудиофайлу
            target_rate: Целевая частота дискретизации

        Returns:
            Строка с распознанным текстом
        """
        sig, sr = librosa.load(
            file_path,
            sr=target_rate,
            mono=True,
            dtype=np.float32
        )

        return self.transcribe(sig, target_rate)


# глобальный экземпляр для использования в приложении
asr_model = None


def get_asr_model(model_path: str = None, device: str = None) -> Parakeet:
    """
    Функция для получения глобального экземпляра модели для распознавания речи.

    Args:
        model_path: Путь к модели (если None, будет использована модель по умолчанию)
        device: Устройство ('cpu' или 'cuda')

    Returns:
        Экземпляр Parakeet
    """
    global asr_model
    if asr_model is None:
        asr_model = Parakeet(model_path, device)
    return asr_model