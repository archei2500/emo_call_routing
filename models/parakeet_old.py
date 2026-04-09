import torch
import librosa
import numpy as np
import nemo.collections.asr as nemo_asr


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
        """
        if not self._initialized:
            self.model_path = model_path or "nvidia/parakeet-tdt-0.6b-v3"
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            self.model = nemo_asr.models.ASRModel.restore_from(restore_path=self.model_path)

            self.model.to(self.device)
            self.model.eval()

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

        # предсказание
        with torch.no_grad():
            output = self.model.transcribe(audio=audio_array, batch_size=1)

        result = output[0].text

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

        # Загружаем аудио
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
        Экземпляр LLaMa
    """
    global asr_model
    if asr_model is None:
        asr_model = Parakeet(model_path, device)
    return asr_model
