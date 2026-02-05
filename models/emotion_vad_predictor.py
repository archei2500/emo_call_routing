import torch.nn as nn
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2PreTrainedModel, Wav2Vec2Model


class RegressionHead(nn.Module):
    """Classification head."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    """Speech emotion classifier."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits


class EmotionVADPredictor:
    """
    Класс-синглтон для предсказания эмоций по аудио.
    Модель инициализируется один раз при первом использовании.

    Модель предсказывает 4 измерения эмоций:
    1. Arousal (активация/возбуждение)
    2. Valence (валентность/приятность)
    3. Dominance (доминирование)

    Значения в диапазоне примерно от 0 до 1.
    """
    _instance = None
    _initialized = False

    # Названия эмоциональных измерений
    EMOTION_DIMENSIONS = ["arousal", "dominance", "valence"]

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmotionVADPredictor, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path: str = None, device: str = None):
        """
        Инициализация модели (происходит только один раз).

        Args:
            model_path: Путь к локальной папке с моделью или имя модели в Hugging Face Hub
            device: Устройство для выполнения ('cpu' или 'cuda')
        """
        if not self._initialized:
            self.model_path = model_path or "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            self.processor = Wav2Vec2Processor.from_pretrained(self.model_path)
            self.model = EmotionModel.from_pretrained(self.model_path)

            self.model.to(self.device)
            self.model.eval()  # режим инференса

            self._initialized = True

    def predict(self, audio_array: np.ndarray, sampling_rate: int = 16000) -> dict:
        """
        Предсказание эмоциональных измерений по аудио.

        Args:
            audio_array: NumPy массив с аудио данными
            sampling_rate: Частота дискретизации аудио (должна быть 16000)

        Returns:
            Словарь с предсказаниями эмоциональных измерений
        """
        inputs = self.processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.to(self.device)

        # предсказание
        with torch.no_grad():
            hidden_states, logits = self.model(input_values)
            logits = logits.cpu()
            predictions = logits.numpy()[0]

        result = {
            "raw_predictions": predictions.tolist(),
            "emotions": {}
        }

        for i, dim_name in enumerate(self.EMOTION_DIMENSIONS):
            result["emotions"][dim_name] = float(predictions[i])

        return result

    def predict_from_file(self, file_path: str, target_rate: int = 16000, duration: float = None) -> dict:
        """
        Предсказание эмоций по аудиофайлу.

        Args:
            file_path: Путь к аудиофайлу
            target_rate: Целевая частота дискретизации
            duration: Максимальная длительность для обработки (секунды)

        Returns:
            Словарь с предсказаниями эмоций
        """

        sig, sr = librosa.load(
            file_path,
            sr=target_rate,
            duration=duration,
            mono=True,
            dtype=np.float32
        )

        return self.predict(sig, target_rate)


# глобальный экземпляр для использования в приложении
emotion_predictor = None


def get_emotion_vad_predictor(model_path: str = None, device: str = None) -> EmotionVADPredictor:
    """
    Функция для получения глобального экземпляра предсказателя эмоций.

    Args:
        model_path: Путь к модели (если None, будет использована модель по умолчанию)
        device: Устройство ('cpu' или 'cuda')

    Returns:
        Экземпляр EmotionPredictor
    """
    global emotion_predictor
    if emotion_predictor is None:
        emotion_predictor = EmotionVADPredictor(model_path, device)
    return emotion_predictor

