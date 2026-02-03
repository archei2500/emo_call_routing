import torch
import torch.nn as nn
import librosa
import numpy as np
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel


class AgeGenderHead(nn.Module):
    def __init__(self, config, num_labels):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = AgeGenderHead(config, 1)
        self.gender = AgeGenderHead(config, 3)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)

        return hidden_states, logits_age, logits_gender


class AgeGenderPredictor:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AgeGenderPredictor, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path: str = None, device: str = None):
        """
        Инициализация модели (происходит только один раз).

        Args:
            model_path: Путь к локальной папке с моделью или имя модели в Hugging Face Hub
            device: Устройство для выполнения ('cpu' или 'cuda')
        """
        if not self._initialized:
            self.model_path = model_path or "audeering/wav2vec2-large-robust-24-ft-age-gender"
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            self.processor = Wav2Vec2Processor.from_pretrained(self.model_path)
            self.model = AgeGenderModel.from_pretrained(self.model_path)

            self.model.to(self.device)
            self.model.eval()

            self._initialized = True

    def _process_model_outputs(self, y):
        """
        Обработка выходов модели.

        Args:
            y: Кортеж из 3 тензоров, возвращаемых моделью

        Returns:
            age_output, female_prob, male_prob, child_prob
        """
        if len(y) == 3:
            combined = torch.hstack([y[1], y[2]])
            combined_np = combined.detach().cpu().numpy()[0]
            age_output = combined_np[0]
            female_prob = combined_np[1]
            male_prob = combined_np[2]
            child_prob = combined_np[3]
        else:
            print(f"Model outputs structure: {len(y)} tensors")
            for i, tensor in enumerate(y):
                print(f"  y[{i}].shape: {tensor.shape}")
            raise ValueError("Unexpected model output structure")

        return age_output, female_prob, male_prob, child_prob

    def predict(self, audio_array: np.ndarray, sampling_rate: int = 16000) -> dict:
        """
        Предсказание возраста и пола по аудио.

        Args:
            audio_array: NumPy массив с аудио данными
            sampling_rate: Частота дискретизации аудио

        Returns:
            Словарь с предсказаниями возраста и пола
        """
        y = self.processor(audio_array, sampling_rate=sampling_rate)
        y = y['input_values'][0]
        y = y.reshape(1, -1)
        y = torch.from_numpy(y).to(self.device)

        # предсказание
        with torch.no_grad():
            model_outputs = self.model(y)
            age_output, female_prob, male_prob, child_prob = self._process_model_outputs(model_outputs)

        predicted_age = round(100 * age_output)

        result = {
            "age": {
                "years": predicted_age,
                "raw_score": float(age_output)
            },
            "gender": {
                "probabilities": {
                    "female": float(female_prob),
                    "male": float(male_prob)
                },
                "predicted": "female" if female_prob > male_prob else "male"
            },
            "age_category": {
                "is_child": child_prob > 0.5 or predicted_age < 18,
                "child_probability": float(child_prob),
                "is_adult": child_prob <= 0.5 and predicted_age >= 18,
                "adult_probability": float(1 - child_prob)
            }
        }

        return result

    def predict_from_file(self, file_path: str, target_rate: int = 16000, duration: float = None) -> dict:
        """
        Предсказание по аудиофайлу.

        Args:
            file_path: Путь к аудиофайлу
            target_rate: Целевая частота дискретизации
            duration: Максимальная длительность для обработки (секунды)

        Returns:
            Словарь с предсказаниями
        """

        # Загружаем аудио
        sig, sr = librosa.load(
            file_path,
            sr=target_rate,
            duration=duration,
            mono=True,
            dtype=np.float32
        )

        return self.predict(sig, target_rate)


# глобальный экземпляр для использования в приложении
predictor = None


def get_age_gender_predictor(model_path: str = None, device: str = None) -> AgeGenderPredictor:
    """
    Функция для получения глобального экземпляра предсказателя.

    Args:
        model_path: Путь к модели (если None, будет использована модель по умолчанию)
        device: Устройство ('cpu' или 'cuda')

    Returns:
        Экземпляр AgeGenderPredictor
    """
    global predictor
    if predictor is None:
        predictor = AgeGenderPredictor(model_path, device)
    return predictor
