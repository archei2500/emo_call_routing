from llama_cpp import Llama
import torch
import json


class LLaMa:
    _instance = None
    _initialized = False

    schema = {
        "type": "object",
        "properties": {
            "emotions": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral", "angry", "happy", "sad", "surprised", "fear", "disgust"]
                },
                "minItems": 1,
                "maxItems": 3,  # ← ограничиваем, чтобы не плодить лишнее
                "uniqueItems": True,
                "description": "Основные эмоции в порядке убывания интенсивности"
            },
            "topics": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["сайт", "бытовая техника", "мобильные телефоны", "компьютеры", "другое"]
                },
                "minItems": 1,
                "maxItems": 2,
                "uniqueItems": True
            },
            "agree": {
                "type": "string",
                "enum": ["да", "нет", "неясно"]
            }
        },
        "required": ["emotions", "topics", "agree"],
        "additionalProperties": False
    }

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LLaMa, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path: str = None, device: str = "auto"):
        """
        Инициализация модели (происходит только один раз).

        Args:
            model_path: Путь к локальной папке с моделью
        """
        if not self._initialized:
            self.model_path = model_path

            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            n_gpu = -1 if device == "cuda" else 0

            try:
                self.model = Llama(
                    model_path=self.model_path,
                    n_ctx=4096,  # длина контекста
                    n_threads=0 if n_gpu == 0 else 6,  # 0 = все ядра CPU
                    n_gpu_layers=n_gpu,
                    verbose=False  # меньше логов
                )
            except Exception as e:
                print(f"Ошибка при загрузке: {e}")
                print("Fallback → чистый CPU режим")

                self.model = Llama(
                    model_path=model_path,
                    n_ctx=4096,
                    n_threads=0,
                    n_gpu_layers=0,
                    verbose=False,
                )

            self._initialized = True

    def get_response(self, text: str, voice_emotion: str) -> dict:
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты строгий классификатор эмоций и тем."
                    "Смотри ТОЛЬКО на текст и на эмоцию по голосу.\n"
                    "1. emotions — только явно выраженные эмоции из списка (до 3)\n"
                    "2. topics — только те категории, которые явно есть в тексте (1–2)\n"
                    "3. agree — совпадает ли вывод об эмоциях по голосу с текстом\n"
                    "Отвечай ИСКЛЮЧИТЕЛЬНО валидным JSON по схеме. Никакого другого текста."
                )
            },
            {
                "role": "user",
                "content": f"Текст: {text}\n\n"
                           f"Возможные эмоции: positive, negative, neutral, angry, happy, sad, surprised, fear, disgust\n"
                           f"Возможные темы: сайт, бытовая техника, мобильные телефоны, компьютеры, другое\n\n"
                           f"Эмоция по голосу (из другой модели): {voice_emotion}\n"
                           f"Согласен с этим выводом? Варианты: да, нет, неясно"
            }
        ]

        response = self.model.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object", "schema": self.schema, "strict": True},
            temperature=0.05,
            max_tokens=120
        )

        try:
            return json.loads(response["choices"][0]["message"]["content"])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Ошибка парсинга ответа: {e}")
            print(f"Ответ модели: {response}")
            # возвращаем пустой словарь или значение по умолчанию
            return {
                "emotions": ["neutral"],
                "topics": ["другое"],
                "agree": "неясно"
            }


# глобальный экземпляр для использования в приложении
llm = None


def get_llm(model_path: str = None, device: str = "auto") -> LLaMa:
    """
    Функция для получения глобального экземпляра модели для распознавания речи.

    Args:
        model_path: Путь к модели (если None, будет использована модель по умолчанию)
        device: Устройство ('cpu' или 'cuda')

    Returns:
        Экземпляр LLaMa
    """
    global llm
    if llm is None:
        llm = LLaMa(model_path, device)
    return llm
