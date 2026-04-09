from llama_cpp import Llama
import torch
import json
import os
from llama_cpp import LlamaGrammar


class QwenAnalyzer:
    _instance = None
    _initialized = False

    # GBNF-грамматика для строгого JSON
    JSON_GRAMMAR = r"""
    root ::= "{" ws "\"emotions\"" ws ":" ws "[" emotions "]" ws "," ws "\"topics\"" ws ":" ws "[" topics "]" ws "}"
    emotions ::= emotion ("," ws emotion)*
    emotion ::= "\"" ("negative" | "positive" | "neutral" | "angry" | "happy" | "sad" | "surprised" | "fear" | "disgust") "\""
    topics ::= topic ("," ws topic)*
    topic ::= "\"" ("сайт" | "бытовая техника" | "мобильные телефоны" | "компьютеры" | "другое") "\""
    ws ::= [ \t\n]*
    """

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(QwenAnalyzer, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path: str = None, device: str = "auto"):
        """
        Инициализация модели Qwen (происходит только один раз).
        """
        if not self._initialized:
            if not model_path or not os.path.isfile(model_path):
                raise ValueError("Укажите правильный путь к GGUF-файлу модели Qwen.")

            self.model_path = model_path

            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            n_gpu_layers = -1 if device == "cuda" else 0  # -1 = весь на GPU, 0 = только CPU

            try:
                self.model = Llama(
                    model_path=self.model_path,
                    n_ctx=4096,
                    n_threads=0,               # 0 = использовать все доступные ядра
                    n_gpu_layers=n_gpu_layers,
                    chat_format="chatml",
                    verbose=False
                )
            except Exception as e:
                print(f"Ошибка загрузки модели: {e}")
                print("Пробуем чистый CPU-режим...")
                self.model = Llama(
                    model_path=self.model_path,
                    n_ctx=4096,
                    n_threads=0,
                    n_gpu_layers=0,
                    chat_format="chatml",
                    verbose=False
                )

            self._initialized = True

    def get_response(self, text: str) -> dict:
        messages = [
            {
                "role": "system",
                "content": (
                    "Отвечай ТОЛЬКО валидным JSON.\n"
                    "Ты классификатор эмоций и тем в контакт-центре.\n"
                    "Получаешь текст звонка.\n"
                    "1. emotions — эмоции из списка (возможные эмоции), до 3 штук\n"
                    "2. topics — только темы, которые очевидны (есть синонимы, четкий контекст)\n"
                    "Будь строгим, не пытайся угадывать, если информации недостаточно.\n"
                    "Примеры:\n"
                    "Текст: Холодильник вообще умер после покупки. Стыд вам!\n"
                    "{'emotions': ['angry'], 'topics': ['бытовая техника']}\n"
                    "Текст: Хочу уточнить насчёт покупки. Покупал вчера кое-что в вашем магазине.\n"
                    "{'emotions': ['neutral'], 'topics': ['другое']}\n"
                    "Текст: Купили смартфон у вас две недели назад. Стал медленно заряжаться.\n"
                    "{'emotions': ['neutral'], 'topics': ['мобильные телефоны']}\n"
                    "Текст: Ваш сайт хрень, виснет. Не могу ничего заказать.\n"
                    "{'emotions': ['negative', 'angry'], 'topics': ['сайт']}\n"
                    "Текст: Я вас засужу\n"
                    "{'emotions': ['angry'], 'topics': ['другое']}\n"
                    "Текст: Я просто уже плачу, сил нет\n"
                    "{'emotions': ['sad'], 'topics': ['другое']}\n"
                    "Текст: У меня вопрос по гарантии на SSD\n"
                    "{'emotions': ['neutral'], 'topics': ['компьютеры']}\n"
                    "Текст: Жёсткие диски ещё есть в продаже?\n"
                    "{'emotions': ['neutral'], 'topics': ['компьютеры']}\n"
                    "Текст: Есть в продаже чулки?\n"
                    "{'emotions': ['neutral'], 'topics': ['другое']}\n"
                    "Текст: Производите ли вы установку Windows и Microsoft офис?\n"
                    "{'emotions': ['neutral'], 'topics': ['компьютеры']}\n"
                    "Теперь классифицируй следующий звонок."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Текст: {text}\n\n"
                    f"Возможные эмоции: positive, negative, neutral, angry, happy, sad, surprised, fear, disgust\n"
                    f"Возможные темы: сайт, бытовая техника, мобильные телефоны, компьютеры, другое\n\n"
                )
            }
        ]

        try:
            grammar_obj = LlamaGrammar.from_string(self.JSON_GRAMMAR)
            response = self.model.create_chat_completion(
                messages=messages,
                temperature=0.0,
                top_p=0.05,
                top_k=10,
                max_tokens=80,
                grammar=grammar_obj
            )

            content = response["choices"][0]["message"]["content"].strip()

            result = json.loads(content)
            return result

        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"Ошибка парсинга ответа модели: {e}")
            print(f"Ответ модели:\n{content if 'content' in locals() else response}")
            # дефолтный возврат при ошибке
            return {
                "emotions": ["neutral"],
                "topics": ["другое"]
            }


# Глобальный экземпляр
llm = None


def get_llm(model_path: str = None, device: str = "auto") -> QwenAnalyzer:
    """
    Получить глобальный экземпляр анализатора Qwen
    """
    global llm
    if llm is None:
        llm = QwenAnalyzer(model_path, device)
    return llm
