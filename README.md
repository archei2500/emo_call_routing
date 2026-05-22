# emo_call_routing
An application demonstrating smart call routing.

Emo call routing — приложение, демонстрирующее интеллектуальную маршрутизацию вызовов на основе паралингвистического и семантического анализа речи.

## Требования

- Python 3.9+
- pip
- (опционально) CUDA-совместимый GPU

## Установка

1. Базовые зависимости (установка в таком порядке):

```bash
pip install gigaam
pip install "onnxruntime>=1.18.1"
pip install -r requirements.txt
```

2. Дополнительные зависимости:

Для CPU:
```bash
pip install -r requirements-cpu.txt
```
Для CUDA:
```bash
pip install -r requirements-cuda.txt
```

## Запуск
После установки зависимостей запустите:
```bash
python app.py
```
Модели могут быть загружены через Интернет автоматически при запуске приложения.
При локальном запуске, однако, необходимо иметь файлы моделей в папке model_files в соответствующих директориях.
```text
emo_call_routing/
│
├── model_files/
│   ├── age_gender_model/              # audeering/wav2vec2-large-robust-24-ft-age-gender
│   ├── adult_child_detector/          # bookbot/wav2vec2-adult-child-cls
│   ├── emotion_vad_model/             # audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
│   ├── rubert_tiny_intent/            # rubert-tiny2 (fine-tuned) - предоставляется по запросу
│   ├── semantic_emotion_classifier/   # cointegrated/rubert-tiny2-cedr-emotion-detection
│   └── parakeet/                      # speakleash/parakeet-tdt-0.6b-v3 (ONNX)
...
```
При этом GigaAM в любом случае скачивается при первом запуске в системную директорию.
При необходимости полностью локального запуска можно поместить ранее скачанные файлы в соответствующую системную директорию.