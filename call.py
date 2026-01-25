import numpy as np
import os
import soundfile as sf
import time

welcome_audio_path = "welcome.mp3"
welcome_duration = 5.433  # в секундах
delay_ms = int((welcome_duration + 0.5) * 1000)
final_audio_path = "final.mp3"


def process_partial_chunk(audio_chunk, state):
    if audio_chunk is None:
        # Конец стриминга — но здесь не финализируем, оставляем для .stop_recording()
        return state

    # Добавляем в полный буфер (для финальной обработки)
    state["full_buffer"].append(audio_chunk)

    # Добавляем в частичный буфер
    state["partial_buffer"].append(audio_chunk)

    # Проверяем длину частичного буфера
    partial_len = sum(len(c) for c in state["partial_buffer"])
    partial_sec = partial_len / state["samplerate"]

    if partial_sec >= 3.0:
        # Накопилось >=3 сек → обрабатываем (например, частичная транскрипция / анализ)
        partial_audio = np.concatenate(state["partial_buffer"])

        # Здесь твоя логика: например, Whisper на partial_audio
        # partial_text = whisper_model.transcribe(partial_audio)['text']
        # Или просто симуляция:
        partial_result = f"Обработан чанк {partial_sec:.1f} сек: [симуляция частичной транскрипции]"

        # Очищаем частичный буфер
        state["partial_buffer"] = []

        # Можно обновить видимый output, если добавишь (например, live_transcript.value += partial_result)
        # print(partial_result)  # для лога

    return state

    return state + "\n" + partial_text, partial_text  # или транскрипцию


# 2. Функция для финальной обработки (вызывается при stop)
# def process_full_audio(state):
#     if not state["full_buffer"]:
#         return "Нет аудио", state
#
#     full_audio = np.concatenate(state["full_buffer"])
#     full_sec = len(full_audio) / state["samplerate"]
#
#     # Здесь полный Whisper
#     # full_text = whisper_model.transcribe(full_audio)['text']
#     # Или сохрани в файл:
#     # import soundfile as sf
#     # sf.write("full_call.wav", full_audio, state["samplerate"])
#
#     # Симуляция:
#     full_result = f"Полное аудио {full_sec:.1f} сек: [симуляция полной транскрипции]"
#
#     # Сбрасываем буферы
#     state["full_buffer"] = []
#     state["partial_buffer"] = []
#
#     # Возвращаем результат (в textbox или status)
#     return full_result, state

def process_full_audio(state):
    if not state.get("full_buffer"):
        return "Нет аудио", state

    full_audio = np.concatenate(state["full_buffer"])

    SAMPLERATE = 48000  # ← самое частое значение в Gradio + browser microphone
    # или 44100, или 16000 — но 48000 встречается чаще всего

    full_sec = len(full_audio) / SAMPLERATE

    # ... дальше сохранение файла как раньше

    output_dir = "recordings"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    mp3_path = os.path.join(output_dir, f"client_call_{timestamp}.mp3")

    # Сохраняем в wav → конвертируем в mp3 (если ffmpeg есть)
    wav_path = mp3_path.replace(".mp3", ".wav")
    sf.write(wav_path, full_audio, SAMPLERATE, subtype='PCM_16')

    # ffmpeg → mp3 (опционально)
    try:
        import subprocess
        subprocess.run([
            "ffmpeg", "-y", "-i", wav_path,
            "-acodec", "libmp3lame", "-q:a", "2",
            mp3_path
        ], check=True, capture_output=True)
        # os.remove(wav_path)  # раскомментируй, если не хочешь оставлять wav
        saved = mp3_path
    except:
        saved = wav_path

    result_text = f"Аудио сохранено: {saved}\nДлительность ≈ {full_sec:.1f} сек"

    # очистка
    state["full_buffer"] = []
    state["partial_buffer"] = []

    return result_text, state