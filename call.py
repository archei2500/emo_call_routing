import numpy as np
import os
import soundfile as sf
import time
from models.age_gender_predictor import get_age_gender_predictor
from models.emotion_vad_predictor import get_emotion_predictor
import librosa

welcome_audio_path = "welcome.mp3"
# welcome_duration = 5.433  # в секундах
# delay_ms = int((welcome_duration + 0.5) * 1000)
final_audio_path = "final.mp3"
SAMPLERATE = 16000


def analyze_audio_chunk(audio_array, sample_rate, age_gender_predictor, emotion_predictor):
    """
    Анализирует аудио-чанк с использованием обеих моделей.
    """
    results = {
        "timestamp": time.time(),
        "duration": len(audio_array) / sample_rate,
        "sample_rate": sample_rate
    }

    try:
        # Анализ возраста и пола
        age_gender_result = age_gender_predictor.predict(audio_array, sample_rate)
        results["age_gender"] = age_gender_result

        # Анализ эмоций
        emotion_result = emotion_predictor.predict(audio_array, sample_rate)
        results["emotion"] = emotion_result

    except Exception as e:
        print(f"Ошибка анализа аудио: {e}")
        results["error"] = str(e)

    return results


def process_partial_chunk(audio_data, audio_state, stream_state):
    """
    Обрабатывает частичный аудио-чанк.
    """
    if audio_data is None:
        return audio_state, stream_state

    # Инициализируем модели один раз при первом вызове
    # if "models_initialized" not in state:
    #     print("Инициализация моделей...")
    #     state["age_gender_predictor"] = get_age_gender_predictor(
    #         model_path="./model_paths/age_gender_model",
    #         device="cpu"
    #     )
    #     state["emotion_predictor"] = get_emotion_predictor(
    #         model_path="./model_paths/emotion_vad_predictor",
    #         device="cpu"
    #     )
    #     state["models_initialized"] = True
    #     print("Модели инициализированы.")

    # Обрабатываем разные форматы входных данных
    audio_array = None
    sample_rate = SAMPLERATE  # значение по умолчанию

    if isinstance(audio_data, tuple):
        # формат: (sampling_rate, audio_array)
        sample_rate, audio_array = audio_data
    elif isinstance(audio_data, np.ndarray):
        # массив
        audio_array = audio_data
    else:
        # неизвестный формат
        return audio_state, stream_state

    if audio_array is None or len(audio_array) == 0:
        return audio_state, stream_state

    # нормализуем форму аудио
    if len(audio_array.shape) > 1:
        # многоканальное аудио
        if audio_array.shape[0] == 2:  # (2, samples) - стерео
            audio_array = np.mean(audio_array, axis=0)  # конвертируем в моно
        elif audio_array.shape[1] == 2:  # (samples, 2) - стерео
            audio_array = np.mean(audio_array, axis=1)  # конвертируем в моно
        else:
            # берем первый канал
            audio_array = audio_array[:, 0] if audio_array.shape[1] > 0 else audio_array[:, 0]

    if sample_rate != SAMPLERATE:
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=SAMPLERATE)
        sample_rate = SAMPLERATE

    # Добавляем в буферы
    audio_state["full_buffer"].append(audio_array)
    audio_state["ag_buffer"].append(audio_array)
    audio_state["emo_vad_buffer"].append(audio_array)

    # Проверяем длину частичного буфера (3 секунды)
    # if state["partial_buffer"]:
    #     total_samples = sum(len(chunk) for chunk in state["partial_buffer"])
    #     total_seconds = total_samples / sample_rate
    #
    #     if total_seconds >= 3.0:
    #         partial_audio = np.concatenate(state["partial_buffer"])
    #
    #         # Очищаем частичный буфер
    #         state["partial_buffer"] = []

    if not stream_state["age_gender_processing"]:
        if stream_state["ag_result"]: # надо ли запускать во второй раз, решаем
            pass
        else:
            pass

    return audio_state, stream_state


def process_full_audio(state):
    """
    Обрабатывает полное аудио после завершения записи
    """
    if not state.get("full_buffer"):
        return state, "Нет аудиоданных для обработки"

    try:
        # Проверяем, есть ли данные в буфере
        valid_chunks = []
        sample_rate = SAMPLERATE

        for chunk in state["full_buffer"]:
            if chunk is not None and len(chunk) > 0:
                # Нормализуем чанк
                if len(chunk.shape) > 1:
                    if chunk.shape[0] == 2:  # (2, samples)
                        chunk = np.mean(chunk, axis=0)
                        chunk = chunk.astype(np.float32)
                    elif chunk.shape[1] == 2:  # (samples, 2)
                        chunk = np.mean(chunk, axis=1)
                        chunk = chunk.astype(np.float32)
                    else:
                        chunk = chunk[:, 0] if chunk.shape[1] > 0 else chunk.flatten()
                        chunk = chunk.astype(np.float32)
                else:
                    chunk = chunk.astype(np.float32)

                valid_chunks.append(chunk)

        if not valid_chunks:
            return state, "Нет валидных аудиоданных"

        # Объединяем все чанки
        full_audio = np.concatenate(valid_chunks)

        # Проверяем длину
        if len(full_audio) == 0:
            return state, "Аудио пустое"

        # Рассчитываем длительность
        duration_seconds = len(full_audio) / SAMPLERATE

        # Сохраняем полное аудио
        output_dir = "recordings"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())
        wav_path = os.path.join(output_dir, f"full_recording_{timestamp}.wav")

        # Нормализуем громкость если нужно
        max_val = np.max(np.abs(full_audio))
        if max_val > 1.0:
            full_audio = full_audio / max_val

        # Сохраняем как WAV
        sf.write(wav_path, full_audio, SAMPLERATE)

        # Пробуем конвертировать в MP3
        try:
            import subprocess
            mp3_path = wav_path.replace(".wav", ".mp3")
            subprocess.run([
                "ffmpeg", "-y", "-i", wav_path,
                "-acodec", "libmp3lame", "-q:a", "2",
                mp3_path
            ], check=True, capture_output=True)
            saved_path = mp3_path
        except Exception as e:
            print(f"Не удалось конвертировать в MP3: {e}")
            saved_path = wav_path

        result_text = f"✅ Аудио сохранено: {saved_path}\n🎵 Длительность: {duration_seconds:.1f} секунд"

        # Очищаем буферы
        state["full_buffer"] = []
        state["partial_buffer"] = []

        return result_text, state

    except Exception as e:
        print(f"Ошибка в process_full_audio: {e}")
        import traceback
        traceback.print_exc()

        # Отладочная информация
        print("\n=== Отладка состояния ===")
        print(f"Количество чанков: {len(state.get('full_buffer', []))}")
        for i, chunk in enumerate(state.get('full_buffer', [])):
            if chunk is not None:
                print(f"Чанк {i}: тип={type(chunk)}, форма={chunk.shape if hasattr(chunk, 'shape') else 'N/A'}")
            else:
                print(f"Чанк {i}: None")

        return state, f"⚠️ Ошибка обработки: {str(e)}"


# def process_partial_chunk(audio_chunk, state):
#     if audio_chunk is None:
#         # Конец стриминга — но здесь не финализируем, оставляем для .stop_recording()
#         return state
#
#     # Добавляем в полный буфер (для финальной обработки)
#     state["full_buffer"].append(audio_chunk)
#
#     # Добавляем в частичный буфер
#     state["partial_buffer"].append(audio_chunk)
#
#     # Проверяем длину частичного буфера
#     partial_len = sum(len(c) for c in state["partial_buffer"])
#     partial_sec = partial_len / 48000
#
#     if partial_sec >= 5.0:
#         # Накопилось >=3 сек → обрабатываем (например, частичная транскрипция / анализ)
#         partial_audio = np.concatenate(state["partial_buffer"])
#
#         # Здесь твоя логика: например, Whisper на partial_audio
#         # partial_text = whisper_model.transcribe(partial_audio)['text']
#         # Или просто симуляция:
#         partial_result = f"Обработан чанк {partial_sec:.1f} сек: [симуляция частичной транскрипции]"
#
#         # Очищаем частичный буфер
#         state["partial_buffer"] = []
#
#         output_dir = "chunks_5s"
#         os.makedirs(output_dir, exist_ok=True)
#         timestamp = int(time.time())
#         mp3_path = os.path.join(output_dir, f"client_call_{timestamp}.mp3")
#         # Сохраняем в wav → конвертируем в mp3 (если ffmpeg есть)
#         wav_path = mp3_path.replace(".mp3", ".wav")
#         try:
#             import subprocess
#             subprocess.run([
#                 "ffmpeg", "-y", "-i", wav_path,
#                 "-acodec", "libmp3lame", "-q:a", "2",
#                 mp3_path
#             ], check=True, capture_output=True)
#             # os.remove(wav_path)  # раскомментируй, если не хочешь оставлять wav
#             saved = mp3_path
#         except:
#             saved = wav_path
#
#         # Можно обновить видимый output, если добавишь (например, live_transcript.value += partial_result)
#         # print(partial_result)  # для лога
#
#     return state
#
#     #return state + "\n" + partial_text, partial_text  # или транскрипцию
#
#
# # 2. Функция для финальной обработки (вызывается при stop)
# # def process_full_audio(state):
# #     if not state["full_buffer"]:
# #         return "Нет аудио", state
# #
# #     full_audio = np.concatenate(state["full_buffer"])
# #     full_sec = len(full_audio) / state["samplerate"]
# #
# #     # Здесь полный Whisper
# #     # full_text = whisper_model.transcribe(full_audio)['text']
# #     # Или сохрани в файл:
# #     # import soundfile as sf
# #     # sf.write("full_call.wav", full_audio, state["samplerate"])
# #
# #     # Симуляция:
# #     full_result = f"Полное аудио {full_sec:.1f} сек: [симуляция полной транскрипции]"
# #
# #     # Сбрасываем буферы
# #     state["full_buffer"] = []
# #     state["partial_buffer"] = []
# #
# #     # Возвращаем результат (в textbox или status)
# #     return full_result, state
#
# def process_full_audio(state):
#     if not state.get("full_buffer"):
#         return "Нет аудио", state
#
#     full_audio = np.concatenate(state["full_buffer"])
#
#     SAMPLERATE = 48000  # ← самое частое значение в Gradio + browser microphone
#     # или 44100, или 16000 — но 48000 встречается чаще всего
#
#     full_sec = len(full_audio) / SAMPLERATE
#
#     # ... дальше сохранение файла как раньше
#
#     output_dir = "recordings"
#     os.makedirs(output_dir, exist_ok=True)
#     timestamp = int(time.time())
#     mp3_path = os.path.join(output_dir, f"client_call_{timestamp}.mp3")
#
#     # Сохраняем в wav → конвертируем в mp3 (если ffmpeg есть)
#     wav_path = mp3_path.replace(".mp3", ".wav")
#     sf.write(wav_path, full_audio, SAMPLERATE, subtype='PCM_16')
#
#     # ffmpeg → mp3 (опционально)
#     try:
#         import subprocess
#         subprocess.run([
#             "ffmpeg", "-y", "-i", wav_path,
#             "-acodec", "libmp3lame", "-q:a", "2",
#             mp3_path
#         ], check=True, capture_output=True)
#         # os.remove(wav_path)  # раскомментируй, если не хочешь оставлять wav
#         saved = mp3_path
#     except:
#         saved = wav_path
#
#     result_text = f"Аудио сохранено: {saved}\nДлительность ≈ {full_sec:.1f} сек"
#
#     # очистка
#     state["full_buffer"] = []
#     state["partial_buffer"] = []
#
#     return result_text
