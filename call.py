import numpy as np
import os
import soundfile as sf
import time

welcome_audio_path = "welcome.mp3"
# welcome_duration = 5.433  # в секундах
# delay_ms = int((welcome_duration + 0.5) * 1000)
final_audio_path = "final.mp3"
SAMPLERATE = 48000

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


def process_partial_chunk(audio_data, state):
    """
    Обрабатывает частичный аудио-чанк.
    audio_data может быть: None, кортеж (sampling_rate, audio_array), или просто audio_array
    """
    if audio_data is None:
        return state

    # Обрабатываем разные форматы входных данных
    audio_array = None
    sample_rate = SAMPLERATE  # значение по умолчанию

    if isinstance(audio_data, tuple):
        # Формат: (sampling_rate, audio_array)
        sample_rate, audio_array = audio_data
    elif isinstance(audio_data, np.ndarray):
        # Уже массив
        audio_array = audio_data
    else:
        # Неизвестный формат
        return state

    if audio_array is None or len(audio_array) == 0:
        return state

    # Нормализуем форму аудио
    if len(audio_array.shape) > 1:
        # Многоканальное аудио
        if audio_array.shape[0] == 2:  # (2, samples) - стерео
            audio_array = np.mean(audio_array, axis=0)  # конвертируем в моно
        elif audio_array.shape[1] == 2:  # (samples, 2) - стерео
            audio_array = np.mean(audio_array, axis=1)  # конвертируем в моно
        else:
            # Берем первый канал
            audio_array = audio_array[:, 0] if audio_array.shape[1] > 0 else audio_array[:, 0]

    # Добавляем в буферы
    state["full_buffer"].append(audio_array)
    state["partial_buffer"].append(audio_array)

    # Проверяем длину частичного буфера (5 секунд)
    if state["partial_buffer"]:
        total_samples = sum(len(chunk) for chunk in state["partial_buffer"])
        total_seconds = total_samples / sample_rate

        if total_seconds >= 5.0:
            try:
                # Объединяем чанки
                partial_audio = np.concatenate(state["partial_buffer"])

                # Сохраняем 5-секундный чанк
                output_dir = "chunks_5s"
                os.makedirs(output_dir, exist_ok=True)
                timestamp = int(time.time())
                wav_path = os.path.join(output_dir, f"chunk_{timestamp}.wav")
                sf.write(wav_path, partial_audio, sample_rate)

                # Пробуем конвертировать в MP3
                try:
                    import subprocess
                    mp3_path = wav_path.replace(".wav", ".mp3")
                    subprocess.run([
                        "ffmpeg", "-y", "-i", wav_path,
                        "-acodec", "libmp3lame", "-q:a", "2",
                        mp3_path
                    ], check=True, capture_output=True)
                    print(f"Сохранен 5-секундный чанк: {mp3_path}")
                except:
                    print(f"Сохранен 5-секундный чанк: {wav_path}")

            except Exception as e:
                print(f"Ошибка сохранения чанка: {e}")

            # Очищаем частичный буфер
            state["partial_buffer"] = []

    return state


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
