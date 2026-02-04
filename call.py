import threading

import numpy as np
import os
import soundfile as sf
import time
from models.age_gender_predictor import get_age_gender_predictor
from models.emotion_vad_predictor import get_emotion_vad_predictor
import librosa

welcome_audio_path = "welcome.mp3"
# welcome_duration = 5.433  # в секундах
# delay_ms = int((welcome_duration + 0.5) * 1000)
final_audio_path = "final.mp3"
SAMPLERATE = 16000


def background_age_gender(audio_snapshot, stream_state):
    '''
    В фоне запускает обработку аудиофрагмента моделью, которая определяет возраст и пол человека по голосу.
    Итоговый результат будет сохранён в stream_state.
    '''
    model = get_age_gender_predictor()
    new_result = model.predict(audio_snapshot, SAMPLERATE)
    with stream_state["lock"]:
        prev_result = stream_state.get("ag_result")

        if prev_result is None:
            # первый запуск: просто присваиваем
            stream_state["ag_result"] = new_result
        else:
            avg_raw_score = (prev_result["age"]["raw_score"] + new_result["age"]["raw_score"]) / 2
            aggregated_age_years = round(100 * avg_raw_score)

            # агрегация вероятностей: берём максимумы для уверенности
            max_child_prob = max(
                prev_result["age_category"]["child_probability"],
                new_result["age_category"]["child_probability"]
            )
            max_female_prob = max(
                prev_result["gender"]["probabilities"]["female"],
                new_result["gender"]["probabilities"]["female"]
            )
            max_male_prob = max(
                prev_result["gender"]["probabilities"]["male"],
                new_result["gender"]["probabilities"]["male"]
            )

            # пересчёт на основе агрегированных вероятностей
            aggregated_result = {"age": {
                "years": aggregated_age_years,  # средний возраст
                "raw_score": avg_raw_score
            }, "gender": {
                "probabilities": {
                    "female": max_female_prob,
                    "male": max_male_prob
                },
                "predicted": "female" if max_female_prob > max_male_prob else "male"
            }, "age_category": {
                "is_child": max_child_prob > 0.5 or aggregated_age_years < 18,
                "child_probability": max_child_prob,
                "is_adult": max_child_prob <= 0.5 and aggregated_age_years >= 18,
                "adult_probability": 1 - max_child_prob
            }}

            stream_state["ag_result"] = aggregated_result

        # установка триггера
        final_result = stream_state["ag_result"]
        if final_result["age_category"]["is_child"] and not stream_state.get("age_trigger", False):
            stream_state["age_trigger"] = True

        # модель свободна
        stream_state["age_gender_processing"] = False


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
    audio_state["emo_vad_buffer"].append(audio_array)

    # ещё надо ж проверку на тишину

    should_run = False  # флаг для запуска фоновой обработки

    with stream_state["lock"]:
        if stream_state["retry_count"] < 2:
            audio_state["ag_buffer"].append(audio_array)
            total_samples = sum(len(chunk) for chunk in audio_state["ag_buffer"])
            total_seconds = total_samples / sample_rate

            if not stream_state["age_gender_processing"] and total_seconds >= 3.0:
                # первый раз
                if not stream_state["ag_result"]:
                    should_run = True
                # низкая уверенность модели
                if (stream_state["ag_result"] and
                        stream_state["ag_result"]["age_category"]["child_probability"] < 0.75 and
                        stream_state["ag_result"]["gender"]["probabilities"]["female"] < 0.75 and
                        stream_state["ag_result"]["gender"]["probabilities"]["male"] < 0.75):
                    should_run = True
                    stream_state["retry_count"] += 1

                if should_run:
                    partial_audio = np.concatenate(audio_state["ag_buffer"])
                    audio_snapshot = partial_audio.copy()
                    audio_state["ag_buffer"] = []
                    stream_state["age_gender_processing"] = True

    if should_run:
        threading.Thread(
            target=background_age_gender,
            args=(audio_snapshot, stream_state),
            daemon=True
        ).start()

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


def load_models():
    get_age_gender_predictor('.model_files/age_gender_model')
    get_emotion_vad_predictor('.model_files/emotion_vad_model')
