import threading
import numpy as np
# import os
# import soundfile as sf
from models.age_gender_predictor import get_age_gender_predictor
from models.emotion_vad_predictor import get_emotion_vad_predictor
from models.parakeet import get_asr_model
from models.emotion_classifier import get_emotion_classifier
from models.adult_child_detector import get_adult_child_detector
from models.intent_classifier import get_intent_classifier
from models.semantic_emotion_classifier import get_semantic_emotion_classifier
import librosa
import webrtcvad
import time


welcome_audio_path = "welcome.mp3"
# welcome_duration = 5.433  # в секундах
# delay_ms = int((welcome_duration + 0.5) * 1000)
final_audio_path = "final.mp3"
SAMPLERATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLERATE * FRAME_MS / 1000)  # 480 для 16000/30мс
VAD_AGGRESSIVENESS = 2  # 0-3
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
STATE_LOCK = threading.Lock()

# Центроиды аффективных состояний (MSP-Podcast, масштаб [0, 1])
# Порядок элементов: [Valence, Arousal, Dominance]
VAD_CENTROIDS = {
    "Angry":    [0.3328, 0.7365, 0.7433],
    "Sad":      [0.3970, 0.4584, 0.5239],
    "Disgust":  [0.3838, 0.5819, 0.6220],
    "Fear":     [0.4094, 0.5675, 0.6040],
    "Neutral":  [0.4870, 0.5029, 0.5597],
    "Surprise": [0.4919, 0.6052, 0.6278],
    "Happy":    [0.6200, 0.6343, 0.6539]
}
EMA_ALPHA = 0.35  # коэффициент забывания
TRIGGER_ANGER = 0.12
TRIGGER_DISTRESS = 0.20


def update_trigger(stream_state):
    ag_res = stream_state.get("ag_result")
    ac_res = stream_state.get("ac_result")

    is_child_now = False

    if ag_res and ag_res["age"]["years"] < 14:
        is_child_now = True
    if ac_res and ac_res["predicted"] == "child":
        is_child_now = True

    # если возраст уже подтверждён, триггер не нужен
    if stream_state.get("age_confirmed", False):
        stream_state["age_trigger"] = False
    else:
        # триггер = текущее мнение "ребёнок"
        stream_state["age_trigger"] = is_child_now


def background_age_gender(audio_snapshot, stream_state, current_call_id):
    '''
    В фоне запускает обработку аудиофрагмента моделью, которая определяет возраст и пол человека по голосу.
    Итоговый результат будет сохранён в stream_state.
    '''
    with STATE_LOCK:
        if stream_state.get("call_id") != current_call_id:
            stream_state["age_gender_processing"] = False
            print(f"Early abort: old call_id {current_call_id}, current is {stream_state['call_id']}")
            return
    model = get_age_gender_predictor()
    new_result = model.predict(audio_snapshot, SAMPLERATE)
    with STATE_LOCK:
        if stream_state.get("call_id") == current_call_id:
            prev_result = stream_state.get("ag_result")

            if prev_result is None:
                # первый запуск: просто присваиваем
                stream_state["ag_result"] = new_result
            else:
                avg_raw_score = (prev_result["age"]["raw_score"] + new_result["age"]["raw_score"]) / 2
                aggregated_age_years = round(100 * avg_raw_score)

                # агрегация вероятностей: берём максимумы для уверенности
                # max_child_prob = max(
                #     prev_result["age_category"]["child_probability"],
                #     new_result["age_category"]["child_probability"]
                # )
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
                    "is_child": new_result["age_category"]["is_child"],
                    "child_probability": new_result["age_category"]["child_probability"],
                    "is_adult": new_result["age_category"]["is_adult"],
                    "adult_probability": new_result["age_category"]["adult_probability"]
                }}

                stream_state["ag_result"] = aggregated_result

            # установка триггера
            update_trigger(stream_state)

            # final_result = stream_state["ag_result"]
            # if final_result["age_category"]["is_child"] and not stream_state.get("age_trigger", False):
            #     stream_state["age_trigger"] = True
            # ag_res = stream_state["ag_result"]
            # ac_res = stream_state.get("ac_result")
            # # проверяем, что триггер ещё не включен и возраст не подтверждён
            # if not stream_state.get("age_trigger", False):
            #     # Модель age/gender считает, что ребёнок
            #     if ag_res["age_category"]["is_child"]:
            #         stream_state["age_trigger"] = True
            #         print("Триггер включён age/gender: возраст < 18 лет")
            #     # ИЛИ модель adult/child (если есть) тоже считает, что ребёнок
            #     elif ac_res and ac_res["predicted"] == "child":
            #         stream_state["age_trigger"] = True
            #         print("Триггер включён age/gender: adult/child модель определила ребёнка")

        # модель свободна
        stream_state["age_gender_processing"] = False
        print("После отработки модели генедра/возраста:")
        print(stream_state)


def background_child_detection(audio_snapshot, stream_state, current_call_id):
    '''
    В фоне запускает обработку аудиофрагмента моделью, которая по голосу определяет, взрослый это или ребёнок.
    '''
    with STATE_LOCK:
        if stream_state.get("call_id") != current_call_id:
            stream_state["adult_child_processing"] = False
            print(f"Early abort: old call_id {current_call_id}, current is {stream_state['call_id']}")
            return

    model = get_adult_child_detector()
    new_result = model.predict(audio_snapshot, SAMPLERATE)

    with STATE_LOCK:
        if stream_state.get("call_id") == current_call_id:
            prev_result = stream_state.get("ac_result")

            if prev_result is None:
                stream_state["ac_result"] = new_result
            else:
                # агрегация: берём максимум вероятностей
                max_child_prob = max(
                    prev_result["probabilities"]["child"],
                    new_result["probabilities"]["child"]
                )
                max_adult_prob = max(
                    prev_result["probabilities"]["adult"],
                    new_result["probabilities"]["adult"]
                )

                stream_state["ac_result"] = {
                    "predicted": "child" if max_child_prob > max_adult_prob else "adult",
                    "confidence": max(max_child_prob, max_adult_prob),
                    "probabilities": {
                        "adult": max_adult_prob,
                        "child": max_child_prob
                    }
                }

            update_trigger(stream_state)

        stream_state["adult_child_processing"] = False
        print("После отработки модели child detection:", stream_state)


def call_gigaam_sync(audio_snapshot, stream_state, current_call_id):
    """
    Синхронный вызов GigaAM внутри фонового потока VAD.
    Записывает результат в stream_state["emo_result"].
    """
    try:
        classifier = get_emotion_classifier()
        result = classifier.predict(audio_snapshot)

        probs = result["probabilities"]
        predicted_class = result["predicted"]

        with STATE_LOCK:
            if stream_state.get("call_id") != current_call_id:
                print(f"[GigaAM] Aborted: call_id mismatch")
                return

            stream_state["emo_result"] = {
                "predicted_class": predicted_class,
                "probs": probs
            }
            print(f"[GigaAM] Success: {predicted_class} | probs={probs}")
    except Exception as e:
        print(f"[GigaAM] Inference error: {e}")
    finally:
        with STATE_LOCK:
            stream_state["gigaam_processing"] = False


def compute_emotion_risks(vad_vec, centroids=VAD_CENTROIDS, tau=0.15, w_sad=0.80, w_fear=0.20):
    vad_vec = np.array(vad_vec, dtype=float)

    # евклидовы расстояния до центроидов
    dists = {emo: np.linalg.norm(vad_vec - np.array(cent))
             for emo, cent in centroids.items()}

    # softmax от отрицательных расстояний
    logits = {emo: -d / tau for emo, d in dists.items()}
    max_logit = max(logits.values())
    exp_vals = {emo: np.exp(log - max_logit) for emo, log in logits.items()}
    sum_exp = sum(exp_vals.values())
    probs = {emo: e / sum_exp for emo, e in exp_vals.items()}

    # агрегация в риски
    anger_risk = probs["Angry"]
    distress_risk = w_sad * probs["Sad"] + w_fear * probs["Fear"]

    return anger_risk, distress_risk, probs


def background_emotion_vad(audio_snapshot, stream_state, current_call_id):
    '''
    В фоне запускает обработку аудиофрагмента моделью VAD.
    Итоговый результат сохраняется в stream_state.
    '''
    with STATE_LOCK:
        if stream_state.get("call_id") != current_call_id:
            stream_state["emotion_vad_processing"] = False
            print(f"Early abort: old call_id {current_call_id}, current is {stream_state['call_id']}")
            return

    model = get_emotion_vad_predictor()
    new_result = model.predict(audio_snapshot, SAMPLERATE)

    with STATE_LOCK:
        if stream_state.get("call_id") != current_call_id:
            stream_state["emotion_vad_processing"] = False
            return

        if stream_state.get("emo_vad_result") is None:
            stream_state["emo_vad_result"] = {}

        prev_result = stream_state["emo_vad_result"]
        new_vec = [
            new_result["emotions"]["valence"],
            new_result["emotions"]["arousal"],
            new_result["emotions"]["dominance"]
        ]

        prev_vec = prev_result.get("vector")
        if prev_vec is None:
            # первый вызов: просто сохраняем вектор
            stream_state["emo_vad_result"]["vector"] = new_vec
        else:
            # EMA-сглаживание
            smoothed_vec = [
                EMA_ALPHA * n + (1.0 - EMA_ALPHA) * p
                for n, p in zip(new_vec, prev_vec)
            ]
            stream_state["emo_vad_result"]["vector"] = smoothed_vec

        # вычисляем риски по текущему вектору
        anger_risk, distress_risk, _ = compute_emotion_risks(
            vad_vec=stream_state["emo_vad_result"]["vector"]
        )
        stream_state["emo_vad_result"]["risks"] = {
            "anger": anger_risk,
            "distress": distress_risk
        }

        prev_giga = stream_state.get("emo_result")
        prev_class = prev_giga.get("predicted_class") if prev_giga else None

        should_call = False
        if anger_risk > TRIGGER_ANGER and prev_class != "angry":
            should_call = True
        elif distress_risk > TRIGGER_DISTRESS and prev_class != "sad":
            should_call = True

        gigaam_busy = stream_state.get("gigaam_processing", False)

        stream_state["emotion_vad_processing"] = False

    if should_call and not gigaam_busy:
        with STATE_LOCK:
            if not stream_state.get("gigaam_processing", False):
                stream_state["gigaam_processing"] = True
                should_call_atomic = True
            else:
                should_call_atomic = False
        if should_call_atomic:
            print(f"[TRIGGER GigaAM] prev={prev_class} | anger={anger_risk:.2f}, distress={distress_risk:.2f}")
            call_gigaam_sync(audio_snapshot, stream_state, current_call_id)
    else:
        print(f"[SKIP GigaAM] confirmed/stable: prev={prev_class}")


# def background_emotion_vad(audio_snapshot, stream_state, current_call_id):
#     '''
#         В фоне запускает обработку аудиофрагмента моделью, которая определяет непрерывные измерения эмоций по голосу.
#         Итоговый результат будет сохранён в stream_state.
#     '''
#     with STATE_LOCK:
#         if stream_state.get("call_id") != current_call_id:
#             stream_state["emotion_vad_processing"] = False
#             print(f"Early abort: old call_id {current_call_id}, current is {stream_state['call_id']}")
#             return
#     model = get_emotion_vad_predictor()
#     new_result = model.predict(audio_snapshot, SAMPLERATE)
#     print(new_result)
#     with STATE_LOCK:
#         if stream_state.get("call_id") == current_call_id:
#             print("Зашла! Они должны быть равны: ", stream_state.get("call_id"), current_call_id)
#             prev_result = stream_state.get("emo_vad_result")
#             if prev_result is None:
#                 # первый запуск: просто присваиваем
#                 stream_state["emo_vad_result"] = new_result
#             else:
#                 aggregated_emotions = {"arousal": max(
#                     prev_result["emotions"]["arousal"],
#                     new_result["emotions"]["arousal"]
#                 )}
#
#                 # dominance и valence — min/max по полусфере
#                 for dim in ["dominance", "valence"]:
#                     p = prev_result["emotions"][dim]
#                     n = new_result["emotions"][dim]
#
#                     if (p >= 0.5 and n >= 0.5) or (p <= 0.5 and n <= 0.5):
#                         aggregated_emotions[dim] = max(p, n) if p >= 0.5 else min(p, n)
#                     else:
#                         aggregated_emotions[dim] = (p + n) / 2
#
#                 aggregated_result = {
#                     "raw_predictions": [
#                         aggregated_emotions["arousal"],
#                         aggregated_emotions["dominance"],
#                         aggregated_emotions["valence"]
#                     ],
#                     "emotions": aggregated_emotions
#                 }
#
#                 stream_state["emo_vad_result"] = aggregated_result
#
#         print("После отработки модели эмоций:")
#         print(stream_state)
#
#         # модель свободна
#         stream_state["emotion_vad_processing"] = False
#
#     print("Модель эмоций отработала.")


def is_chunk_speech(audio_array: np.ndarray, sample_rate: int = SAMPLERATE) -> bool:
    if len(audio_array) == 0:
        return False

    # конвертация в int16 (WebRTC требует PCM 16-bit) - надо проверить, не нужно ли такое остальным моделям!
    if audio_array.dtype != np.int16:
        if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
            audio_array = np.int16(audio_array * 32767)  # нормализация из [-1..1] в [-32768..32767]
        else:
            audio_array = audio_array.astype(np.int16)

    audio_array = audio_array.ravel()  # 1D массив

    n_samples = len(audio_array)
    n_frames = n_samples // FRAME_SAMPLES
    if n_frames == 0:
        return False

    voiced_count = 0
    for i in range(n_frames):
        start = i * FRAME_SAMPLES
        frame = audio_array[start: start + FRAME_SAMPLES]
        if len(frame) != FRAME_SAMPLES:
            continue  # пропускаем неполный фрейм (обычно в конце чанка)

        frame_bytes = frame.tobytes()
        is_speech = vad.is_speech(frame_bytes, sample_rate)
        if is_speech:
            voiced_count += 1

    ratio = voiced_count / n_frames if n_frames > 0 else 0
    return ratio >= 0.60  # порог (0.5-0.75)


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
        # Конвертируем в float32 для librosa
        audio_array = audio_array.astype(np.float32)

        # Если это int типы, нормализуем к [-1, 1]
        if np.issubdtype(audio_array.dtype, np.integer):
            # Автоматически определяем тип
            if audio_array.dtype == np.int16:
                audio_array = audio_array / 32768.0
            elif audio_array.dtype == np.int32:
                audio_array = audio_array / 2147483648.0
            elif audio_array.dtype == np.int8:
                audio_array = audio_array / 128.0
            elif audio_array.dtype == np.uint8:
                audio_array = (audio_array - 128) / 128.0
            # Если другой тип, просто делим на max(abs())
            else:
                max_val = np.max(np.abs(audio_array))
                if max_val > 0:
                    audio_array = audio_array / max_val

        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=SAMPLERATE)
        sample_rate = SAMPLERATE

    # проверка voice activity в чанке
    is_speech = is_chunk_speech(audio_array)

    # Добавляем в буферы
    if is_speech or audio_state["full_buffer"]:
        audio_state["full_buffer"].append(audio_array)
    if is_speech or audio_state["emo_buffer"]:
        audio_state["emo_buffer"].append(audio_array)

    should_run = False  # флаг для запуска фоновой обработки

    # оценка необходимости запуска модели предсказания пола и возраста по голосу и её запуск
    with STATE_LOCK:
        if stream_state["retry_count"] < 2:
            if is_speech or audio_state["ag_buffer"]:
                audio_state["ag_buffer"].append(audio_array)

            if audio_state["ag_buffer"]:
                total_samples_needed = int(2.5 * sample_rate)
                current_samples = 0
                last_chunks = []  # список чанков с конца

                for chunk in reversed(audio_state["ag_buffer"]):
                    current_samples += len(chunk)
                    last_chunks.append(chunk)
                    if current_samples >= total_samples_needed:
                        break

                total_seconds = current_samples / sample_rate

                if not stream_state["age_gender_processing"] and total_seconds >= 2.5:
                    # первый раз
                    if not stream_state["ag_result"]:
                        should_run = True
                    # низкая уверенность модели
                    # (включает проверку результата модели adult_child - так как уверенности female и male будут
                    # низкими, если голос детский)
                    if (stream_state["ag_result"] and
                            (stream_state["ac_result"] is None or
                             (stream_state["ac_result"] and stream_state["ac_result"]["child_probability"] < 0.75)) and
                            stream_state["ag_result"]["gender"]["probabilities"]["female"] < 0.75 and
                            stream_state["ag_result"]["gender"]["probabilities"]["male"] < 0.75):
                        should_run = True
                        stream_state["retry_count"] += 1
                    elif stream_state["ag_result"]:  # высокая уверенность
                        stream_state["retry_count"] = 2

                    if should_run:
                        current_call_id = stream_state["call_id"]
                        # берём только эти последние чанки (разворачиваем обратно в порядок)
                        partial_audio = np.concatenate(last_chunks[::-1])
                        # partial_audio = np.concatenate(audio_state["ag_buffer"])
                        audio_snapshot = partial_audio.copy()
                        audio_state["ag_buffer"] = []
                        stream_state["age_gender_processing"] = True

    if should_run:
        threading.Thread(
            target=background_age_gender,
            args=(audio_snapshot, stream_state, current_call_id),
            daemon=True
        ).start()

    should_run = False

    # оценка необходимости запуска модели детекции детей по голосу и её запуск
    with STATE_LOCK:
        if stream_state["child_retry_count"] < 2:
            if is_speech or audio_state["ac_buffer"]:
                audio_state["ac_buffer"].append(audio_array)

            if audio_state["ac_buffer"]:
                total_samples_needed = int(2.5 * sample_rate)
                current_samples = 0
                last_chunks = []  # список чанков с конца

                for chunk in reversed(audio_state["ac_buffer"]):
                    current_samples += len(chunk)
                    last_chunks.append(chunk)
                    if current_samples >= total_samples_needed:
                        break

                total_seconds = current_samples / sample_rate

                if not stream_state["adult_child_processing"] and total_seconds >= 2.5:
                    # первый раз
                    if not stream_state["ac_result"]:
                        should_run = True
                    # низкая уверенность модели
                    if (stream_state["ac_result"] and
                            stream_state["ac_result"]["age_category"]["child_probability"] < 0.75 and
                            stream_state["ac_result"]["age_category"]["adult_probability"] < 0.75):
                        should_run = True
                        stream_state["child_retry_count"] += 1
                    elif stream_state["ac_result"]:  # высокая уверенность
                        stream_state["child_retry_count"] = 2

                    if should_run:
                        current_call_id = stream_state["call_id"]
                        # берём только эти последние чанки (разворачиваем обратно в порядок)
                        partial_audio = np.concatenate(last_chunks[::-1])
                        # partial_audio = np.concatenate(audio_state["ag_buffer"])
                        audio_snapshot_3 = partial_audio.copy()
                        audio_state["ac_buffer"] = []
                        stream_state["adult_child_processing"] = True

    if should_run:
        threading.Thread(
            target=background_child_detection,
            args=(audio_snapshot_3, stream_state, current_call_id),
            daemon=True
        ).start()

    should_run = False

    # оценка необходимости (накоплено достаточно чанков) запуска модели предсказания эмоции по голосу и её запуск
    with STATE_LOCK:
        if audio_state["emo_vad_buffer"]:
            total_samples_needed = int(3.5 * sample_rate)
            current_samples = 0
            last_chunks = []  # список чанков с конца

            for chunk in reversed(audio_state["emo_vad_buffer"]):
                current_samples += len(chunk)
                last_chunks.append(chunk)
                if current_samples >= total_samples_needed:
                    break

            # total_samples = sum(len(chunk) for chunk in audio_state["emo_vad_buffer"])
            total_seconds = current_samples / sample_rate
            if not stream_state["emotion_vad_processing"] and total_seconds >= 3:
                should_run = True
                # partial_audio = np.concatenate(audio_state["emo_vad_buffer"])
                partial_audio = np.concatenate(last_chunks[::-1])
                audio_snapshot_2 = partial_audio.copy()
                audio_state["emo_vad_buffer"] = []
                stream_state["emotion_vad_processing"] = True
                current_call_id = stream_state["call_id"]

    if should_run:
        threading.Thread(
            target=background_emotion_vad,
            args=(audio_snapshot_2, stream_state, current_call_id),
            daemon=True
        ).start()

    return audio_state, stream_state


def process_full_audio(audio_state, stream_state):
    max_wait_seconds = 5.0  # максимум ждём 5 секунд
    sleep_interval = 0.2  # проверяем каждые 200 мс
    waited = 0.0
    while waited < max_wait_seconds:
        with STATE_LOCK:
            age_done = not stream_state.get("age_gender_processing", False)
            emo_done = not stream_state.get("emotion_vad_processing", False)
            age_has_result = stream_state.get("ag_result") is not None
            emo_has_result = (stream_state.get("emo_vad_result") is not None)
        if (age_done and emo_done) and (age_has_result or emo_has_result):
            break
        time.sleep(sleep_interval)
        waited += sleep_interval

    routing_mode = 0  # полноценная маршрутизация (1 - без эмоций, 2 - без демографии, 3 - только по тексту)
    with STATE_LOCK:
        final_emo = stream_state.get("emo_vad_result")
        final_age = stream_state.get("ag_result")
    if waited >= max_wait_seconds:
        print(f"[TIMEOUT] Ожидание прервано после {waited:.1f} сек. Используем fallback-маршрутизацию.")
    if final_emo is None or not final_emo.get("risks"):
        print("[FALLBACK] VAD-риски отсутствуют → маршрутизация по умолчанию / только по тексту")
        routing_mode = 1
    if not final_age:
        print("[FALLBACK] Age/Gender отсутствуют → игнорируем демографический фильтр")
        if routing_mode == 0:
            routing_mode = 2
        else:
            routing_mode = 3

    if stream_state["age_trigger"] and not stream_state["age_confirmed"]:
        routing_result = "Вы не подтвердили, что вам больше 14 лет, поэтому мы не можем обработать ваш запрос."
        return routing_result, audio_state, stream_state

    if audio_state["full_buffer"]:
        # распознавание речи
        full_audio = np.concatenate(audio_state["full_buffer"])
        asr_model = get_asr_model()
        transcription = asr_model.transcribe(full_audio, SAMPLERATE)
        # определение темы обращения
        intent_classifier = get_intent_classifier()
        intent_result = intent_classifier.predict(transcription)
        # определение эмоции по тексту
        semantic_emotion_classifier = get_semantic_emotion_classifier()
        emotion_result = semantic_emotion_classifier.predict(transcription)
        # TODO: вызов функции маршрутизации
    else:
        print("Ошибка! Буфер пуст. Невозможно выполнить маршрутизацию без интента.")
        return "Извините, не удалось распознать ваш запрос.", audio_state, stream_state


    # return routing_result, audio_state, stream_state, transcription


# def process_full_audio(audio_state, stream_state):
#     print("Что у нас тут такое?")
#     print(stream_state)
#
#     max_wait_seconds = 8.0  # максимум ждём 8 секунд
#     sleep_interval = 0.2  # проверяем каждые 200 мс
#
#     waited = 0.0
#     while waited < max_wait_seconds:
#         with STATE_LOCK:
#             age_done = not stream_state["age_gender_processing"] or stream_state["ag_result"] is not None
#             emo_done = not stream_state["emotion_vad_processing"] or stream_state["emo_vad_result"] is not None
#         if age_done and emo_done:
#             break
#         time.sleep(sleep_interval)
#         waited += sleep_interval
#
#     if waited >= max_wait_seconds:
#         print(f"Warning: время ожидания результатов истекло после {waited:.1f} сек")
#
#     error = False
#
#     with STATE_LOCK:
#         if stream_state["age_trigger"] and not stream_state["age_confirmed"]:
#             routing_result = "Вы не подтвердили, что вам больше 18 лет, поэтому мы не можем обработать ваш запрос."
#             audio_state["full_buffer"] = []
#             audio_state["ag_buffer"] = []
#             audio_state["emo_vad_buffer"] = []
#             return routing_result, audio_state, stream_state
#
#         if stream_state["ag_result"]:
#             routing_result = "Возраст: " + str(stream_state["ag_result"]["age"]["years"])
#             routing_result += "\nПол: "
#             if stream_state["ag_result"]["gender"]["predicted"] == "male":
#                 routing_result += "мужской"
#             else:
#                 routing_result += "женский"
#         else:
#             routing_result = "Ошибка: демографические признаки не были определены."
#             error = True
#
#         if stream_state["emo_vad_result"]:
#             routing_result += "\nЭмоциональное состояние:"
#             routing_result += "\nValence: " + str(stream_state["emo_vad_result"]["emotions"]["valence"])
#             routing_result += "\nArousal: " + str(stream_state["emo_vad_result"]["emotions"]["arousal"])
#             routing_result += "\nDominance: " + str(stream_state["emo_vad_result"]["emotions"]["dominance"])
#
#             if stream_state["emo_vad_result"]["emotions"]["arousal"] > 0.65:
#                 if stream_state["emo_vad_result"]["emotions"]["valence"] < 0.5:
#                     voice_emotion = "Клиент испытывает резко негативную эмоцию."
#                     if stream_state["emo_vad_result"]["emotions"]["dominance"] > 0.5:
#                         voice_emotion += " Вероятно, гнев."
#                     else:
#                         voice_emotion += " Вероятно, страх."
#                 else:
#                     voice_emotion = "Клиент испытывает яркую позитивную эмоцию."
#             else:
#                 voice_emotion = "Клиент достаточно спокоен."
#         else:
#             routing_result += "\nОшибка: эмоции не были определены."
#             error = True
#
#         routing_result += "\n\nРекомендации:"
#
#         if not error:
#             if stream_state["ag_result"]["age"]["years"] >= 60:
#                 routing_result += "\nТерпеливый специалист, который готов спокойно и долго объяснять."
#             if stream_state["ag_result"]["gender"]["predicted"] == "male":
#                 routing_result += "\nСпециалист-мужчина."
#             elif stream_state["ag_result"]["gender"]["predicted"] == "female":
#                 routing_result += "\nСпециалист-женщина."
#             if (stream_state["emo_vad_result"]["emotions"]["arousal"] > 0.65 and
#                     stream_state["emo_vad_result"]["emotions"]["valence"] < 0.5):
#                 routing_result += "\nСтрессоустойчивый специалист. Такой, у которого это не конец смены."
#
#     full_audio = np.concatenate(audio_state["full_buffer"])
#     asr_model = get_asr_model()
#     transcription = asr_model.transcribe(full_audio, SAMPLERATE)
#     # llm = get_llm()
#     # llm_response = llm.get_response(transcription)
#     # routing_result += "\n\n" + str(llm_response)
#
#     # очистка
#     # audio_state["full_buffer"] = []
#     # audio_state["ag_buffer"] = []
#     # audio_state["emo_vad_buffer"] = []
#
#     return routing_result, audio_state, stream_state, transcription


# def process_full_audio(state):
#     """
#     Обрабатывает полное аудио после завершения записи
#     """
#     if not state.get("full_buffer"):
#         return state, "Нет аудиоданных для обработки"
#
#     try:
#         # Проверяем, есть ли данные в буфере
#         valid_chunks = []
#         sample_rate = SAMPLERATE
#
#         for chunk in state["full_buffer"]:
#             if chunk is not None and len(chunk) > 0:
#                 # Нормализуем чанк
#                 if len(chunk.shape) > 1:
#                     if chunk.shape[0] == 2:  # (2, samples)
#                         chunk = np.mean(chunk, axis=0)
#                         chunk = chunk.astype(np.float32)
#                     elif chunk.shape[1] == 2:  # (samples, 2)
#                         chunk = np.mean(chunk, axis=1)
#                         chunk = chunk.astype(np.float32)
#                     else:
#                         chunk = chunk[:, 0] if chunk.shape[1] > 0 else chunk.flatten()
#                         chunk = chunk.astype(np.float32)
#                 else:
#                     chunk = chunk.astype(np.float32)
#
#                 valid_chunks.append(chunk)
#
#         if not valid_chunks:
#             return state, "Нет валидных аудиоданных"
#
#         # Объединяем все чанки
#         full_audio = np.concatenate(valid_chunks)
#
#         # Проверяем длину
#         if len(full_audio) == 0:
#             return state, "Аудио пустое"
#
#         # Рассчитываем длительность
#         duration_seconds = len(full_audio) / SAMPLERATE
#
#         # Сохраняем полное аудио
#         output_dir = "recordings"
#         os.makedirs(output_dir, exist_ok=True)
#         timestamp = int(time.time())
#         wav_path = os.path.join(output_dir, f"full_recording_{timestamp}.wav")
#
#         # Нормализуем громкость если нужно
#         max_val = np.max(np.abs(full_audio))
#         if max_val > 1.0:
#             full_audio = full_audio / max_val
#
#         # Сохраняем как WAV
#         sf.write(wav_path, full_audio, SAMPLERATE)
#
#         # Пробуем конвертировать в MP3
#         try:
#             import subprocess
#             mp3_path = wav_path.replace(".wav", ".mp3")
#             subprocess.run([
#                 "ffmpeg", "-y", "-i", wav_path,
#                 "-acodec", "libmp3lame", "-q:a", "2",
#                 mp3_path
#             ], check=True, capture_output=True)
#             saved_path = mp3_path
#         except Exception as e:
#             print(f"Не удалось конвертировать в MP3: {e}")
#             saved_path = wav_path
#
#         result_text = f"✅ Аудио сохранено: {saved_path}\n🎵 Длительность: {duration_seconds:.1f} секунд"
#
#         # Очищаем буферы
#         state["full_buffer"] = []
#         state["partial_buffer"] = []
#
#         return result_text, state
#
#     except Exception as e:
#         print(f"Ошибка в process_full_audio: {e}")
#         import traceback
#         traceback.print_exc()
#
#         # Отладочная информация
#         print("\n=== Отладка состояния ===")
#         print(f"Количество чанков: {len(state.get('full_buffer', []))}")
#         for i, chunk in enumerate(state.get('full_buffer', [])):
#             if chunk is not None:
#                 print(f"Чанк {i}: тип={type(chunk)}, форма={chunk.shape if hasattr(chunk, 'shape') else 'N/A'}")
#             else:
#                 print(f"Чанк {i}: None")
#
#         return state, f"⚠️ Ошибка обработки: {str(e)}"


def load_models():
    age_gender = get_age_gender_predictor('model_files/age_gender_model')
    adult_child = get_adult_child_detector('model_files/adult_child_detector')
    emotion_vad = get_emotion_vad_predictor('model_files/emotion_vad_model')
    emotion_classifier = get_emotion_classifier("emo", "model_files/GigaAm_emo")
    asr_model = get_asr_model('model_files/parakeet/parakeet-tdt-0.6b-v3.nemo')
    intent_classifier = get_intent_classifier('model_files/rubert_tiny_intent')
    semantic_emotion = get_semantic_emotion_classifier('model_files/semantic_emotion_classifier')
    # get_llm('model_files/qwen/Qwen2.5-1.5B-Instruct-Q5_K_M.gguf', "cpu")

    print("Прогрев моделей...")
    print("  Прогрев ASR...")
    asr_model.transcribe_from_file(welcome_audio_path)
    print("  Прогрев Age/Gender...")
    age_gender.predict_from_file(welcome_audio_path)
    print("  Прогрев Adult/Child...")
    adult_child.predict_from_file(welcome_audio_path)
    print("  Прогрев Emotion VAD...")
    emotion_vad.predict_from_file(welcome_audio_path)
    print("  Прогрев Emotion Classifier...")
    emotion_classifier.predict_from_file(welcome_audio_path)
    default_text = "Здравствуйте, у меня проблема с картой"
    print("  Прогрев Intent Classifier...")
    intent_classifier.predict(default_text)
    print("  Прогрев Semantic Emotion Classifier...")
    semantic_emotion.predict(default_text)
    print("Все модели загружены и прогреты!")
