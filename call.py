welcome_audio_path = "welcome.mp3"
welcome_duration = 5  # в секундах
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
        print(partial_result)  # для лога

    return state

    return state + "\n" + partial_text, partial_text  # или транскрипцию


# 2. Функция для финальной обработки (вызывается при stop)
def process_full_audio(state):
    if not state["full_buffer"]:
        return "Нет аудио", state

    full_audio = np.concatenate(state["full_buffer"])
    full_sec = len(full_audio) / state["samplerate"]

    # Здесь полный Whisper
    # full_text = whisper_model.transcribe(full_audio)['text']
    # Или сохрани в файл:
    # import soundfile as sf
    # sf.write("full_call.wav", full_audio, state["samplerate"])

    # Симуляция:
    full_result = f"Полное аудио {full_sec:.1f} сек: [симуляция полной транскрипции]"

    # Сбрасываем буферы
    state["full_buffer"] = []
    state["partial_buffer"] = []

    # Возвращаем результат (в textbox или status)
    return full_result, state