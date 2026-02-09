import threading

import gradio as gr
import os
import pandas as pd
import call
# import iso639
# import torch
# import importlib
os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-user'
os.environ['ALSA_CONFIG_PATH'] = '/dev/null'


initial_df = pd.DataFrame({
    "Имя": ["Анна", "Борис", "Кирилл"],
    "Возраст": [28, 34, 25],
    "Город": ["Москва", "Санкт-Петербург", "Казань"]
})


def update_visibility():
    return gr.update(visible=True), gr.update(visible=True)


def update_invisibility():
    return gr.update(visible=False), gr.update(visible=False)


def update_ui_from_state(state):
    if state.get("age_trigger") and not state.get("age_confirmed"):
        return gr.update(visible=True)
    return gr.update(visible=False)


def confirm_age(state):
    state["age_confirmed"] = True
    return state


call.load_models()


with gr.Blocks() as demo:
    with gr.Tab("Симуляция звонка"):
        gr.Markdown("### <center>Нажмите на кнопку ниже, чтобы начать звонок", visible=False)
        call_btn = gr.Button("Позвонить", visible=True)
        recorder_text = gr.Markdown("Нажмите на кнопку записи, чтобы описать проблему устно.", visible=False)
        recorder = gr.Audio(sources=["microphone"], streaming=True, type="numpy", visible=False, interactive=True, elem_id="recorder")
        recorder_final_text = gr.Markdown("Затем завершите запись, когда будете готовы.", visible=False)
        confirm_age_btn = gr.Button("Мне есть 18", visible=False)  # для подтверждения возраста (если распознает ребёнка)
        audio_state = gr.State(value={"full_buffer": [], "ag_buffer": [], "emo_vad_buffer": []})  # cостояние для накопления чанков
        # состояние для фоновой обработки моделями
        stream_state = gr.State(
            value={
                "age_gender_processing": False,
                "emotion_vad_processing": False,
                "ag_result": None,
                "retry_count": 0,
                "age_trigger": False,
                "age_confirmed": False,
                "emo_vad_result": None
            }
        )
        # "state_lock": threading.Lock()
        player_start_text = gr.Markdown("Нажмите на кнопку воспроизведения, чтобы прослушать приветствие.", visible=False)
        player_end_text = gr.Markdown("Нажмите на кнопку воспроизведения, чтобы прослушать финальное сообщение.", visible=False)
        player1 = gr.Audio(format="mp3", visible=False, autoplay=False, interactive=True, elem_id="player")
        player2 = gr.Audio(format="mp3", visible=False, autoplay=False, interactive=True, elem_id="player")
        routing_result = gr.Textbox(label="Подобранный специалист", visible=False)
    with gr.Tab("Панель специалиста"):
        gr.Markdown("### <center>Описание проблемы клиента:")
        problem_text = gr.Textbox(label="Цель звонка", visible=True)
    with gr.Tab("Панель администратора"):
        admin_table = gr.Dataframe(
            value=initial_df,
            interactive=True,       # разрешает редактирование
            label="Список сотрудников")
        admin_button = gr.Button("Сохранить изменения")


    # 1. ВКЛАДКА С ИМИТАЦИЕЙ ЗВОНКА
    # при нажатии на кнопку звонка (появляется проигрыватель с приветственным аудио)
    call_btn.click(
        fn=lambda: call.welcome_audio_path,
        outputs=player1
    ).then(
        fn=lambda: gr.update(visible=False),
        outputs=call_btn
    ).then(
        fn=update_visibility,
        outputs=[player_start_text, player1]
    )

    # когда воспроизведение записи останавливается (появляется рекордер)
    player1.stop(
        fn=update_invisibility,
        outputs=[player_start_text, player1]
    ).then(
        fn=lambda: [gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)],
        outputs=[recorder_text, recorder, recorder_final_text]
    )

    # стриминг с микрофона
    recorder.stream(
        fn=call.process_partial_chunk,
        inputs=[recorder, audio_state, stream_state],
        outputs=[audio_state, stream_state],
        time_limit=30,
        stream_every=0.5
    )

    # показ кнопки подтверждения возраста (если триггер сработал)
    # stream_state.change(
    #     fn=update_ui_from_state,
    #     inputs=stream_state,
    #     outputs=confirm_age_btn
    # )
    # или так
    timer = gr.Timer(value=1.0)
    timer.tick(
        fn=update_ui_from_state,
        inputs=stream_state,
        outputs=confirm_age_btn
    )
    # и скрываем её, если возраст 18+ подтвердили
    confirm_age_btn.click(
        fn=confirm_age,
        inputs=stream_state,
        outputs=stream_state
    ).then(
        fn=update_ui_from_state,
        inputs=stream_state,
        outputs=confirm_age_btn
    )

    # при остановке записи (пользователем или как истечёт время)
    recorder.stop_recording(
        fn=lambda: call.final_audio_path,
        outputs=player2
    ).then(
        fn=lambda: [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)],
        outputs=[recorder_text, recorder, recorder_final_text]
    ).then(
        fn=update_visibility,
        outputs=[player_end_text, player2]
    ).then(
        fn=call.process_full_audio,
        inputs=[audio_state, stream_state],
        outputs=[routing_result, audio_state, stream_state]
    )

    # когда воспроизведение второй записи останавливается (появляется финальное поле)
    player2.stop(
        fn=update_invisibility,
        outputs=[player_end_text, player2]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=routing_result
    )


demo.launch(share=True, max_file_size=None)
