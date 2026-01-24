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


with gr.Blocks() as demo:
    with gr.Tab("Симуляция звонка"):
        gr.Markdown("### <center>Нажмите на кнопку ниже, чтобы начать звонок")
        call_btn = gr.Button("Позвонить")
        hidden_recorder = gr.Audio(sources=["microphone"], streaming=True, type="numpy", visible=False, interactive=True, elem_id="recorder")
        audio_state = gr.State(value={"full_buffer": [], "partial_buffer": []})  # cостояние для накопления чанков
        hidden_player = gr.Audio(format="mp3", visible=False, autoplay=False, interactive=False, elem_id="player")
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

    # при нажатии на кнопку звонка
    call_btn.click(
        fn=lambda: call.welcome_audio_path,
        outputs=hidden_player
    ).then(
        js=f"""
            () => {{
                const player = document.querySelector('#player audio');
                if (!player) return;
    
                player.play().catch(e => console.log("play error:", e));
    
                setTimeout(() => {{
                    const btn = document.querySelector('#recorder button');
                    if (btn) btn.click();
                }}, {call.delay_ms});
            }}
            """
    ).then(
        js="""
        () => {
            const recorder = document.querySelector('#recorder');
            if (!recorder) return;

            // сомнительно
            const stopBtn = recorder.querySelectorAll('button')[1];  // или 'button[aria-label="Stop Recording"]'
            const stopBtn = buttons.length >= 2 ? buttons[1] : null;
            if (stopBtn && stopBtn.innerHTML.includes('square') || stopBtn.textContent.trim() === '') {
                stopBtn.click();
            }

            setTimeout(() => {
                if (stopBtn) {
                    stopBtn.click();
                    console.log("Авто-остановка записи через 120 секунд");
                }
            }, 120000);
        }
        """
    )

    # стриминг с микрофона
    hidden_recorder.stream(
        fn=call.process_partial_chunk,
        inputs=[hidden_recorder, audio_state],
        outputs=audio_state,
        time_limit=150,
        stream_every=0.5
    )

    # при остановке записи
    hidden_recorder.stop_recording(
        fn=lambda: call.final_audio_path,
        outputs=hidden_player
    ).then(
        fn=call.process_full_audio,
        inputs=audio_state,
        outputs=[routing_result]
    ).then(
        fn=lambda: {"full_buffer": [], "partial_buffer": []},  # очистка
        outputs=audio_state
    )


demo.launch(share=True, max_file_size=None)
