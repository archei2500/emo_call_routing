import gradio as gr
import os
import pandas as pd
# import iso639
# import torch
# import importlib
os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-user'
os.environ['ALSA_CONFIG_PATH'] = '/dev/null'
# import torchvision.transforms.functional as F
# sys.modules['torchvision.transforms.functional_tensor'] = F
# if 'basicsr.data.degradations' in sys.modules:
#     importlib.reload(sys.modules['basicsr.data.degradations'])

# path_to_video = 'vid.mp4'
# asr_model_downloaded = False


# def toggle_vid_upload_fields(upload_method):
#     return [
#         gr.File(visible=upload_method == "From the device"),
#         gr.Textbox(visible=upload_method == "Link from YouTube"),
#     ]
#
#
# def update_uploads(cg_1, cg_2):
#     return [gr.Markdown(visible=False), gr.File(visible="Subtitles" in cg_1, interactive="Subtitles" in cg_1),
#             gr.File(visible="Clone sample (audio prompt)" in cg_1, interactive="Clone sample (audio prompt)" in cg_1),
#             gr.File(visible="Prompt transcription" in cg_1, interactive="Prompt transcription" in cg_1),
#             gr.File(visible="Synthesized speech fragments (zip)" in cg_2,
#                     interactive="Synthesized speech fragments (zip)" in cg_2),
#             gr.File(visible="Video fragments (zip)" in cg_2, interactive="Video fragments (zip)" in cg_2),
#             gr.Checkbox(visible="Clone sample (audio prompt)" not in cg_1,
#                         interactive="Clone sample (audio prompt)" not in cg_1,
#                         value=False),
#             gr.Checkbox(visible="Prompt transcription" not in cg_1, interactive="Prompt transcription" not in cg_1, value=False),
#             gr.Checkbox(visible="Clone sample (audio prompt)" in cg_1,
#                         interactive="Clone sample (audio prompt)" in cg_1,
#                         value=False)
#             #gr.Dropdown(visible="Clone sample (audio prompt)" in cg_1, interactive="Clone sample (audio prompt)" in cg_1)
#             ]

initial_df = pd.DataFrame({
    "Имя": ["Анна", "Борис", "Кирилл"],
    "Возраст": [28, 34, 25],
    "Город": ["Москва", "Санкт-Петербург", "Казань"]
})


with gr.Blocks() as demo:
    with gr.Tab("Симуляция звонка"):
        gr.Markdown("### <center>Нажмите на кнопку ниже, чтобы начать звонок")
        call_btn = gr.Button("Позвонить")
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


demo.launch(share=True, max_file_size=None)
