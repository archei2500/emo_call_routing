# emo_call_routing
An application demonstrating smart call routing.

requirements_cuda - требует пересмотра версии первой экстры в зависимости от типа устройства cuda.

Порядок установки зависимостей:

pip install gigaam
pip install "onnxruntime>=1.18.1"
pip install -r requirements.txt
Далее запустите один из файлов в зависимости от того, у вас в системе CPU или CUDA:

pip install -r requirements-cpu.txt
ИЛИ
pip install -r requirements-cuda.txt