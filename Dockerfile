FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN git clone https://huggingface.co/spaces/hkchengrex/MMAudio /MMAudio
WORKDIR /MMAudio

RUN sed -e '/^torch ==/d' -e '/^torchvision$/d' -e '/^torchaudio$/d' requirements.txt > /tmp/req.txt && \
    pip install -r /tmp/req.txt runpod --no-cache-dir
RUN pip install -e . --no-cache-dir --no-deps

RUN python -c "\
from mmaudio.eval_utils import all_model_cfg; \
all_model_cfg['large_44k_v2'].download_if_needed(); \
from open_clip import create_model_from_pretrained; \
create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384', return_transform=False)"

COPY handler.py /MMAudio/handler.py

CMD ["python", "-u", "/MMAudio/handler.py"]
