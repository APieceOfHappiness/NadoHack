# FROM python:3.11-slim

ARG USE_CUDA

FROM pytorch/torchserve:latest-gpu

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U bitsandbytes[cuda]

COPY . .

EXPOSE 5221

CMD ["python3", "main.py"]
# CMD ["gunicorn", "-w", "10", "-b", "0.0.0.0:5221", "main:app"]
