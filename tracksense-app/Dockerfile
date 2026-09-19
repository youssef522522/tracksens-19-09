FROM python:3.10-slim

WORKDIR /app

# System deps needed by Pillow/OpenCV (ultralytics pulls in opencv-python-headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (much smaller than the default CUDA build)
RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces (Docker SDK) expects the app on port 7860
ENV PORT=7860
EXPOSE 7860

# Writable dir for the SQLite file / ultralytics config (HF Spaces runs as non-root)
RUN mkdir -p /app && chmod -R 777 /app
ENV YOLO_CONFIG_DIR=/app/.config/Ultralytics

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
