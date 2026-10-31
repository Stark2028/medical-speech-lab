FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    MODEL_ID=openai/whisper-tiny.en \
    MODEL_REVISION=87c7102498dcde7456f24cfd30239ca606ed9063

WORKDIR /app
RUN groupadd --system app && useradd --system --gid app --home /app app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-api.txt .
RUN pip install --no-cache-dir torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-api.txt

COPY medspeech ./medspeech
COPY web ./web
RUN mkdir -p /opt/hf && chown -R app:app /opt/hf /app
USER app

# Bake the pinned public baseline into the image for predictable cold starts.
RUN python -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration; m='openai/whisper-tiny.en'; r='87c7102498dcde7456f24cfd30239ca606ed9063'; WhisperProcessor.from_pretrained(m,revision=r); WhisperForConditionalGeneration.from_pretrained(m,revision=r)"
ENV HF_HUB_OFFLINE=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready',timeout=4)" || exit 1
CMD ["uvicorn", "medspeech.service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
