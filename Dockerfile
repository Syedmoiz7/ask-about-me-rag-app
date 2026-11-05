# Container recipe for deploying to Hugging Face Spaces (or any Docker host).
FROM python:3.12-slim

# Hugging Face Spaces runs containers as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    DJANGO_DEBUG=false \
    DJANGO_ALLOWED_HOSTS=.hf.space,localhost,127.0.0.1 \
    LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/ \
    LLM_MODEL=gemini-3.5-flash-lite
# LLM_API_KEY is NOT set here: add it as a secret in the Space settings.

WORKDIR /home/user/app

# CPU-only PyTorch: the default Linux build bundles GPU (CUDA) libraries,
# several GB we don't need. Installed first so requirements.txt reuses it.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Build the FAISS index during the image build. This also downloads the
# embedding model into the image, so the app starts fast and works offline.
RUN python ingest.py

# The model is now inside the image: don't contact Hugging Face Hub at startup.
ENV HF_HUB_OFFLINE=1

EXPOSE 7860
# 1 worker: each worker loads its own copy of the model (~650 MB).
# Threads handle several requests at once; 60 s timeout covers slow LLM calls.
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:7860", \
     "--workers", "1", "--threads", "4", "--timeout", "60"]
