FROM python:3.11-slim

# HF Spaces requires user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Install system dependencies
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
USER user

# Install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Demucs model weights at build time (~200MB)
# This prevents a slow cold-start download on first request
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs_6s')"

# Copy application code
COPY --chown=user app/ app/

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
