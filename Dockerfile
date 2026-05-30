# Use Python 3.12 slim image to keep the container lightweight
FROM python:3.12-slim

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# Install system dependencies (ffmpeg is REQUIRED for Whisper and Torchaudio)
RUN apt-get update && apt-get install -y ffmpeg git build-essential && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Pre-install PyTorch CPU-only version first so Cloud Build doesn't OOM downloading 3GB CUDA drivers
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install dependencies globally using standard pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Pre-download the heavy ML models into the Docker image so Cloud Run doesn't download them on every cold start!
# Cloud Run's filesystem is a RAM disk. Downloading at runtime consumes memory and causes OOM crashes.
RUN python -c "import whisper; whisper.load_model('base', device='cpu')"
RUN python -c "from transformers import pipeline; pipeline('audio-classification', model='ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')"

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Run the application using the JSON array syntax to bypass any shell expansion quirks
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
