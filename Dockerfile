# Use Python 3.12 slim image to keep the container lightweight
FROM python:3.12-slim

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies globally using standard pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Run the application (uvicorn is installed globally, so no venv path is needed)
CMD python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
