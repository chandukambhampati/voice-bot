# Use Python 3.12 slim image to keep the container lightweight
FROM python:3.12-slim

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

# Set the working directory
WORKDIR /app

# Install 'uv' for incredibly fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy only dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment using uv
RUN uv sync --frozen --no-dev

# Copy the rest of the application code
COPY . /app

# Expose port 8000 for the Uvicorn server
EXPOSE 8000

# Run the application using the uv-managed virtual environment
CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
