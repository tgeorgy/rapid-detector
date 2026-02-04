FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA environment for building flash-attn
ENV CUDA_HOME=/usr/local/cuda

# Install uv for faster package installation
RUN pip install --no-cache-dir uv

# Copy only dependency files first
COPY pyproject.toml .

# Create minimal package structure for editable install
RUN mkdir -p rapid_detector && touch rapid_detector/__init__.py

# Install flash-attn separately (requires torch at build time)
RUN uv pip install --system --no-build-isolation flash-attn

# Install remaining dependencies (cached unless pyproject.toml changes)
RUN uv pip install --system -e .

# Copy actual source code
COPY rapid_detector/ rapid_detector/
COPY app/ app/
COPY start_services.py .

# Create cache directory with open permissions for --user flag
RUN mkdir -p /cache/rapid_detector/images && chmod -R 777 /cache
ENV RAPID_DETECTOR_CACHE=/cache/rapid_detector

EXPOSE 8000

CMD ["python", "start_services.py"]
