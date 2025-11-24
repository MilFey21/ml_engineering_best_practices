# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

# Copy project files
COPY pyproject.toml ./
COPY src ./src
COPY Makefile ./

# Install dependencies using pixi
RUN pixi install

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "src.modeling.train"]
