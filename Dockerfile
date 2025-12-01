# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

# Copy project files
COPY pyproject.toml ./
COPY pixi.lock ./
COPY src ./src
COPY Makefile ./
COPY .dvc ./.dvc
COPY .dvcignore ./.dvcignore
COPY README.md ./
# Copy data files
COPY data ./data

# Install dependencies using pixi
RUN pixi install

# Initialize DVC (if not already initialized)
RUN dvc init --no-scm || true
RUN dvc remote add -d local ./dvc_storage || true

# Set Python path
ENV PYTHONPATH=/app

# Expose MLflow UI port
EXPOSE 5000

# Default command - run training using pixi's environment
# Use pixi run train task which uses the correct Python environment
CMD ["pixi", "run", "train"]
