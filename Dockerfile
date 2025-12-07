# ML Odyssey - Mojo Development Environment
# Multi-stage Dockerfile for Mojo-based AI research platform

# Stage 1: Base image with Pixi and system dependencies
FROM ubuntu:22.04 AS base

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    ca-certificates \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Pixi (modern conda alternative)
# Pixi manages Mojo and Python dependencies
RUN curl -fsSL https://pixi.sh/install.sh | bash

# Add Pixi to PATH
ENV PATH="/root/.pixi/bin:${PATH}"

# Set working directory
WORKDIR /workspace

# Stage 2: Development environment
FROM base AS development

# Copy dependency files first for better caching
COPY pixi.toml pixi.lock ./
COPY pyproject.toml requirements.txt requirements-dev.txt ./

# Install project dependencies via Pixi
# This installs Mojo, pre-commit, and other conda dependencies
RUN pixi install

# Copy pre-commit configuration
COPY .pre-commit-config.yaml ./

# Install pre-commit hooks
# Note: Hooks will be installed in the Git working directory when mounted
RUN pixi run pre-commit install --install-hooks || true

# Copy the rest of the project
COPY . .

# Set up Python path
ENV PYTHONPATH=/workspace:${PYTHONPATH}

# Default command: Start a bash shell with Pixi environment activated
CMD ["pixi", "shell"]

# Stage 3: CI/Testing environment (minimal, optimized for CI)
FROM development AS ci

# Run tests by default in CI mode
CMD ["pixi", "run", "pytest", "tests/", "-v"]

# Stage 4: Production (if needed for deployment)
FROM base AS production

# Copy only necessary files for production
COPY --from=development /workspace /workspace

# Set production environment
ENV ENVIRONMENT=production

# Default production command (customize as needed)
CMD ["pixi", "shell"]
