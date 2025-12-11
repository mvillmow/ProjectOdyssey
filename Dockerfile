# ML Odyssey - Mojo Development Environment
# Multi-stage Dockerfile

# ---------------------------
# Stage 1: Base image with system deps
# ---------------------------
FROM ubuntu:22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies as root
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    ca-certificates \
    vim \
    uuid \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# Stage 1.5: Create dev user
# ---------------------------
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME=dev

RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USER_NAME}

# Set environment for dev user
ENV HOME=/home/${USER_NAME}
ENV PATH="$HOME/.pixi/bin:$PATH"

# ---------------------------
# Stage 2: Development environment
# ---------------------------
FROM base AS development

# Switch to dev user
USER ${USER_NAME}
WORKDIR /workspace

# Install Pixi as dev user
RUN curl -fsSL https://pixi.sh/install.sh | bash

# Ensure Rattler/Conda cache is writable
RUN mkdir -p $HOME/.cache/rattler/cache && \
    chmod -R 700 $HOME/.cache/rattler

# Copy project dependency files
COPY --chown=${USER_NAME}:${USER_NAME} pixi.toml pixi.lock pyproject.toml requirements.txt requirements-dev.txt ./

# Install project dependencies
RUN pixi install

# Pre-commit
COPY --chown=${USER_NAME}:${USER_NAME} .pre-commit-config.yaml ./
RUN pixi run pre-commit install --install-hooks || true

# Copy the rest of the workspace
COPY --chown=${USER_NAME}:${USER_NAME} . .

# Set Python path
ENV PYTHONPATH=/workspace:${PYTHONPATH:-}

# Default shell
CMD ["pixi", "shell"]

# ---------------------------
# Stage 3: CI / Testing
# ---------------------------
FROM development AS ci

CMD ["pixi", "run", "pytest", "tests/", "-v"]

# ---------------------------
# Stage 4: Production
# ---------------------------
FROM base AS production

# Copy dev workspace
COPY --from=development /workspace /workspace

ENV ENVIRONMENT=production
USER ${USER_NAME}
WORKDIR /workspace

CMD ["pixi", "shell"]
