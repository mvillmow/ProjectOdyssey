# Docker Support for ML Odyssey

This document provides detailed information about using Docker with ML Odyssey.

## Overview

ML Odyssey includes Docker support for consistent development environments across different machines and platforms.
The Docker setup uses Pixi for environment management and supports Mojo development.

## Architecture

### Multi-Stage Dockerfile

The Dockerfile uses a multi-stage build approach:

1. **base** - Ubuntu 22.04 with Pixi and system dependencies
1. **development** - Full development environment with all tools
1. **ci** - Optimized for automated testing
1. **production** - Minimal production deployment

### Docker Compose Services

Three services are defined in `docker-compose.yml`:

- **ml-odyssey-dev** - Interactive development environment
- **ml-odyssey-ci** - Automated testing environment
- **ml-odyssey-prod** - Production deployment

## Quick Start

### Prerequisites

- Docker 20.10 or later
- Docker Compose 1.29 or later

### Build and Run

```bash
# Build all images
docker-compose build

# Start development environment
docker-compose up -d ml-odyssey-dev

# Enter the container
docker-compose exec ml-odyssey-dev bash

# Inside the container
pixi shell                              # Activate Pixi environment
pixi run pytest tests/                  # Run tests
pixi run pre-commit run --all-files    # Run pre-commit hooks
```text

## Common Tasks

### Development Workflow

```bash
# Start development environment
docker-compose up -d ml-odyssey-dev

# Attach to running container
docker-compose exec ml-odyssey-dev bash

# Run Mojo code
pixi run mojo your_script.mojo

# Run Python scripts
pixi run python scripts/your_script.py

# Run tests
pixi run pytest tests/ -v

# Run specific test file
pixi run pytest tests/test_specific.py

# Run pre-commit hooks
pixi run pre-commit run --all-files
```text

### Running Tests

```bash
# Run tests in CI environment (one-off)
docker-compose run --rm ml-odyssey-ci

# Run specific test suite
docker-compose run --rm ml-odyssey-ci pixi run pytest tests/agents/ -v

# Run tests with coverage
docker-compose run --rm ml-odyssey-ci pixi run pytest tests/ --cov=src --cov-report=html
```text

### Code Quality Checks

```bash
# Run pre-commit hooks
docker-compose exec ml-odyssey-dev pixi run pre-commit run --all-files

# Format Mojo code
docker-compose exec ml-odyssey-dev pixi run mojo format src/

# Run ruff linter
docker-compose exec ml-odyssey-dev pixi run ruff check src/

# Run mypy type checking
docker-compose exec ml-odyssey-dev pixi run mypy src/
```text

## Volume Mounts

The development service mounts several volumes:

1. **Source Code** - `.:/workspace` - Live sync with host
1. **Pixi Cache** - `pixi-cache:/root/.pixi` - Persist Pixi packages
1. **Pre-commit Cache** - `pre-commit-cache:/root/.cache/pre-commit` - Persist hook installations
1. **Git Config** - `~/.gitconfig:/root/.gitconfig:ro` - Use host Git configuration (read-only)

## Environment Variables

### Development

- `ENVIRONMENT=development` - Indicates development mode
- `TERM=xterm-256color` - Enable color output
- `PYTHONPATH=/workspace` - Add workspace to Python path

### CI

- `ENVIRONMENT=ci` - Indicates CI mode
- `CI=true` - CI flag for tools that check for CI environment

### Production

- `ENVIRONMENT=production` - Indicates production mode

## Troubleshooting

### Build Failures

If the Docker build fails:

```bash
# Clean build (no cache)
docker-compose build --no-cache ml-odyssey-dev

# View build logs
docker-compose build ml-odyssey-dev 2>&1 | tee build.log
```text

### Pixi Installation Issues

If Pixi fails to install dependencies:

```bash
# Enter container and debug
docker-compose exec ml-odyssey-dev bash

# Manually run Pixi install
pixi install -v

# Check Pixi version
pixi --version

# Update Pixi
curl -fsSL https://pixi.sh/install.sh | bash
```text

### Permission Issues

If you encounter permission issues with mounted volumes:

```bash
# Check file ownership in container
docker-compose exec ml-odyssey-dev ls -la /workspace

# Fix permissions on host (if needed)
sudo chown -R $USER:$USER .
```text

### Container Not Starting

```bash
# View container logs
docker-compose logs ml-odyssey-dev

# Check container status
docker-compose ps

# Restart services
docker-compose restart ml-odyssey-dev
```text

## Advanced Usage

### Custom Build Arguments

You can customize the build with build arguments:

```bash
# Build with specific Python version
docker-compose build --build-arg PYTHON_VERSION=3.11 ml-odyssey-dev
```text

### Using Different Stages

```bash
# Build specific stage
docker build --target ci -t ml-odyssey:ci .

# Run specific stage
docker run --rm -v $(pwd):/workspace ml-odyssey:ci
```text

### Cleaning Up

```bash
# Stop and remove containers
docker-compose down

# Remove containers and volumes
docker-compose down -v

# Remove images
docker rmi ml-odyssey:dev ml-odyssey:ci ml-odyssey:prod

# Full cleanup (containers, volumes, images)
docker-compose down -v --rmi all
```text

## Testing the Docker Setup

To verify the Docker setup is working correctly:

```bash
# 1. Build the development image
docker-compose build ml-odyssey-dev

# 2. Run a quick test
docker-compose run --rm ml-odyssey-dev pixi run python --version

# 3. Verify Mojo is installed
docker-compose run --rm ml-odyssey-dev pixi run mojo --version

# 4. Run the test suite
docker-compose run --rm ml-odyssey-ci

# 5. Start interactive development
docker-compose up -d ml-odyssey-dev
docker-compose exec ml-odyssey-dev bash
```text

## Integration with CI/CD

The Docker setup can be integrated with GitHub Actions or other CI systems:

```yaml
# Example GitHub Actions workflow
- name: Build Docker image
  run: docker-compose build ml-odyssey-ci

- name: Run tests
  run: docker-compose run --rm ml-odyssey-ci
```text

## Best Practices

1. **Use named volumes** for caches to improve build times
1. **Mount source code** as volumes for live development
1. **Use .dockerignore** to exclude unnecessary files from build context
1. **Use multi-stage builds** to optimize image size
1. **Tag images properly** for version management
1. **Keep containers running** with `stdin_open: true` and `tty: true` for development

## Security Considerations

1. **Read-only mounts** in production (`/workspace:ro`)
1. **No hardcoded secrets** in Dockerfile or docker-compose.yml
1. **Regular base image updates** for security patches
1. **Minimal attack surface** in production images
1. **Non-root user** (future improvement)

## Future Enhancements

Potential improvements to the Docker setup:

- [ ] Add non-root user for better security
- [ ] GPU support for ML training (NVIDIA Docker runtime)
- [ ] Multi-architecture builds (ARM64 support)
- [ ] Optimized production image with minimal dependencies
- [ ] Health checks for containers
- [ ] Resource limits (CPU, memory)
- [ ] Docker Compose profiles for different use cases
