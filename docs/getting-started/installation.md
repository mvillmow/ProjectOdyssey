# ML Odyssey Installation Guide

## Overview

This guide explains how to install and set up the ML Odyssey development environment. ML Odyssey is a Mojo-based AI
research platform for reproducing classic research papers.

## Installation Options

### Option 1: Docker (Recommended)

The easiest and most consistent way to get started:

**Requirements**:

- Docker 20.10 or later
- Docker Compose 1.29 or later
- 8GB RAM minimum, 16GB recommended
- 10GB free disk space

**Installation Steps**:

1. Clone the repository:

```bash
git clone https://github.com/mvillmow/ml-odyssey.git
cd ml-odyssey
```text

1. Build and start the development environment:

```bash
docker-compose up -d ml-odyssey-dev
```text

1. Enter the development container:

```bash
docker-compose exec ml-odyssey-dev bash
```text

1. Verify installation:

```bash
# Check Mojo version
mojo --version

# Run tests
pixi run pytest tests/

# Run pre-commit hooks
pixi run pre-commit run --all-files
```text

**Docker Services**:

- `ml-odyssey-dev` - Development environment with full tooling
- `ml-odyssey-ci` - CI/Testing environment (optimized for automated tests)
- `ml-odyssey-prod` - Production environment (read-only volumes)

**Common Docker Commands**:

```bash
# Build all services
docker-compose build

# Start development environment
docker-compose up -d ml-odyssey-dev

# Run tests in CI environment
docker-compose run --rm ml-odyssey-ci

# Stop all services
docker-compose down

# Clean up volumes (removes caches)
docker-compose down -v
```text

### Option 2: Local Installation with Pixi

For native development without Docker:

**Requirements**:

- Python 3.11 or later
- Git
- Linux, macOS (Intel/Apple Silicon), or Windows (WSL2)
- 8GB RAM minimum, 16GB recommended
- 5GB free disk space

**Installation Steps**:

1. Clone the repository:

```bash
git clone https://github.com/mvillmow/ml-odyssey.git
cd ml-odyssey
```text

1. Install Pixi:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```text

1. Install project dependencies:

```bash
pixi install
```text

1. Activate the Pixi environment:

```bash
pixi shell
```text

1. Install pre-commit hooks:

```bash
pre-commit install
```text

1. Verify installation:

```bash
# Check Mojo version
mojo --version

# Run tests
pytest tests/

# Run pre-commit hooks
pre-commit run --all-files
```text

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_example.py

# Run tests with coverage
pytest --cov=src tests/
```text

### Code Quality Checks

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Run specific hook
pre-commit run mojo-format --all-files

# Format Mojo code
mojo format src/**/*.mojo
```text

### Working with GitHub Issues

The project uses a hierarchical planning structure with automated GitHub issue creation:

```bash
# Create issues from plan files
python3 scripts/create_issues.py

# Create issues for a specific section
python3 scripts/create_issues.py --section 01-foundation

# Resume interrupted issue creation
python3 scripts/create_issues.py --resume
```text

## Troubleshooting

### Docker Issues

**Issue**: Docker container fails to start

**Solution**:

1. Verify Docker is running: `docker ps`
2. Check Docker version: `docker --version` (requires 20.10+)
3. Rebuild containers: `docker-compose build --no-cache`
4. Check logs: `docker-compose logs ml-odyssey-dev`

**Issue**: Permission denied errors

**Solution**:

1. Ensure your user is in the docker group: `sudo usermod -aG docker $USER`
2. Log out and log back in for group changes to take effect
3. Or run with sudo: `sudo docker-compose up -d`

### Pixi Issues

**Issue**: Pixi installation fails

**Solution**:

1. Verify shell profile was updated: Check `~/.bashrc` or `~/.zshrc`
2. Restart shell: `exec $SHELL`
3. Verify Pixi is in PATH: `which pixi`

**Issue**: Dependencies fail to install

**Solution**:

1. Update Pixi: `pixi self-update`
2. Clear cache: `rm -rf ~/.pixi/cache`
3. Retry installation: `pixi install`

### Mojo Issues

**Issue**: Mojo command not found

**Solution**:

1. Verify Pixi environment is active: `pixi shell`
2. Check Mojo installation: `pixi list | grep mojo`
3. Reinstall dependencies: `pixi install`

**Issue**: Mojo compilation errors

**Solution**:

1. Check Mojo version: `mojo --version` (requires 0.25.7+)
2. Clean build artifacts: `find . -name "*.mojopkg" -delete`
3. Update Mojo: Pixi will handle this automatically

### Pre-commit Hook Issues

**Issue**: Pre-commit hooks fail

**Solution**:

1. Verify pre-commit is installed: `pre-commit --version`
2. Reinstall hooks: `pre-commit install --install-hooks`
3. Update hooks: `pre-commit autoupdate`
4. Run with verbose output: `pre-commit run --all-files --verbose`

### Test Failures

**Issue**: Tests fail to run

**Solution**:

1. Verify pytest is installed: `pytest --version`
2. Install test dependencies: `pixi install`
3. Check Python version: `python --version` (requires 3.11+)

## Environment Variables

The project uses these environment variables:

- `GITHUB_TOKEN` - Required for GitHub API access (issue creation, PR management)
- `PIXI_ENVIRONMENT` - Managed by Pixi, do not modify manually

Set environment variables in your shell profile or `.env` file (not tracked by git).

## Upgrading

### Docker Upgrade

```bash
# Pull latest changes
git pull

# Rebuild containers
docker-compose build --no-cache

# Restart services
docker-compose down && docker-compose up -d
```text

### Local Upgrade

```bash
# Pull latest changes
git pull

# Update dependencies
pixi install

# Update pre-commit hooks
pre-commit autoupdate
```text

## Uninstallation

### Docker Uninstallation

```bash
# Stop and remove containers
docker-compose down

# Remove volumes (optional - deletes caches)
docker-compose down -v

# Remove images (optional)
docker rmi ml-odyssey:dev ml-odyssey:ci ml-odyssey:prod
```text

### Local Uninstallation

```bash
# Remove Pixi environment
pixi clean

# Remove pre-commit hooks
pre-commit uninstall

# Remove repository (optional)
cd .. && rm -rf ml-odyssey
```text

## Additional Resources

- **Documentation**: See [README.md](README.md) for project overview
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines
- **Agent System**: See [agents/README.md](agents/README.md) for agent documentation
- **Language Strategy**: See [notes/review/adr/ADR-001-language-selection-tooling.md](notes/review/adr/ADR-001-language-selection-tooling.md)
- **Issue Tracking**: [GitHub Issues](https://github.com/mvillmow/ml-odyssey/issues)

## Support

For issues, questions, or contributions:

- **GitHub Issues**: [https://github.com/mvillmow/ml-odyssey/issues](https://github.com/mvillmow/ml-odyssey/issues)
- **Documentation**: Check the `/notes/review/` directory for comprehensive guides

## License

BSD 3-Clause License - See [LICENSE](LICENSE) file for details
