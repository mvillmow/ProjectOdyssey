# Installation Guide

Complete setup instructions for ML Odyssey.

## System Requirements

### Operating System

- **Linux**: Ubuntu 20.04+ or similar
- **macOS**: 11.0 (Big Sur) or later
- **Windows**: WSL2 (Ubuntu 20.04+)

### Software Requirements

- **Python**: 3.10 or later
- **Git**: 2.30 or later
- **Pixi**: Latest version (installed automatically)
- **Disk Space**: At least 2GB for dependencies

## Step 1: Install Prerequisites

### Linux (Ubuntu/Debian)

```bash
# Update package lists
sudo apt update

# Install Python, Git, and build tools
sudo apt install -y python3 python3-pip git build-essential curl
```

### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and Git
brew install python git
```

### Windows (WSL2)

1. Install WSL2: [Microsoft's WSL2 Installation Guide](https://docs.microsoft.com/en-us/windows/wsl/install)
2. Install Ubuntu 20.04+ from Microsoft Store
3. Follow Linux instructions above in WSL2 terminal

## Step 2: Clone the Repository

```bash
# Clone from GitHub
git clone https://github.com/mvillmow/ml-odyssey.git

# Navigate to the project directory
cd ml-odyssey
```

## Step 3: Install Pixi

[Pixi](https://pixi.sh) is a fast package manager that handles all ML Odyssey dependencies:

```bash
# Install Pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Reload shell configuration
source ~/.bashrc  # For bash
# or
source ~/.zshrc   # For zsh
```

Verify Pixi installation:

```bash
pixi --version
```

## Step 4: Set Up Project Environment

Pixi will install all dependencies including Mojo:

```bash
# Install all dependencies
pixi install

# This may take 5-10 minutes on first run
```

## Step 5: Verify Installation

### Check Mojo Installation

```bash
pixi run mojo --version
# Should output: mojo 0.25.7 or later
```

### Run Tests

```bash
# Run the test suite
pixi run pytest tests/

# Expected output: All tests should pass
```

### Build Documentation

```bash
# Build docs (optional)
pixi run mkdocs build

# Serve docs locally (optional)
pixi run mkdocs serve
```

## Development Tools Setup

### Pre-commit Hooks

Enable automatic code formatting and linting:

```bash
# Install pre-commit hooks
pixi run pre-commit install

# Test the hooks
pixi run pre-commit run --all-files
```

### VS Code (Recommended)

Install recommended extensions:

1. **Mojo** - Syntax highlighting and LSP
2. **Python** - Python support
3. **Markdown All in One** - Documentation editing

Settings (`.vscode/settings.json`):

```json
{
  "mojo.sdkPath": ".pixi/envs/default/bin/mojo",
  "python.defaultInterpreterPath": ".pixi/envs/default/bin/python",
  "editor.formatOnSave": true
}
```

## Configuration

### Environment Variables

Create a `.env` file (optional):

```bash
# Copy template
cp .env.example .env

# Edit with your preferences
nano .env
```

### Pixi Configuration

The `pixi.toml` file contains all project dependencies. To add new dependencies:

```bash
# Add a Python package
pixi add numpy

# Add a development dependency
pixi add --dev pytest
```

## Troubleshooting

### Pixi Installation Issues

**Problem**: Pixi not found after installation

```bash
# Add Pixi to PATH manually
export PATH="$HOME/.pixi/bin:$PATH"

# Add to shell config for persistence
echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc
```

### Mojo Not Found

**Problem**: `mojo` command not found

```bash
# Use pixi run prefix
pixi run mojo --version

# Or activate the environment
pixi shell
mojo --version
```

### Permission Errors

**Problem**: Permission denied errors

```bash
# Fix file permissions
chmod +x scripts/*.sh

# Fix Python virtual environment
pixi install --force-reinstall
```

### Build Failures

**Problem**: Compilation errors with Mojo code

```bash
# Clean and rebuild
pixi clean
pixi install

# Update Mojo compiler
pixi update mojo
```

### Network Issues

**Problem**: Slow downloads or timeouts

```bash
# Use a different mirror (if available)
pixi config set default_channels conda-forge,defaults

# Retry installation
pixi install --retry 3
```

## Updating ML Odyssey

### Update Code

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pixi update

# Reinstall if needed
pixi install
```

### Update Mojo

```bash
# Update to latest Mojo version
pixi update mojo

# Verify version
pixi run mojo --version
```

## Uninstallation

### Remove Project

```bash
# Navigate to parent directory
cd ..

# Remove project directory
rm -rf ml-odyssey
```

### Remove Pixi

```bash
# Remove Pixi installation
rm -rf ~/.pixi

# Remove from PATH (edit shell config)
nano ~/.bashrc  # Remove Pixi PATH export
```

## Next Steps

- **[Quickstart Guide](quickstart.md)** - Get started in 5 minutes
- **[First Model Tutorial](first_model.md)** - Build your first model
- **[Project Structure](../core/project-structure.md)** - Understand the codebase
- **[Development Guide](../dev/architecture.md)** - Contributing to ML Odyssey

## Getting Help

- **Documentation**: See [docs/index.md](../index.md)
- **GitHub Issues**: [Report problems](https://github.com/mvillmow/ml-odyssey/issues)
- **Discussions**: Ask questions in GitHub Discussions
