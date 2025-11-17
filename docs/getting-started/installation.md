# Installation Guide

This guide will help you set up ML Odyssey on your local machine. ML Odyssey is a Mojo-based AI research platform
for reproducing classic research papers.

## Prerequisites

### System Requirements

**Operating System**:

- Linux (64-bit) - Primary supported platform
- macOS (Intel/Apple Silicon) - Experimental support through Mojo
- Windows - Via WSL2 (Windows Subsystem for Linux)

**Hardware Requirements**:

- CPU: Modern 64-bit processor (x86_64 or ARM64)
- RAM: Minimum 4GB, recommended 8GB+ for training models
- Disk Space: At least 2GB free space
- Internet connection for downloading dependencies

### Required Software

Before installing ML Odyssey, ensure you have:

1. **Git** - Version control system
   - Check: `git --version`
   - Install: [https://git-scm.com/downloads](https://git-scm.com/downloads)

2. **Python 3.7+** - For automation scripts
   - Check: `python3 --version`
   - Install: [https://www.python.org/downloads/](https://www.python.org/downloads/)

3. **Pixi** - Environment manager (we'll install this in the next section)
   - Manages Mojo and all project dependencies automatically

**Note**: You do NOT need to install Mojo separately - Pixi will handle this automatically.

## Installation Steps

### Step 1: Install Pixi

Pixi is our environment manager that handles all dependencies including Mojo.

**Linux/macOS**:

```bash

curl -fsSL <https://pixi.sh/install.sh> | bash

```text

**Windows (PowerShell)**:

```powershell

iwr -useb <https://pixi.sh/install.ps1> | iex

```text

**Verify Installation**:

```bash

pixi --version

```text

You should see output like: `pixi 0.x.x`

**Troubleshooting**: If `pixi` command is not found, restart your terminal or add Pixi to your PATH:

```bash

# Add to ~/.bashrc or ~/.zshrc
export PATH="$HOME/.pixi/bin:$PATH"

```text

### Step 2: Clone the Repository

Clone the ML Odyssey repository to your local machine:

```bash

# Clone via HTTPS
git clone <https://github.com/your-org/ml-odyssey.git>

# OR clone via SSH (if you have SSH keys configured)
git clone git@github.com:your-org/ml-odyssey.git

# Navigate into the directory
cd ml-odyssey

```text

### Step 3: Install Dependencies with Pixi

Pixi will automatically install Mojo and all other dependencies defined in `pixi.toml`:

```bash

# Install all dependencies (this may take a few minutes)
pixi install

# Activate the Pixi environment
pixi shell

```text

**What happens during `pixi install`**:

- Downloads and installs Mojo compiler (v0.25.7+)
- Installs pre-commit hooks framework
- Installs PyGitHub for GitHub automation
- Sets up the project environment

**Note**: The first installation may take 5-10 minutes depending on your internet connection.

### Step 4: Set Up Pre-commit Hooks

Pre-commit hooks automatically check code quality before commits:

```bash

# Install pre-commit hooks (one-time setup)
pre-commit install

```text

**Configured hooks**:

- `markdownlint-cli2` - Lint markdown files
- `trailing-whitespace` - Remove trailing whitespace
- `end-of-file-fixer` - Ensure files end with newline
- `check-yaml` - Validate YAML syntax
- `check-added-large-files` - Prevent large files (max 1MB)
- `mixed-line-ending` - Fix mixed line endings

**Note**: The `mojo format` hook is currently disabled due to a known bug. See `.pre-commit-config.yaml` for details.

## Verification

### Verify Pixi Installation

```bash

pixi --version

```text

Expected output: `pixi 0.x.x` or similar

### Verify Mojo Installation

```bash

# Inside the pixi shell
mojo --version

```text

Expected output: `mojo 0.25.7` or later

**If Mojo is not found**:

```bash

# Ensure you're in the pixi shell
pixi shell

# Try again
mojo --version

```text

### Verify Python Installation

```bash

python3 --version

```text

Expected output: `Python 3.7.x` or later

### Verify Pre-commit Hooks

```bash

# Run pre-commit on all files (should pass)
pre-commit run --all-files

```text

Expected output: All hooks should pass with green checkmarks.

### Run a Simple Test

Test that everything is working by running a simple Mojo command:

```bash

# Create a simple test file
cat > test.mojo << 'EOF'
fn main():
    print("ML Odyssey is ready!")
EOF

# Run it
mojo run test.mojo

# Clean up
rm test.mojo

```text

Expected output: `ML Odyssey is ready!`

## Next Steps

Now that you have ML Odyssey installed, you can:

1. **Explore the repository structure**: Read [repository-structure.md](repository-structure.md)
2. **Try the quick start guide**: Follow [quickstart.md](quickstart.md)
3. **Run your first model**: See [first_model.md](first_model.md)
4. **Read the documentation**: Browse `docs/` directory
5. **Explore examples**: Check `examples/` directory

## Troubleshooting

### Issue: `pixi` command not found after installation

**Solution**: Restart your terminal or add Pixi to PATH manually:

```bash

# For bash
echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# For zsh
echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

```text

### Issue: `mojo` command not found

**Solution**: Ensure you're in the Pixi shell:

```bash

# Activate the environment
pixi shell

# Verify Mojo is available
mojo --version

```text

### Issue: Pixi installation fails on Linux

**Solution**: Install dependencies for your distribution:

**Ubuntu/Debian**:

```bash

sudo apt-get update
sudo apt-get install curl build-essential

```text

**Fedora/RHEL**:

```bash

sudo dnf install curl gcc gcc-c++ make

```text

### Issue: Pre-commit hooks fail

**Solution**: Run pre-commit manually to see detailed errors:

```bash

# Run with verbose output
pre-commit run --all-files --verbose

# Auto-fix most issues
pre-commit run --all-files

```text

Most formatting issues are auto-fixed by pre-commit. Just review and commit the changes.

### Issue: Permission denied when cloning repository

**Solution**: Check your GitHub authentication:

```bash

# For HTTPS, you may need a personal access token
# For SSH, ensure your SSH key is configured
ssh -T git@github.com

```text

See GitHub's documentation for setting up SSH keys: [https://docs.github.com/en/authentication](https://docs.github.com/en/authentication)

### Issue: Mojo version mismatch

**Solution**: Update the Pixi environment:

```bash

# Update all dependencies to latest compatible versions
pixi update

# Verify Mojo version
pixi shell
mojo --version

```text

### Issue: Disk space errors during installation

**Solution**: Free up disk space and retry:

```bash

# Check available space
df -h

# Clean up Pixi cache if needed
pixi clean cache

```text

### Issue: Network timeouts during installation

**Solution**: Try installing with increased timeout:

```bash

# Set longer timeout for downloads
export PIXI_TIMEOUT=600

# Retry installation
pixi install

```text

## Platform-Specific Notes

### Linux

ML Odyssey is primarily developed and tested on Linux. Installation should be straightforward following the steps above.

**Recommended distributions**:

- Ubuntu 20.04+
- Fedora 35+
- Debian 11+

### macOS

Mojo support on macOS is improving but may have limitations:

- Intel Macs: Generally well-supported
- Apple Silicon (M1/M2): Experimental support

**Known issues**:

- Some SIMD operations may have reduced performance
- Hardware-specific optimizations may not be available

### Windows (WSL2)

Windows users should use WSL2 (Windows Subsystem for Linux):

1. Install WSL2: [https://docs.microsoft.com/en-us/windows/wsl/install](https://docs.microsoft.com/en-us/windows/wsl/install)
2. Install Ubuntu from Microsoft Store
3. Follow the Linux installation instructions above inside WSL2

**Note**: Native Windows support is not currently available.

## Getting Help

If you encounter issues not covered in this guide:

1. **Check existing documentation**: Browse `docs/` directory
2. **Search GitHub Issues**: [https://github.com/your-org/ml-odyssey/issues](https://github.com/your-org/ml-odyssey/issues)
3. **Create a new issue**: Report installation problems with:
   - Your operating system and version
   - Mojo version (`mojo --version`)
   - Python version (`python3 --version`)
   - Full error message and stack trace
   - Steps to reproduce

4. **Read Mojo documentation**: [https://docs.modular.com/mojo/](https://docs.modular.com/mojo/)
5. **Check Pixi documentation**: [https://pixi.sh/latest/](https://pixi.sh/latest/)

## Development Setup (For Contributors)

If you plan to contribute to ML Odyssey, see [CONTRIBUTING.md](../../CONTRIBUTING.md) for additional setup steps including:

- Running tests with `pixi run test`
- Code style guidelines
- Pull request process
- Agent system documentation

---

**Congratulations!** You now have ML Odyssey installed and ready to use. Proceed to [quickstart.md](quickstart.md) to
learn how to use the platform.
