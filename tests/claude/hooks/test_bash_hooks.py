import json
import subprocess
import sys
from pathlib import Path
import pytest

HOOK_SCRIPT = Path(".claude/hooks/pre-bash-exec.py")  # Path to your Python hook
PROJECT_ROOT = Path.cwd()

# Helper to run hook with a command and return True if allowed, False if blocked
def run_hook(cmd: str) -> bool:
    payload = {
        "tool": "bash",
        "input": {
            "command": cmd,
            "cwd": str(PROJECT_ROOT)
        }
    }
    try:
        proc = subprocess.run(
            [sys.executable, str(HOOK_SCRIPT)],
            input=json.dumps(payload).encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except Exception as e:
        pytest.fail(f"Hook execution failed: {e}")

    if proc.returncode == 0:
        return True  # allowed
    else:
        try:
            output = json.loads(proc.stdout.decode())
        except json.JSONDecodeError:
            output = {}
        return False  # blocked

# ---- Dangerous commands ----
dangerous_commands = [
    "rm -rf /",
    "rm -rf / ",
    "rm -rf ~/",
    "rm -rf $HOME",
    "rm .git",
    "rm .git/config",
    "rm /etc/passwd",
    "rm -rf /tmp/something",
    "sudo rm -rf /tmp/file",
    "rm ./logs/test.log",
    "rm logs/test.log",
    "rm -rf",
    "echo yes | rm file.txt",
    "cp README.md README.md && rm -rf ~/",
    "rm ./foo || echo yes",
    "rm ./foo; echo yes",
    "rm ./foo && rm ./bar",
    "dd if=/dev/zero of=test.img",
    "mkfs.ext4 /dev/sda",
    "wipefs -a /dev/sda",
    "mount /dev/sda /mnt",
    "umount /mnt",
    "chmod -R 777 ./tests",
    "chown -R user:user ./tests",
    "chgrp -R staff ./tests",
    "tar -xf archive.tar",
    "unzip file.zip",
    "find ./tests -delete",
    "find ./tests -exec rm {} \\;",
    "echo ./tests | xargs rm",
    "git clean -fdx",
    "rsync -av --delete src/ dst/",
    "echo $(ls)",
    "echo `ls`",
    "echo ${PATH}"
]

# ---- Safe commands ----
safe_commands = [
    "rm ./tests/README.md",
    "rm tests/README.md",
    "rm README.md CLAUDE.md",
    "ls -la",
    "git status",
    "echo hello world",
    "mkdir -p ./tmp && touch ./tmp/file",
    "echo foo | tee ./tmp/output",
    "cp README.md ./tmp/",
    f"rm -rf {PROJECT_ROOT}/build" if (PROJECT_ROOT / "build").exists() else None
]
safe_commands = [c for c in safe_commands if c is not None]

@pytest.mark.parametrize("cmd", dangerous_commands)
def test_dangerous_commands_blocked(cmd):
    allowed = run_hook(cmd)
    assert not allowed, f"Dangerous command passed but should be blocked: {cmd}"

@pytest.mark.parametrize("cmd", safe_commands)
def test_safe_commands_allowed(cmd):
    allowed = run_hook(cmd)
    assert allowed, f"Safe command blocked incorrectly: {cmd}"
