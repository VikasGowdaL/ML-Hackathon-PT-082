#!/usr/bin/env python3
"""Cross-platform environment setup script for PathSnap.

Works uniformly on macOS, Linux, and Windows.
Uses pure Python with standard library only (pathlib, subprocess, sys, venv).
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import venv
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    venv_dir = repo_root / ".venv"
    req_file = repo_root / "requirements.txt"

    print("=" * 60)
    print(" PathSnap Cross-Platform Environment Setup")
    print(f" OS: {platform.system()} ({platform.machine()})")
    print(f" Python Executable: {sys.executable}")
    print(f" Python Version: {platform.python_version()}")
    print("=" * 60)

    # 1. Create virtual environment if it doesn't already exist
    if not venv_dir.exists():
        print(f"[*] Creating virtual environment in: {venv_dir}")
        builder = venv.EnvBuilder(with_pip=True, clear=False)
        builder.create(venv_dir)
        print("[+] Virtual environment created successfully.")
    else:
        print(f"[*] Existing virtual environment detected at: {venv_dir}")

    # Determine paths to python and pip within the venv
    is_windows = platform.system() == "Windows"
    if is_windows:
        venv_python = venv_dir / "Scripts" / "python.exe"
        venv_pip = venv_dir / "Scripts" / "pip.exe"
    else:
        venv_python = venv_dir / "bin" / "python"
        venv_pip = venv_dir / "bin" / "pip"

    if not venv_python.exists():
        print(f"[!] Error: Virtualenv Python not found at {venv_python}", file=sys.stderr)
        return 1

    # 2. Upgrade pip, setuptools, wheel
    print("[*] Upgrading pip, setuptools, and wheel...")
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        check=True,
    )

    # 3. Install requirements
    if req_file.exists():
        print(f"[*] Installing requirements from {req_file.name}...")
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-r", str(req_file)],
            check=True,
        )
    else:
        print(f"[!] Warning: {req_file} not found; skipping requirements install.")

    # 4. Install local package in editable mode
    pyproject = repo_root / "pyproject.toml"
    if pyproject.exists():
        print("[*] Installing pathsnap in editable mode (pip install -e .)...")
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-e", "."],
            cwd=str(repo_root),
            check=True,
        )

    # 5. Output activation instructions
    print("\n" + "=" * 60)
    print(" [✓] Environment setup completed successfully!")
    print("=" * 60)
    print("To activate this environment:")
    if is_windows:
        print("\n  In Command Prompt (cmd.exe):")
        print(r"    .venv\Scripts\activate.bat")
        print("\n  In PowerShell:")
        print(r"    .venv\Scripts\Activate.ps1")
        print("    (Note: If script execution is restricted in PowerShell, run:")
        print("     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser)")
    else:
        print("\n  In macOS / Linux (bash or zsh):")
        print("    source .venv/bin/activate")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
