"""Check that this machine can record and, later, train."""

import importlib
import shutil
import subprocess
import sys
from pathlib import Path


RECORDING_IMPORTS = ("numpy", "scipy", "torch", "cv2", "mediapipe")


def import_status():
    status = {}
    for name in RECORDING_IMPORTS:
        try:
            importlib.import_module(name)
            status[name] = True
        except ImportError:
            status[name] = False
    return status


def modal_target():
    """Return the active profile and environment without printing credentials."""
    candidates = [sys.executable, "python3"]
    for executable in candidates:
        if executable != "python3" and not Path(executable).exists():
            continue
        if executable == "python3" and shutil.which("python3") is None:
            continue
        profile = subprocess.run([executable, "-m", "modal", "profile", "current"],
                                 capture_output=True, text=True)
        if profile.returncode != 0:
            continue
        environment = subprocess.run(
            [executable, "-c", "from modal.config import config; print(config.get('environment') or '')"],
            capture_output=True, text=True)
        if environment.returncode != 0:
            continue
        return profile.stdout.strip(), environment.stdout.strip()
    return None, None


def report(model_path):
    lines = []
    ready = True
    for name, present in import_status().items():
        lines.append(f"{'ok' if present else 'missing':7} {name}")
        if not present:
            ready = False
    profile, environment = modal_target()
    if profile == "yaklilu" and environment == "main":
        lines.append("ok      modal profile yaklilu / main")
    elif profile:
        lines.append(f"warn    modal profile {profile} / {environment or 'unset'}")
    else:
        lines.append("warn    modal is not available to this interpreter")
    model = Path(model_path)
    if model.exists() and model.stat().st_size > 0:
        lines.append(f"ok      hand landmarker {model}")
    else:
        lines.append(f"missing hand landmarker {model}")
        ready = False
    return ready, lines
