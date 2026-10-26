"""Stem separation with Demucs.

Demucs runs as a subprocess rather than an in-process import. That keeps torch
out of the MCP server's memory, means a crashed separation can't take the server
down with it, and makes the exact command reproducible from the shell.

htdemucs_6s produces six stems. Five are usefully isolated; `other` is the
catch-all where strings, brass, pads, and synths end up together — no open model
separates those from each other, so `other` is best treated as raw audio rather
than something to transcribe note-for-note.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from config import DEMUCS_MODEL, DEVICE

# What each model emits, so callers know what to expect before it runs.
MODEL_STEMS = {
    "htdemucs": ["drums", "bass", "other", "vocals"],
    "htdemucs_ft": ["drums", "bass", "other", "vocals"],
    "htdemucs_6s": ["drums", "bass", "other", "vocals", "guitar", "piano"],
    "mdx_extra": ["drums", "bass", "other", "vocals"],
}


def resolve_device(device: str = DEVICE) -> str:
    """Turn 'auto' into 'cuda' or 'cpu' by asking torch."""
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def available() -> bool:
    """True if demucs is importable in this interpreter."""
    try:
        import demucs  # noqa: F401
        return True
    except ImportError:
        return False


def separate(source: Path, out_dir: Path, model: str = DEMUCS_MODEL,
             device: str | None = None, timeout: int = 7200) -> dict[str, Path]:
    """Split `source` into stems under `out_dir`. Returns {stem_name: wav_path}.

    Demucs writes to <out_dir>/<model>/<track stem>/<stem>.wav; the files are
    moved up into out_dir so the layout doesn't depend on the model choice.
    """
    if not available():
        raise RuntimeError(
            "demucs is not installed — pip install demucs (it pulls in torch, "
            "which is a large download)."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = resolve_device(device or DEVICE)

    cmd = [sys.executable, "-m", "demucs",
           "-n", model,
           "-o", str(out_dir),
           "--device", dev,
           str(source)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"Demucs timed out after {timeout}s. Long track on CPU? "
            f"A GPU (or a shorter excerpt) will help."
        ) from e
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-10:]
        raise RuntimeError("Demucs failed:\n" + "\n".join(tail))

    produced = out_dir / model / source.stem
    if not produced.is_dir():
        raise RuntimeError(
            f"Demucs finished but no output at {produced}. "
            f"Check the model name {model!r}."
        )

    stems: dict[str, Path] = {}
    for wav in sorted(produced.glob("*.wav")):
        dest = out_dir / wav.name
        if dest.exists():
            dest.unlink()
        wav.replace(dest)
        stems[wav.stem] = dest

    # Clean up the now-empty <out_dir>/<model>/<track>/ scaffolding.
    for d in (produced, produced.parent):
        try:
            d.rmdir()
        except OSError:
            pass  # not empty (another track separated here) — leave it

    if not stems:
        raise RuntimeError("Demucs produced no stem files.")
    return stems
