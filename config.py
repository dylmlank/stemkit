"""Shared settings for StemKit. Env vars override the defaults."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).parent

# Where per-song projects live: <PROJECTS_DIR>/<project>/{source,stems,midi,...}
PROJECTS_DIR = Path(os.environ.get("STEMKIT_PROJECTS", str(ROOT / "projects")))

# Demucs model. htdemucs_6s gives 6 stems (drums/bass/vocals/guitar/piano/other);
# htdemucs gives the faster 4-stem split (drums/bass/vocals/other).
DEMUCS_MODEL = os.environ.get("STEMKIT_DEMUCS_MODEL", "htdemucs_6s")

# auto -> cuda when torch reports it, else cpu. Force with cuda|cpu.
DEVICE = os.environ.get("STEMKIT_DEVICE", "auto")

# External binaries. ffmpeg is required; yt-dlp is only needed for URL imports.
FFMPEG = os.environ.get("STEMKIT_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("STEMKIT_FFPROBE", "ffprobe")
YTDLP = os.environ.get("STEMKIT_YTDLP", "yt-dlp")

# Everything is resampled to this before analysis so timings line up.
SAMPLE_RATE = int(os.environ.get("STEMKIT_SAMPLE_RATE", "44100"))

# MIDI ticks per quarter note for exported files.
TICKS_PER_BEAT = 480

# Generation uses the native Anthropic API. Requires ANTHROPIC_API_KEY in .env
MODEL = os.environ.get("STEMKIT_MODEL", "claude-opus-5")
