"""Getting audio into a project.

Everything StemKit does downstream assumes one canonical file: 44.1 kHz stereo
WAV. Separation, analysis, and transcription all read that same file, so their
timings line up and a note at 12.4 s means the same thing in every stage.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from config import FFMPEG, FFPROBE, SAMPLE_RATE, YTDLP

AUDIO_SUFFIXES = {
    ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus",
    ".aiff", ".aif", ".wma", ".mp4", ".webm", ".mkv",
}


def have(binary: str) -> bool:
    """True if a binary is on PATH (or is an existing absolute path)."""
    return shutil.which(binary) is not None or Path(binary).exists()


def _run(cmd: list[str], what: str, timeout: int = 900) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"{what} failed — {cmd[0]!r} not found on PATH.") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"{what} timed out after {timeout}s.") from e
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        raise RuntimeError(f"{what} failed:\n" + "\n".join(tail))
    return proc


def probe(path: Path) -> dict:
    """Read duration / sample rate / channels via ffprobe."""
    proc = _run(
        [FFPROBE, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        "ffprobe", timeout=60,
    )
    data = json.loads(proc.stdout)
    stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {}
    )
    duration = float(data.get("format", {}).get("duration") or 0.0)
    return {
        "duration": round(duration, 3),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
        "codec": stream.get("codec_name", "unknown"),
    }


def to_wav(src: Path, dest: Path, sample_rate: int = SAMPLE_RATE) -> Path:
    """Transcode anything ffmpeg can read into 44.1 kHz stereo 16-bit WAV."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [FFMPEG, "-y", "-i", str(src),
         "-vn",                     # drop any video track (mp4/webm sources)
         "-ac", "2",
         "-ar", str(sample_rate),
         "-c:a", "pcm_s16le",
         str(dest)],
        "ffmpeg transcode",
    )
    return dest


def download(url: str, dest_dir: Path) -> Path:
    """Fetch the audio track of a URL with yt-dlp. Returns the downloaded file.

    This is a thin wrapper — StemKit does not bundle yt-dlp and does not work
    around DRM. Downloading from a site whose terms forbid it is the caller's
    call to make.
    """
    if not have(YTDLP):
        raise RuntimeError(
            "yt-dlp is not installed. Install it (`pip install yt-dlp`) or import "
            "a local file instead."
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    before = set(dest_dir.iterdir())
    _run(
        [YTDLP, "-f", "bestaudio/best", "--no-playlist",
         "-o", str(dest_dir / "%(title).80s.%(ext)s"), url],
        "yt-dlp download", timeout=1800,
    )
    new = [p for p in dest_dir.iterdir() if p not in before and p.is_file()]
    if not new:
        raise RuntimeError("yt-dlp reported success but produced no file.")
    return max(new, key=lambda p: p.stat().st_size)


def load_mono(path: Path, sample_rate: int = SAMPLE_RATE):
    """Load audio as a mono float32 numpy array. Used by the analysis stages."""
    try:
        import librosa
    except ImportError as e:
        raise RuntimeError(
            "librosa is required for analysis — pip install librosa soundfile"
        ) from e
    y, sr = librosa.load(str(path), sr=sample_rate, mono=True)
    return y, sr
