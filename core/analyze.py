"""Tempo, key, and chord analysis.

All of this runs on librosa alone. madmom would give better downbeats and
chords, but it breaks on current numpy/Python and pinning the whole kit to an
old numpy to get it is a bad trade — the chroma-template approach below is
accurate enough to hand a musician a usable chord chart.

What each result is worth:
  bpm      reliable on steady-tempo material; half/double-time errors happen
  key      reliable for tonal music; meaningless for atonal or drum-only audio
  chords   good as *labels* (Am7, F, C). Extensions are the weak part
  bars     assumed 4/4 from the first detected beat — not true downbeat tracking
"""
from __future__ import annotations

PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles: how strongly each scale degree is
# expected to sound in major and minor keys.
KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
            2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
            2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

# Chord templates as (suffix, semitone offsets from the root, penalty).
#
# The penalty is what makes this usable. Cosine similarity can only go up when
# a template adds notes, so without it every F reads as Dm7 and every triad
# picks up a sus or a 7th from bleed and overtones. Each template pays for its
# complexity, and an extension has to genuinely be in the chroma to earn its
# keep. Tuned against synthesized progressions with known ground truth.
CHORD_TEMPLATES: list[tuple[str, tuple[int, ...], float]] = [
    ("",      (0, 4, 7),      0.00),   # major
    ("m",     (0, 3, 7),      0.00),   # minor
    ("7",     (0, 4, 7, 10),  0.05),
    ("m7",    (0, 3, 7, 10),  0.05),
    ("maj7",  (0, 4, 7, 11),  0.06),
    ("sus4",  (0, 5, 7),      0.07),
    ("sus2",  (0, 2, 7),      0.07),
    ("dim",   (0, 3, 6),      0.06),
    ("aug",   (0, 4, 8),      0.08),
    ("m6",    (0, 3, 7, 9),   0.09),
    ("6",     (0, 4, 7, 9),   0.09),
]


def _np():
    """numpy, imported on demand so `check_setup` still runs without it."""
    try:
        import numpy
        return numpy
    except ImportError as e:
        raise RuntimeError("numpy is required — pip install numpy") from e


def _unit(v):
    np = _np()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# Built once on first use: all 12 roots x every template, as unit vectors,
# plus the per-template penalty subtracted from each match score.
_CHORD_BANK_CACHE: tuple[list, object, object] | None = None


def _chord_bank() -> tuple[list, object, object]:
    global _CHORD_BANK_CACHE
    if _CHORD_BANK_CACHE is None:
        np = _np()
        names: list[str] = []
        rows = []
        penalties: list[float] = []
        for root in range(12):
            for suffix, offsets, penalty in CHORD_TEMPLATES:
                vec = np.zeros(12)
                for o in offsets:
                    vec[(root + o) % 12] = 1.0
                names.append(f"{PITCHES[root]}{suffix}")
                rows.append(_unit(vec))
                penalties.append(penalty)
        _CHORD_BANK_CACHE = (names, np.vstack(rows), np.array(penalties))
    return _CHORD_BANK_CACHE


def detect_key(chroma) -> dict:
    """Correlate the average chroma against all 24 keys. Returns the best fit."""
    np = _np()
    avg = chroma.mean(axis=1)
    if avg.sum() <= 0:
        return {"key": "unknown", "confidence": 0.0}

    best = ("unknown", -2.0, "")
    for tonic in range(12):
        for mode, profile in (("major", KS_MAJOR), ("minor", KS_MINOR)):
            rotated = np.roll(np.array(profile), tonic)
            # Pearson correlation between the observed and expected profiles.
            corr = float(np.corrcoef(avg, rotated)[0, 1])
            if corr > best[1]:
                best = (f"{PITCHES[tonic]} {mode}", corr, mode)
    return {
        "key": best[0],
        "mode": best[2],
        "confidence": round(max(0.0, best[1]), 3),
    }


def detect_chords(chroma, beat_times, beats_per_chord: int = 2) -> list[dict]:
    """Label chords over groups of beats, then merge consecutive repeats.

    Chroma is averaged across `beats_per_chord` beats before matching — single
    beats are too noisy (passing tones dominate), whole bars miss fast changes.
    """
    if chroma.shape[1] == 0 or len(beat_times) < 2:
        return []

    np = _np()
    chord_names, chord_bank, chord_penalties = _chord_bank()
    chords: list[dict] = []
    for i in range(0, chroma.shape[1], beats_per_chord):
        window = chroma[:, i:i + beats_per_chord]
        if window.shape[1] == 0:
            continue
        vec = _unit(window.mean(axis=1))
        if not vec.any():
            continue
        scores = (chord_bank @ vec) - chord_penalties
        idx = int(np.argmax(scores))
        start = float(beat_times[min(i, len(beat_times) - 1)])
        end_i = min(i + beats_per_chord, len(beat_times) - 1)
        chords.append({
            "chord": chord_names[idx],
            "start": round(start, 3),
            "end": round(float(beat_times[end_i]), 3),
            "confidence": round(float(scores[idx]), 3),
        })

    # Merge runs of the same chord into one span.
    merged: list[dict] = []
    for c in chords:
        if merged and merged[-1]["chord"] == c["chord"]:
            merged[-1]["end"] = c["end"]
            merged[-1]["confidence"] = round(
                max(merged[-1]["confidence"], c["confidence"]), 3
            )
        else:
            merged.append(dict(c))
    return merged


def analyze(path, sample_rate: int | None = None) -> dict:
    """Full analysis pass over one audio file."""
    from core.audio import load_mono

    np = _np()
    y, sr = load_mono(path, sample_rate) if sample_rate else load_mono(path)
    if y.size == 0:
        raise RuntimeError(f"{path} contains no audio.")

    import librosa

    # Beat tracking on the percussive component is noticeably steadier.
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])

    # Chords come from the harmonic component for the same reason.
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    key = detect_key(chroma)

    if len(beat_frames) >= 2:
        beat_chroma = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
        chords = detect_chords(beat_chroma, beat_times)
    else:
        chords = []

    return {
        "bpm": round(bpm, 2),
        "key": key["key"],
        "key_confidence": key["confidence"],
        "duration": round(float(len(y) / sr), 3),
        "beat_count": int(len(beat_times)),
        "first_beat": round(float(beat_times[0]), 3) if len(beat_times) else 0.0,
        # 4/4 assumed — librosa does not track downbeats.
        "bars_estimated": int(len(beat_times) // 4),
        "beats": [round(float(t), 3) for t in beat_times],
        "chords": chords,
    }


def chord_chart(analysis: dict, max_lines: int = 40) -> str:
    """Render the chord list as a readable timeline."""
    chords = analysis.get("chords") or []
    if not chords:
        return "No chords detected (percussive or atonal source?)."
    lines = []
    for c in chords[:max_lines]:
        lines.append(
            f"  {c['start']:>7.2f}s - {c['end']:>7.2f}s   {c['chord']:<6} "
            f"(fit {c['confidence']:.2f})"
        )
    if len(chords) > max_lines:
        lines.append(f"  ... {len(chords) - max_lines} more")
    return "\n".join(lines)
