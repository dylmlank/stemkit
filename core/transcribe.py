"""Audio-to-notes for each kind of stem.

Three different engines, because no single one is good at everything:

  pitched  Basic Pitch (Spotify). Excellent on monophonic material — bass and
           lead lines come out clean. Polyphonic material is approximate.
  piano    ByteDance's piano transcription model, if installed. Near-human on
           solo piano; falls back to Basic Pitch when unavailable.
  drums    Onset detection plus a spectral-band classifier written here. Kick,
           snare, and hats land reliably; ghost notes and toms do not. This is
           a heuristic, not a trained model — treat the output as a groove
           sketch to correct, not a transcription.

Every engine returns the same note shape so downstream code doesn't care which
one ran:  {"start": s, "end": s, "pitch": 0-127, "velocity": 1-127}
"""
from __future__ import annotations

from pathlib import Path


def _np():
    """numpy, imported on demand so `check_setup` still runs without it."""
    try:
        import numpy
        return numpy
    except ImportError as e:
        raise RuntimeError("numpy is required — pip install numpy") from e


# Frequency bounds that stop Basic Pitch from hallucinating octave errors.
# A bass line has no business being detected at 2 kHz.
STEM_RANGES = {
    "bass":   {"minimum_frequency": 30.0,  "maximum_frequency": 400.0},
    "vocals": {"minimum_frequency": 80.0,  "maximum_frequency": 1200.0},
    "guitar": {"minimum_frequency": 70.0,  "maximum_frequency": 1400.0},
    "piano":  {"minimum_frequency": 27.5,  "maximum_frequency": 4200.0},
    "other":  {"minimum_frequency": 55.0,  "maximum_frequency": 2000.0},
}

# General MIDI drum map — the notes FL's drum racks expect.
GM_KICK, GM_SNARE, GM_HAT_CLOSED, GM_HAT_OPEN, GM_CRASH, GM_TOM = 36, 38, 42, 46, 49, 45


def _velocity(amplitude: float) -> int:
    """Map a 0..1 confidence/amplitude onto a musical velocity range."""
    return max(1, min(127, int(round(40 + amplitude * 87))))


# -- pitched material ------------------------------------------------------

def transcribe_pitched(path: Path, stem: str = "other",
                       onset_threshold: float = 0.5,
                       frame_threshold: float = 0.3,
                       min_note_ms: float = 58.0) -> list[dict]:
    """Basic Pitch. Good on bass and single-line melodies, rough on chords."""
    try:
        from basic_pitch.inference import predict
    except ImportError as e:
        raise RuntimeError(
            "basic-pitch is not installed — pip install basic-pitch"
        ) from e

    bounds = STEM_RANGES.get(stem, STEM_RANGES["other"])
    _, _, note_events = predict(
        str(path),
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_ms,
        minimum_frequency=bounds["minimum_frequency"],
        maximum_frequency=bounds["maximum_frequency"],
        melodia_trick=True,
    )

    notes = []
    for event in note_events:
        # (start_s, end_s, midi_pitch, amplitude, pitch_bends)
        start, end, pitch, amplitude = event[0], event[1], event[2], event[3]
        if end <= start:
            continue
        notes.append({
            "start": round(float(start), 4),
            "end": round(float(end), 4),
            "pitch": int(pitch),
            "velocity": _velocity(float(amplitude)),
        })
    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    return notes


def transcribe_piano(path: Path, device: str = "auto") -> list[dict]:
    """ByteDance piano transcription. Falls back to Basic Pitch if absent."""
    try:
        from piano_transcription_inference import PianoTranscription, load_audio
        from piano_transcription_inference import sample_rate as pt_sample_rate
    except ImportError:
        return transcribe_pitched(path, stem="piano")

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    audio, _ = load_audio(str(path), sr=pt_sample_rate, mono=True)
    transcriptor = PianoTranscription(device=device)
    result = transcriptor.transcribe(audio, None)

    events = result.get("est_note_events") or []
    notes = []
    for e in events:
        start = float(e.get("onset_time", 0.0))
        end = float(e.get("offset_time", 0.0))
        if end <= start:
            continue
        notes.append({
            "start": round(start, 4),
            "end": round(end, 4),
            "pitch": int(e.get("midi_note", 60)),
            "velocity": max(1, min(127, int(e.get("velocity", 80)))),
        })
    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    return notes


# -- drums -----------------------------------------------------------------

def transcribe_drums(path: Path, sensitivity: float = 1.0) -> list[dict]:
    """Detect drum hits and guess which drum each one is.

    For every onset, the energy in three bands decides the label: sub/low for
    kick, mid for snare, high for cymbals. A long high-frequency decay reads as
    a crash rather than a hat. This gets the backbone of a groove right and
    misses nuance — expect to fix ghost notes and toms by hand.
    """
    try:
        import librosa
    except ImportError as e:
        raise RuntimeError("librosa is required — pip install librosa") from e

    np = _np()
    y, sr = librosa.load(str(path), sr=None, mono=True)
    if y.size == 0:
        return []

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # backtrack=False on purpose: backtracking walks each onset back to the
    # preceding energy minimum, which on a decaying kick lands mid-tail and
    # invents a second hit a beat later.
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, units="time",
        backtrack=False, delta=0.06 / max(sensitivity, 0.1),
    )
    if len(onsets) == 0:
        return []

    # Nothing on a real kit is struck twice inside 55 ms — collapse anything
    # closer than that into one hit.
    deduped = [float(onsets[0])]
    for t in onsets[1:]:
        if float(t) - deduped[-1] >= 0.055:
            deduped.append(float(t))
    onsets = deduped

    peak = float(np.max(np.abs(y))) or 1.0
    window = int(0.045 * sr)      # 45 ms is enough to characterise a hit
    tail = int(0.20 * sr)         # 200 ms to tell a crash from a closed hat

    # Pass 1: measure how much energy each onset has in each frequency band.
    #
    # Energy is averaged *per bin*, not summed. The bands cover very different
    # widths (150 Hz vs 1850 vs 18000), so summing lets the top band win on
    # anything broadband and every snare reads as a cymbal.
    measured: list[tuple[float, float, float, float, float]] = []
    for t in onsets:
        i = int(t * sr)
        seg = y[i:i + window]
        if seg.size < 32:
            continue
        spectrum = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
        freqs = np.fft.rfftfreq(seg.size, 1.0 / sr)

        def density(mask) -> float:
            band = spectrum[mask]
            return float(band.mean()) if band.size else 0.0

        low = density(freqs < 150)
        mid = density((freqs >= 150) & (freqs < 2000))
        high = density(freqs >= 4000)

        after = y[i + window:i + tail]
        decay_rms = float(np.sqrt(np.mean(after ** 2))) if after.size else 0.0
        onset_rms = float(np.sqrt(np.mean(seg ** 2))) or 1e-9
        measured.append((float(t), low, mid, high, decay_rms / onset_rms))

    if not measured:
        return []

    # Pass 2: calibrate against this track. A drum is "present" at an onset if
    # its band carries a real share of the loudest such hit in the song — so a
    # kick and a snare struck together produce both notes instead of only the
    # louder one. Single-label classification loses every backbeat that shares
    # a beat with the kick, which is most of them.
    lows = np.array([m[1] for m in measured])
    mids = np.array([m[2] for m in measured])
    highs = np.array([m[3] for m in measured])
    ref_low = float(np.percentile(lows, 90)) or 1e-9
    ref_mid = float(np.percentile(mids, 90)) or 1e-9
    ref_high = float(np.percentile(highs, 90)) or 1e-9

    PRESENT = 0.30      # share of the reference hit that counts as "played"

    notes = []
    for t, low, mid, high, decay_ratio in measured:
        i = int(t * sr)
        seg = y[i:i + window]
        amplitude = float(np.max(np.abs(seg))) / peak if seg.size else 0.5
        vel = _velocity(amplitude)

        hits: list[int] = []
        if low / ref_low > PRESENT:
            hits.append(GM_KICK)
        if mid / ref_mid > PRESENT:
            hits.append(GM_SNARE)
        # A cymbal must be brighter than it is mid-heavy. Snares are noisy too,
        # so band strength alone puts a phantom hat on every backbeat; a real
        # cymbal has almost nothing under 2 kHz holding it down.
        if high / ref_high > PRESENT and high > mid:
            # A hit still ringing 200 ms later is a crash, not a closed hat.
            hits.append(GM_CRASH if decay_ratio > 0.55 else (
                GM_HAT_OPEN if decay_ratio > 0.30 else GM_HAT_CLOSED))
        if not hits:
            # Something was struck — attribute it to the strongest band.
            strongest = max(
                ((low / ref_low, GM_KICK), (mid / ref_mid, GM_SNARE),
                 (high / ref_high, GM_HAT_CLOSED)), key=lambda p: p[0])
            hits.append(strongest[1])

        for pitch in hits:
            notes.append({
                "start": round(t, 4),
                "end": round(t + 0.08, 4),   # drums are triggers, not sustains
                "pitch": pitch,
                "velocity": vel,
            })

    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    return notes


# -- dispatch --------------------------------------------------------------

# Which engine suits which Demucs stem.
DEFAULT_METHOD = {
    "drums": "drums",
    "bass": "pitched",
    "piano": "piano",
    "guitar": "pitched",
    "vocals": "pitched",
    "other": "pitched",
}

# Honest expectations, surfaced in tool output so nobody is surprised.
QUALITY = {
    "drums": "good — kick/snare/hats solid, ghost notes and toms unreliable",
    "bass": "very good — monophonic lines transcribe cleanly",
    "piano": "excellent on solo piano, good in a mix",
    "guitar": "fair — chords and strums come out approximate",
    "vocals": "good for the melody line",
    "other": "rough — strings/pads/synths are mixed together in this stem",
}


def transcribe(path: Path, stem: str, method: str = "auto", **kwargs) -> list[dict]:
    """Run the right engine for a stem. `method`: auto|pitched|piano|drums."""
    if method == "auto":
        method = DEFAULT_METHOD.get(stem, "pitched")
    if method == "drums":
        return transcribe_drums(path, **kwargs)
    if method == "piano":
        return transcribe_piano(path, **kwargs)
    if method == "pitched":
        return transcribe_pitched(path, stem=stem, **kwargs)
    raise ValueError(f"Unknown method {method!r} (use auto|pitched|piano|drums)")
