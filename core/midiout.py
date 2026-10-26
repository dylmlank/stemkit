"""Writing notes out as MIDI, and reading them back as text.

Two outputs matter:

  per-stem .mid       drag one file onto one channel in FL's playlist
  combined .mid       a type-1 multitrack with named tracks and the detected
                      tempo; FL's "one channel per track" import lays the whole
                      song out at once, which is the closest thing to opening
                      a finished project

Drums are written to MIDI channel 10 (index 9) so FL routes them to a drum rack
instead of a pitched instrument.
"""
from __future__ import annotations

from pathlib import Path

from config import TICKS_PER_BEAT

PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

GM_DRUM_NAMES = {
    35: "Kick 2", 36: "Kick", 37: "Rim", 38: "Snare", 39: "Clap",
    40: "Snare 2", 41: "Low Tom", 42: "Hat (closed)", 43: "Low Tom 2",
    44: "Hat (pedal)", 45: "Mid Tom", 46: "Hat (open)", 47: "Mid Tom 2",
    48: "High Tom", 49: "Crash", 50: "High Tom 2", 51: "Ride",
    52: "China", 53: "Ride Bell", 55: "Splash", 57: "Crash 2", 59: "Ride 2",
}

DRUM_CHANNEL = 9


def pitch_name(midi_note: int) -> str:
    """60 -> 'C4'. Middle C is C4 here, matching FL's piano roll."""
    return f"{PITCHES[midi_note % 12]}{midi_note // 12 - 1}"


def _require_mido():
    try:
        import mido
        return mido
    except ImportError as e:
        raise RuntimeError("mido is required — pip install mido") from e


def _sec_to_ticks(seconds: float, bpm: float) -> int:
    return max(0, int(round(seconds * (TICKS_PER_BEAT * bpm / 60.0))))


def _build_track(mido, notes: list[dict], bpm: float, name: str,
                 channel: int, program: int | None = None):
    """One MIDI track from a note list, with correct delta timing."""
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=name[:32], time=0))
    if program is not None and channel != DRUM_CHANNEL:
        track.append(mido.Message("program_change", program=program,
                                  channel=channel, time=0))

    # Absolute-tick events first, then sort — note_off must land before a
    # note_on at the same tick or overlapping notes come out stuck.
    events: list[tuple[int, int, int, int]] = []  # (tick, order, pitch, velocity)
    for n in notes:
        on = _sec_to_ticks(n["start"], bpm)
        off = max(on + 1, _sec_to_ticks(n["end"], bpm))
        pitch = int(max(0, min(127, n["pitch"])))
        vel = int(max(1, min(127, n.get("velocity", 90))))
        events.append((off, 0, pitch, 0))     # order 0 = note_off
        events.append((on, 1, pitch, vel))    # order 1 = note_on
    events.sort(key=lambda e: (e[0], e[1]))

    prev = 0
    for tick, order, pitch, vel in events:
        delta = tick - prev
        prev = tick
        msg = "note_on" if order == 1 else "note_off"
        track.append(mido.Message(msg, note=pitch, velocity=vel,
                                  channel=channel, time=delta))
    track.append(mido.MetaMessage("end_of_track", time=1))
    return track


def write_midi(notes: list[dict], path: Path, bpm: float = 120.0,
               name: str = "stem", is_drums: bool = False) -> Path:
    """Write one stem's notes to a single-track .mid."""
    mido = _require_mido()
    path.parent.mkdir(parents=True, exist_ok=True)

    mid = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    meta.append(mido.MetaMessage("end_of_track", time=1))
    mid.tracks.append(meta)

    channel = DRUM_CHANNEL if is_drums else 0
    mid.tracks.append(_build_track(mido, notes, bpm, name, channel))
    mid.save(str(path))
    return path


def write_multitrack(stems: dict[str, list[dict]], path: Path,
                     bpm: float = 120.0, title: str = "StemKit") -> Path:
    """Write every stem into one type-1 file, one named track per stem."""
    mido = _require_mido()
    path.parent.mkdir(parents=True, exist_ok=True)

    mid = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name=title[:32], time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    meta.append(mido.MetaMessage("end_of_track", time=1))
    mid.tracks.append(meta)

    # Channel 9 is reserved for drums; pitched stems take the rest in order.
    pitched_channels = [c for c in range(16) if c != DRUM_CHANNEL]
    next_channel = 0
    for stem, notes in stems.items():
        if not notes:
            continue
        if stem == "drums":
            channel = DRUM_CHANNEL
        else:
            channel = pitched_channels[next_channel % len(pitched_channels)]
            next_channel += 1
        mid.tracks.append(_build_track(mido, notes, bpm, stem, channel))

    mid.save(str(path))
    return path


# -- reading notes back ----------------------------------------------------

def summarize(notes: list[dict], bpm: float = 120.0) -> dict:
    """Range, count, and density — the numbers worth seeing at a glance."""
    if not notes:
        return {"count": 0}
    pitches = [n["pitch"] for n in notes]
    span = max(n["end"] for n in notes) - min(n["start"] for n in notes)
    used = sorted({p % 12 for p in pitches})
    return {
        "count": len(notes),
        "lowest": pitch_name(min(pitches)),
        "highest": pitch_name(max(pitches)),
        "pitch_classes": [PITCHES[p] for p in used],
        "span_seconds": round(span, 2),
        "notes_per_bar": round(len(notes) / max(span * bpm / 240.0, 0.01), 1),
    }


def as_text(notes: list[dict], bpm: float = 120.0, is_drums: bool = False,
            start: float = 0.0, end: float | None = None,
            limit: int = 120) -> str:
    """Render notes as a readable list — bar position, name, and length."""
    if not notes:
        return "(no notes)"

    window = [n for n in notes
              if n["start"] >= start and (end is None or n["start"] < end)]
    if not window:
        return f"(no notes between {start}s and {end}s)"

    beats_per_second = bpm / 60.0
    lines = []
    for n in window[:limit]:
        beat = n["start"] * beats_per_second
        bar = int(beat // 4) + 1
        beat_in_bar = beat % 4 + 1
        label = (GM_DRUM_NAMES.get(n["pitch"], f"note {n['pitch']}")
                 if is_drums else pitch_name(n["pitch"]))
        length = n["end"] - n["start"]
        lines.append(
            f"  bar {bar:>3}.{beat_in_bar:<4.2f}  {label:<12} "
            f"{length:>5.2f}s  vel {n['velocity']:>3}"
        )
    if len(window) > limit:
        lines.append(f"  ... {len(window) - limit} more notes")
    return "\n".join(lines)
