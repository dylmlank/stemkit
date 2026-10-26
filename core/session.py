"""Assembling everything into something you can actually open.

There is no .flp writer here on purpose. PyFLP parses and edits existing FL
projects; authoring a valid one from nothing is not a supported use of it, and a
half-working .flp writer would be worse than no writer at all. A type-1
multitrack MIDI plus a written session sheet gets to the same place reliably:
FL imports the MIDI as one named channel per instrument at the right tempo, and
the sheet carries the key, chords, and caveats the MIDI can't.
"""
from __future__ import annotations

from pathlib import Path

from core import analyze as analysis_mod
from core import midiout
from core.transcribe import QUALITY


def write_sheet(state: dict, project_dir: Path) -> Path:
    """Write session/SESSION.md summarising the whole analysis."""
    a = state.get("analysis") or {}
    stems = state.get("stems") or {}
    midi = state.get("midi") or {}
    notes_meta = state.get("notes") or {}

    if a:
        key_line = (f"- **Key:** {a.get('key', '?')} "
                    f"(fit {a.get('key_confidence', 0):.2f})")
    else:
        key_line = "- **Key:** not analysed"

    lines: list[str] = [
        f"# {state.get('title', state.get('slug', 'Untitled'))}",
        "",
        "## Session",
        "",
        f"- **Tempo:** {a.get('bpm', '?')} BPM",
        key_line,
        f"- **Length:** {a.get('duration', 0):.1f}s "
        f"(~{a.get('bars_estimated', 0)} bars at 4/4)",
        "",
    ]

    if stems:
        lines += ["## Stems", ""]
        for name in sorted(stems):
            note = QUALITY.get(name, "")
            count = (notes_meta.get(name) or {}).get("count")
            got = f"{count} notes" if count is not None else "not transcribed"
            lines.append(f"- **{name}** — {got}" + (f" · {note}" if note else ""))
        lines.append("")

    if midi:
        lines += [
            "## MIDI",
            "",
            "Drag `midi/all.mid` into FL Studio and choose *one channel per "
            "track* to lay out every instrument at once, or drop individual "
            "files onto specific channels:",
            "",
        ]
        for name in sorted(midi):
            lines.append(f"- `{midi[name]}` → {name}")
        lines += [
            "",
            "Set the project tempo to "
            f"**{a.get('bpm', '?')} BPM** before importing so the bars line up.",
            "",
        ]

    chords = a.get("chords") or []
    if chords:
        lines += ["## Chords", "", "```", analysis_mod.chord_chart(a, max_lines=80), "```", ""]

    lines += [
        "## Read this before trusting the MIDI",
        "",
        "- Bass, drums, and piano are the dependable parts. Check them, then "
        "build on them.",
        "- The `other` stem holds strings, brass, pads, and synths mixed "
        "together — no model separates those from each other, so its MIDI is a "
        "sketch at best. The audio stem is more useful than its transcription.",
        "- Reverb tails, slides, and bends confuse every transcriber. Expect "
        "stray short notes around them.",
        "- Bars assume 4/4 counted from the first detected beat. If the track "
        "has an intro pickup, everything shifts.",
        "",
    ]

    out = project_dir / "session" / "SESSION.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def build(workspace, project: str) -> dict:
    """Write the combined multitrack MIDI and the session sheet."""
    state = workspace.state(project)
    a = state.get("analysis") or {}
    bpm = float(a.get("bpm") or 120.0)
    base = workspace.path(project)

    # Reload every transcribed stem's notes from disk so the combined file
    # always reflects the latest transcription pass.
    import json
    stems_notes: dict[str, list[dict]] = {}
    for stem, meta in (state.get("notes") or {}).items():
        note_file = base / meta["notes_path"]
        if note_file.exists():
            stems_notes[stem] = json.loads(note_file.read_text(encoding="utf-8"))

    combined = None
    if stems_notes:
        combined = midiout.write_multitrack(
            stems_notes, base / "midi" / "all.mid",
            bpm=bpm, title=state.get("title", project),
        )
        state.setdefault("midi", {})["all"] = workspace.rel(project, combined)

    sheet = write_sheet(state, base)
    state["session"] = {
        "sheet": workspace.rel(project, sheet),
        "combined_midi": workspace.rel(project, combined) if combined else None,
    }
    workspace.save(project, state)

    return {
        "sheet": sheet,
        "combined_midi": combined,
        "stems_included": sorted(stems_notes),
        "bpm": bpm,
    }
