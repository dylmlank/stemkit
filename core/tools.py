"""StemKit's capability layer — defined once, exposed through both the MCP
server (mcp_server.py) and the standalone Claude agent (agent.py).

Each entry in TOOLS has a JSON-schema spec (for Claude's tool-use API) and a
handler `fn(workspace, **kwargs) -> str`. dispatch() runs a handler by name.
The handlers are the single source of truth for what StemKit can do.

The pipeline is ordered, and each stage records its result in project.json:

    import_audio -> analyze_audio -> separate_stems -> transcribe_stem(s)
                                                    -> build_session
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from config import DEMUCS_MODEL, PROJECTS_DIR
from core import analyze as analysis_mod
from core import audio as audio_mod
from core import midiout
from core import separate as separate_mod
from core import session as session_mod
from core import transcribe as transcribe_mod
from core.workspace import Workspace, slugify


# -- setup -----------------------------------------------------------------

def check_setup(ws: Workspace) -> str:
    """Report which pieces are installed. Nothing here is fatal on its own."""
    rows: list[tuple[str, bool, str]] = []

    rows.append(("ffmpeg", audio_mod.have(os.environ.get("STEMKIT_FFMPEG", "ffmpeg")),
                 "required — audio import"))
    rows.append(("ffprobe", audio_mod.have(os.environ.get("STEMKIT_FFPROBE", "ffprobe")),
                 "required — reads duration/sample rate"))

    def importable(mod: str) -> bool:
        try:
            __import__(mod)
            return True
        except ImportError:
            return False

    rows.append(("librosa", importable("librosa"), "required — tempo/key/chords/drums"))
    rows.append(("mido", importable("mido"), "required — MIDI export"))
    rows.append(("demucs", importable("demucs"), "required — stem separation"))
    rows.append(("basic-pitch", importable("basic_pitch"), "required — pitched transcription"))
    rows.append(("piano_transcription_inference",
                 importable("piano_transcription_inference"),
                 "optional — much better piano, falls back to basic-pitch"))
    rows.append(("yt-dlp", audio_mod.have(os.environ.get("STEMKIT_YTDLP", "yt-dlp")),
                 "optional — only for URL imports"))

    device = separate_mod.resolve_device()
    lines = [f"Device for separation: {device}"]
    if device == "cpu":
        lines.append("  (CPU works but a 4-minute song takes ~10-20 min to separate)")
    lines.append("")
    for name, ok, note in rows:
        lines.append(f"  [{'ok' if ok else '--'}] {name:<32} {note}")

    missing = [n for n, ok, note in rows if not ok and note.startswith("required")]
    lines.append("")
    lines.append(f"Projects directory: {PROJECTS_DIR}")
    if missing:
        lines.append(f"Missing required: {', '.join(missing)} — see README.md")
    else:
        lines.append("All required pieces present.")
    return "\n".join(lines)


# -- projects --------------------------------------------------------------

def list_projects(ws: Workspace) -> str:
    projects = ws.list_projects()
    if not projects:
        return "No projects yet. Run import_audio to start one."
    lines = []
    for slug in projects:
        try:
            s = ws.state(slug)
        except OSError:
            continue
        a = s.get("analysis") or {}
        bits = [slug]
        if a.get("bpm"):
            bits.append(f"{a['bpm']} BPM {a.get('key', '')}".strip())
        if s.get("stems"):
            bits.append(f"{len(s['stems'])} stems")
        if s.get("notes"):
            bits.append(f"{len(s['notes'])} transcribed")
        lines.append("  " + " | ".join(bits))
    return f"{len(lines)} project(s):\n" + "\n".join(lines)


def project_status(ws: Workspace, project: str) -> str:
    """What has been done and what the next step is."""
    s = ws.state(project)
    a = s.get("analysis") or {}
    src = s.get("source") or {}

    lines = [f"{s.get('title', project)}  ({s['slug']})", ""]
    lines.append(f"  source      {'yes — %.1fs' % src['duration'] if src else 'not imported'}")
    lines.append(f"  analysis    {('%s BPM, %s' % (a.get('bpm'), a.get('key'))) if a else 'not run'}")
    lines.append(f"  stems       {', '.join(sorted(s['stems'])) if s.get('stems') else 'not separated'}")

    notes = s.get("notes") or {}
    if notes:
        parts = [f"{k} ({v['count']})" for k, v in sorted(notes.items())]
        lines.append(f"  transcribed {', '.join(parts)}")
    else:
        lines.append("  transcribed none")
    lines.append(f"  session     {'built' if s.get('session') else 'not built'}")

    # Point at the next thing worth doing.
    if not src:
        nxt = "import_audio"
    elif not a:
        nxt = "analyze_audio"
    elif not s.get("stems"):
        nxt = "separate_stems"
    elif not notes:
        nxt = "transcribe_all"
    elif not s.get("session"):
        nxt = "build_session"
    else:
        nxt = "done — open the session sheet, or re-transcribe a stem to tune it"
    lines += ["", f"  next: {nxt}"]
    return "\n".join(lines)


def import_audio(ws: Workspace, source: str, project: str = "") -> str:
    """Bring a local file or a URL into a new project as 44.1k stereo WAV."""
    is_url = source.startswith(("http://", "https://"))

    if not project:
        project = Path(source).stem if not is_url else "download"
    slug = slugify(project)

    if ws.exists(slug):
        return (f"Project {slug!r} already exists. Pick another name, or delete "
                f"{ws.path(slug)} first.")

    ws.create(slug, title=project)
    base = ws.path(slug)

    try:
        if is_url:
            raw = audio_mod.download(source, base / "source")
            origin = source
        else:
            raw = Path(source).expanduser()
            if not raw.exists():
                raise FileNotFoundError(f"No such file: {raw}")
            origin = str(raw)

        wav = audio_mod.to_wav(raw, base / "source" / "audio.wav")
        info = audio_mod.probe(wav)

        # The original download is redundant once transcoded.
        if is_url and raw.exists() and raw != wav:
            raw.unlink(missing_ok=True)
    except Exception as e:
        # Don't leave a half-built project behind — it has no audio but its
        # existence would block retrying under the same name.
        import shutil
        shutil.rmtree(base, ignore_errors=True)
        return f"Import failed: {e}"

    ws.update(slug, source={
        "path": ws.rel(slug, wav),
        "duration": info["duration"],
        "sample_rate": info["sample_rate"],
        "channels": info["channels"],
        "origin": origin,
    })
    return (f"Imported into project {slug!r}: {info['duration']:.1f}s, "
            f"{info['sample_rate']} Hz, {info['channels']}ch.\n"
            f"Next: analyze_audio, then separate_stems.")


def set_tempo(ws: Workspace, project: str, bpm: float) -> str:
    """Override detected tempo. Use when analysis lands on half or double time."""
    s = ws.state(project)
    a = s.get("analysis") or {}
    old = a.get("bpm")
    a["bpm"] = round(float(bpm), 2)
    a["bpm_source"] = "manual"
    ws.update(project, analysis=a)
    return (f"Tempo set to {bpm} BPM (was {old}). Re-run build_session so the "
            f"exported MIDI uses it.")


# -- analysis --------------------------------------------------------------

def analyze_audio(ws: Workspace, project: str) -> str:
    """Detect tempo, key, beat grid, and the chord progression."""
    try:
        src = ws.source_wav(project)
    except Exception as e:
        return str(e)
    try:
        result = analysis_mod.analyze(src)
    except Exception as e:
        return f"Analysis failed: {e}"

    # Keep a manual tempo override if one was set.
    previous = ws.state(project).get("analysis") or {}
    if previous.get("bpm_source") == "manual":
        result["bpm"] = previous["bpm"]
        result["bpm_source"] = "manual"

    ws.update(project, analysis=result)
    chords = result.get("chords") or []
    return (
        f"{result['bpm']} BPM | key {result['key']} "
        f"(fit {result['key_confidence']:.2f}) | {result['duration']:.1f}s | "
        f"~{result['bars_estimated']} bars | {len(chords)} chord changes\n"
        f"If the BPM looks like half or double the real tempo, fix it with "
        f"set_tempo before exporting."
    )


def detect_chords(ws: Workspace, project: str) -> str:
    """Show the chord chart. Runs analysis first if it hasn't been run."""
    s = ws.state(project)
    a = s.get("analysis")
    if not a:
        msg = analyze_audio(ws, project)
        if "failed" in msg.lower():
            return msg
        a = ws.state(project)["analysis"]
    return (f"Key {a.get('key')} at {a.get('bpm')} BPM\n\n"
            + analysis_mod.chord_chart(a))


# -- separation ------------------------------------------------------------

def separate_stems(ws: Workspace, project: str, model: str = "") -> str:
    """Split the source into stems with Demucs. Slow on CPU, fast on a GPU."""
    try:
        src = ws.source_wav(project)
    except Exception as e:
        return str(e)

    model = model or DEMUCS_MODEL
    out = ws.subdir(project, "stems")
    try:
        stems = separate_mod.separate(src, out, model=model)
    except Exception as e:
        return f"Separation failed: {e}"

    rel = {name: ws.rel(project, p) for name, p in stems.items()}
    ws.update(project, stems=rel, separation_model=model)

    lines = [f"Separated into {len(stems)} stems with {model}:"]
    for name in sorted(stems):
        note = transcribe_mod.QUALITY.get(name, "")
        lines.append(f"  {name:<8} {note}")
    lines.append("\nNext: transcribe_all, or transcribe_stem for one at a time.")
    return "\n".join(lines)


def list_stems(ws: Workspace, project: str) -> str:
    s = ws.state(project)
    stems = s.get("stems") or {}
    if not stems:
        return "No stems yet — run separate_stems."
    base = ws.path(project)
    lines = [f"{len(stems)} stems:"]
    for name in sorted(stems):
        p = base / stems[name]
        size = p.stat().st_size // 1024 if p.exists() else 0
        done = (s.get("notes") or {}).get(name)
        status = f"{done['count']} notes" if done else "not transcribed"
        lines.append(f"  {name:<8} {size:>7} KB  {status}")
    return "\n".join(lines)


# -- transcription ---------------------------------------------------------

def _store_notes(ws: Workspace, project: str, stem: str, notes: list[dict],
                 method: str, bpm: float) -> dict:
    """Persist a stem's notes as JSON and write its per-stem .mid."""
    base = ws.path(project)
    note_file = base / "midi" / f"{stem}.notes.json"
    note_file.parent.mkdir(parents=True, exist_ok=True)
    note_file.write_text(json.dumps(notes), encoding="utf-8")

    mid = midiout.write_midi(
        notes, base / "midi" / f"{stem}.mid",
        bpm=bpm, name=stem, is_drums=(stem == "drums"),
    )

    state = ws.state(project)
    state.setdefault("notes", {})[stem] = {
        "count": len(notes),
        "method": method,
        "notes_path": ws.rel(project, note_file),
        "summary": midiout.summarize(notes, bpm),
    }
    state.setdefault("midi", {})[stem] = ws.rel(project, mid)
    ws.save(project, state)
    return state["notes"][stem]


def transcribe_stem(ws: Workspace, project: str, stem: str,
                    method: str = "auto") -> str:
    """Turn one stem into notes and a .mid. method: auto|pitched|piano|drums."""
    s = ws.state(project)
    stems = s.get("stems") or {}
    if stem not in stems:
        return (f"No stem {stem!r}. Available: "
                f"{', '.join(sorted(stems)) or 'none — run separate_stems'}")

    bpm = float((s.get("analysis") or {}).get("bpm") or 120.0)
    path = ws.path(project) / stems[stem]
    try:
        notes = transcribe_mod.transcribe(path, stem, method=method)
    except Exception as e:
        return f"Transcription of {stem} failed: {e}"

    used = method if method != "auto" else transcribe_mod.DEFAULT_METHOD.get(stem, "pitched")
    meta = _store_notes(ws, project, stem, notes, used, bpm)
    summary = meta["summary"]

    if not notes:
        return (f"{stem}: no notes detected. If the stem has audible content, "
                f"try transcribe_stem with method='pitched' and a lower threshold.")

    detail = (f"range {summary['lowest']}-{summary['highest']}, "
              f"{summary['notes_per_bar']} notes/bar"
              if stem != "drums" else
              f"{summary['notes_per_bar']} hits/bar")
    return (f"{stem}: {len(notes)} notes via {used} ({detail})\n"
            f"  quality: {transcribe_mod.QUALITY.get(stem, 'unknown')}\n"
            f"  wrote midi/{stem}.mid")


def transcribe_all(ws: Workspace, project: str, skip_vocals: bool = True) -> str:
    """Transcribe every separated stem, each with the engine that suits it."""
    s = ws.state(project)
    stems = s.get("stems") or {}
    if not stems:
        return "No stems yet — run separate_stems."

    order = ["drums", "bass", "piano", "guitar", "vocals", "other"]
    targets = [x for x in order if x in stems] + \
              [x for x in sorted(stems) if x not in order]
    if skip_vocals:
        targets = [t for t in targets if t != "vocals"]

    lines = []
    for stem in targets:
        lines.append(transcribe_stem(ws, project, stem).splitlines()[0])
    lines.append("\nNext: build_session to write the combined MIDI and sheet.")
    return "\n".join(lines)


def get_notes(ws: Workspace, project: str, stem: str,
              start_bar: int = 1, end_bar: int = 0, limit: int = 120) -> str:
    """Read a stem's transcribed notes as text, by bar range."""
    s = ws.state(project)
    meta = (s.get("notes") or {}).get(stem)
    if not meta:
        return f"{stem!r} has not been transcribed. Run transcribe_stem first."

    notes = json.loads((ws.path(project) / meta["notes_path"]).read_text(encoding="utf-8"))
    bpm = float((s.get("analysis") or {}).get("bpm") or 120.0)
    seconds_per_bar = 240.0 / bpm

    start = max(0, (start_bar - 1)) * seconds_per_bar
    end = end_bar * seconds_per_bar if end_bar else None

    header = f"{stem} — {meta['count']} notes total, {bpm} BPM"
    if stem != "drums":
        sm = meta["summary"]
        header += (f"\nrange {sm['lowest']}-{sm['highest']}, "
                   f"pitch classes used: {', '.join(sm['pitch_classes'])}")
    body = midiout.as_text(notes, bpm=bpm, is_drums=(stem == "drums"),
                           start=start, end=end, limit=limit)
    return f"{header}\n\n{body}"


# -- output ----------------------------------------------------------------

def export_midi(ws: Workspace, project: str, stem: str = "") -> str:
    """Report where the .mid files are (they are written on transcription)."""
    s = ws.state(project)
    midi = s.get("midi") or {}
    if not midi:
        return "No MIDI yet — transcribe a stem first."
    base = ws.path(project)
    if stem:
        if stem not in midi:
            return f"No MIDI for {stem!r}. Have: {', '.join(sorted(midi))}"
        return str(base / midi[stem])
    lines = [f"MIDI in {base / 'midi'}:"]
    for name in sorted(midi):
        lines.append(f"  {name:<8} {midi[name]}")
    return "\n".join(lines)


def build_session(ws: Workspace, project: str) -> str:
    """Write the combined multitrack MIDI and the session sheet."""
    s = ws.state(project)
    if not (s.get("notes") or {}):
        return "Nothing transcribed yet — run transcribe_all first."
    try:
        result = session_mod.build(ws, project)
    except Exception as e:
        return f"Session build failed: {e}"

    lines = [
        f"Session built at {result['bpm']} BPM.",
        f"  sheet         {result['sheet']}",
    ]
    if result["combined_midi"]:
        lines.append(f"  combined midi {result['combined_midi']}")
        lines.append(f"  tracks        {', '.join(result['stems_included'])}")
    lines += [
        "",
        f"In FL Studio: set the tempo to {result['bpm']}, then drag all.mid in "
        f"and pick 'one channel per track'.",
    ]
    return "\n".join(lines)


def open_project_folder(ws: Workspace, project: str) -> str:
    """Open the project directory in the OS file browser."""
    path = ws.path(project)
    if not path.exists():
        return f"No such project: {path}"
    try:
        if sys.platform == "win32":
            os.startfile(path)  # noqa: S606 — intended: reveal in Explorer
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as e:
        return f"{path}  (couldn't open a file browser: {e})"
    return f"Opened {path}"


# name -> handler
_HANDLERS = {
    "check_setup": check_setup,
    "list_projects": list_projects,
    "project_status": project_status,
    "import_audio": import_audio,
    "set_tempo": set_tempo,
    "analyze_audio": analyze_audio,
    "detect_chords": detect_chords,
    "separate_stems": separate_stems,
    "list_stems": list_stems,
    "transcribe_stem": transcribe_stem,
    "transcribe_all": transcribe_all,
    "get_notes": get_notes,
    "export_midi": export_midi,
    "build_session": build_session,
    "open_project_folder": open_project_folder,
}

# JSON-schema specs for Claude's tool-use API. Kept next to the handlers so the
# two front-ends share one definition.
TOOLS = [
    {"name": "check_setup",
     "description": "Check which StemKit dependencies are installed (ffmpeg, demucs, basic-pitch, librosa) and whether a GPU is available.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "list_projects",
     "description": "List every StemKit project with its tempo, key, and how far through the pipeline it is.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "project_status",
     "description": "Show what has been done to one project and what the next step is.",
     "input_schema": {"type": "object", "properties": {"project": {"type": "string"}},
                      "required": ["project"]}},
    {"name": "import_audio",
     "description": "Start a project from a local audio file path or a URL. Transcodes to 44.1kHz stereo WAV. Spotify is not supported (DRM).",
     "input_schema": {"type": "object", "properties": {
         "source": {"type": "string", "description": "Local file path or URL"},
         "project": {"type": "string", "description": "Project name; defaults to the filename"}},
         "required": ["source"]}},
    {"name": "set_tempo",
     "description": "Override the detected tempo. Use when detection lands on half or double the real BPM.",
     "input_schema": {"type": "object", "properties": {
         "project": {"type": "string"}, "bpm": {"type": "number"}},
         "required": ["project", "bpm"]}},
    {"name": "analyze_audio",
     "description": "Detect tempo, key, beat grid, and chord progression for a project's source audio.",
     "input_schema": {"type": "object", "properties": {"project": {"type": "string"}},
                      "required": ["project"]}},
    {"name": "detect_chords",
     "description": "Show the chord chart with timestamps. Runs analysis first if needed.",
     "input_schema": {"type": "object", "properties": {"project": {"type": "string"}},
                      "required": ["project"]}},
    {"name": "separate_stems",
     "description": "Split the source into stems with Demucs (drums, bass, vocals, guitar, piano, other). Slow on CPU; minutes on a GPU.",
     "input_schema": {"type": "object", "properties": {
         "project": {"type": "string"},
         "model": {"type": "string", "description": "Demucs model; default htdemucs_6s"}},
         "required": ["project"]}},
    {"name": "list_stems",
     "description": "List a project's separated stems, their file sizes, and whether each is transcribed.",
     "input_schema": {"type": "object", "properties": {"project": {"type": "string"}},
                      "required": ["project"]}},
    {"name": "transcribe_stem",
     "description": "Turn one stem into notes and a .mid file. method: auto|pitched|piano|drums.",
     "input_schema": {"type": "object", "properties": {
         "project": {"type": "string"}, "stem": {"type": "string"},
         "method": {"type": "string"}},
         "required": ["project", "stem"]}},
    {"name": "transcribe_all",
     "description": "Transcribe every separated stem using the engine that suits each one.",
     "input_schema": {"type": "object", "properties": {
         "project": {"type": "string"},
         "skip_vocals": {"type": "boolean", "description": "Default true"}},
         "required": ["project"]}},
    {"name": "get_notes",
     "description": "Read a stem's transcribed notes as text — bar position, note name, length, velocity. Use this to SEE what was played.",
     "input_schema": {"type": "object", "properties": {
         "project": {"type": "string"}, "stem": {"type": "string"},
         "start_bar": {"type": "integer"}, "end_bar": {"type": "integer"},
         "limit": {"type": "integer"}},
         "required": ["project", "stem"]}},
    {"name": "export_midi",
     "description": "Report the path to a project's .mid files (written automatically on transcription).",
     "input_schema": {"type": "object", "properties": {
         "project": {"type": "string"}, "stem": {"type": "string"}},
         "required": ["project"]}},
    {"name": "build_session",
     "description": "Write the combined multitrack MIDI (one named track per instrument, correct tempo) and the session sheet for FL Studio.",
     "input_schema": {"type": "object", "properties": {"project": {"type": "string"}},
                      "required": ["project"]}},
    {"name": "open_project_folder",
     "description": "Open the project's folder in the OS file browser.",
     "input_schema": {"type": "object", "properties": {"project": {"type": "string"}},
                      "required": ["project"]}},
]


def dispatch(ws: Workspace, name: str, arguments: dict) -> str:
    """Run a tool by name with the given arguments. Returns a text result."""
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"Unknown tool: {name}"
    try:
        return handler(ws, **(arguments or {}))
    except Exception as e:  # surface tool errors back to the model
        return f"Error in {name}: {e}"
