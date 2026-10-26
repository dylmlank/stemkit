# StemKit

Pull a song apart into stems, notes, and MIDI you can open in FL Studio.

Point it at an audio file. It separates the instruments, works out the tempo,
key, and chord progression, transcribes each stem into notes, and writes a
multitrack MIDI file plus a session sheet. Sixth kit alongside studiokit,
renderkit, editkit, robloxkit, and resolvekit — same layout, same MCP + agent
pair.

```
import_audio → analyze_audio → separate_stems → transcribe_all → build_session
```

## What you actually get

| Part | Engine | How good it is |
|---|---|---|
| Tempo / key | librosa | Reliable on steady-tempo tonal music |
| Chords | chroma templates | Good as labels (Am7, F, C); extensions are the weak part |
| Bass | Basic Pitch | Very good — monophonic lines come out clean |
| Drums | onsets + spectral bands | Kick, snare, hats solid; ghost notes and toms are not |
| Piano | ByteDance model (optional) | Excellent solo, good in a mix |
| Guitar | Basic Pitch | Fair — chords and strums are approximate |
| Vocals | Basic Pitch | Good for the melody line |
| `other` | Basic Pitch | Rough — see below |

**The `other` stem is the honest limit.** Demucs isolates drums, bass, vocals,
guitar, and piano. Everything else — strings, brass, pads, organ, synths — lands
in `other` as one mixed signal, and no open model splits those from each other.
Its audio is useful; its MIDI is a sketch.

What you should expect overall: a very good working skeleton — correct tempo,
key, chord progression, an accurate bassline, a usable drum pattern, and
piano/lead notes you can trust — plus approximations for anything buried in a
dense mix. A fast head start, not a photocopy.

## Setup

```bash
install.bat
```

That makes a venv, installs everything, and prints a dependency report. It
takes a while — torch alone is a ~2 GB download.

`ffmpeg` and `ffprobe` must be on PATH separately; they aren't pip packages.

**If you have an NVIDIA GPU**, uncomment the CUDA torch line near the top of
`install.bat` before running it. Separation drops from roughly 15 minutes a
song to about one.

Optional extras:

```bash
venv\Scripts\pip install piano_transcription_inference   # much better piano
venv\Scripts\pip install yt-dlp                          # import from a URL
```

### Why the install isn't just `pip install -r requirements.txt`

Two things bite on Windows + Python 3.12, both handled by `install.bat`:

- **basic-pitch** declares a hard dependency on `tensorflow<2.15.1`, and
  TensorFlow has no Windows wheel for 3.12. Plain `pip install basic-pitch`
  can't resolve it and dies trying to compile an ancient numpy from source.
  basic-pitch ships four interchangeable backends, so it installs with
  `--no-deps` and runs on `onnxruntime` instead. Verified working.
- **setuptools must be < 81.** `resampy` still imports `pkg_resources`, which
  setuptools 81 removed.

Check the result any time with:

```bash
venv\Scripts\python -c "from core import Workspace; import core.tools as T; print(T.check_setup(Workspace()))"
```

## Use it

**As an MCP server** — add to `claude_desktop_config.json`:

```json
"stemkit": {
  "command": "C:\\Users\\dylan\\.local\\bin\\uv.exe",
  "args": ["run", "--python", "3.12", "--with", "mcp",
           "python", "E:\\stemkit\\mcp_server.py"]
}
```

**As a terminal agent:**

```bash
python agent.py "pull apart D:\music\track.wav and show me the bassline"
```

**Directly, without a model in the loop:**

```python
from core import Workspace
import core.tools as T

ws = Workspace()
print(T.import_audio(ws, r"D:\music\track.wav", "my-song"))
print(T.analyze_audio(ws, "my-song"))
print(T.separate_stems(ws, "my-song"))
print(T.transcribe_all(ws, "my-song"))
print(T.build_session(ws, "my-song"))
print(T.get_notes(ws, "my-song", "bass", start_bar=1, end_bar=9))
```

## Getting it into FL Studio

`build_session` writes `midi/all.mid` — a type-1 multitrack with one named
track per instrument, the detected tempo, and drums on channel 10.

1. Set the FL project tempo to the BPM in the session sheet **first**.
2. Drag `all.mid` into the playlist.
3. Choose **one channel per track** when FL asks.

Individual `midi/<stem>.mid` files work too if you'd rather place instruments
one at a time.

**There is no `.flp` writer, deliberately.** PyFLP parses and edits existing FL
projects; authoring a valid one from scratch isn't something it supports, and a
half-working writer would be worse than none. Multitrack MIDI import gets to the
same place reliably.

## Project layout on disk

```
projects/<slug>/
    project.json     what's been done — tempo, key, stems, note counts
    source/          the normalized 44.1k stereo WAV
    stems/           separated stems
    midi/            per-stem .mid, the combined all.mid, raw notes as JSON
    session/         SESSION.md — tempo, key, chord chart, caveats
```

Everything is re-runnable. Re-transcribing a stem overwrites just that stem;
`build_session` always rebuilds the combined file from whatever is current.

## Where the audio comes from

Local files you own are the intended input. Spotify is a dead end — the desktop
and web players are DRM-protected and the API never gave more than 30-second
previews. `import_audio` accepts a URL and shells out to `yt-dlp` if it's
installed; whether a given site's terms permit that is your call, and StemKit
does nothing to work around DRM.

## Tools

`check_setup` · `list_projects` · `project_status` · `import_audio` ·
`set_tempo` · `analyze_audio` · `detect_chords` · `separate_stems` ·
`list_stems` · `transcribe_stem` · `transcribe_all` · `get_notes` ·
`export_midi` · `build_session` · `open_project_folder`
