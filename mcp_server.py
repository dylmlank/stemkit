"""StemKit MCP server — exposes song-to-MIDI extraction as MCP tools so Claude
Desktop can pull a track apart in plain language.

Add to claude_desktop_config.json:

    "stemkit": {
      "command": "C:\\\\Users\\\\dylan\\\\.local\\\\bin\\\\uv.exe",
      "args": ["run", "--python", "3.12", "--with", "mcp",
               "python", "E:\\\\stemkit\\\\mcp_server.py"]
    }

Each tool here is a thin typed wrapper over a core.tools handler — the behaviour
lives in core/tools.py, shared with the standalone agent.
"""
from __future__ import annotations

try:
    # mcp >= 2.0 renamed FastMCP to MCPServer; the .tool()/.run() API is the same.
    from mcp.server import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from core import Workspace
import core.tools as T

mcp = _Server("stemkit")
ws = Workspace()


@mcp.tool()
def check_setup() -> str:
    """Check which dependencies are installed (ffmpeg, demucs, basic-pitch) and whether a GPU is available."""
    return T.check_setup(ws)


@mcp.tool()
def list_projects() -> str:
    """List every project with its tempo, key, and pipeline progress."""
    return T.list_projects(ws)


@mcp.tool()
def project_status(project: str) -> str:
    """Show what has been done to one project and what the next step is."""
    return T.project_status(ws, project)


@mcp.tool()
def import_audio(source: str, project: str = "") -> str:
    """Start a project from a local audio file or a URL. Spotify is not supported (DRM)."""
    return T.import_audio(ws, source, project)


@mcp.tool()
def set_tempo(project: str, bpm: float) -> str:
    """Override the detected tempo — use when detection lands on half or double time."""
    return T.set_tempo(ws, project, bpm)


@mcp.tool()
def analyze_audio(project: str) -> str:
    """Detect tempo, key, beat grid, and chord progression."""
    return T.analyze_audio(ws, project)


@mcp.tool()
def detect_chords(project: str) -> str:
    """Show the chord chart with timestamps."""
    return T.detect_chords(ws, project)


@mcp.tool()
def separate_stems(project: str, model: str = "") -> str:
    """Split the source into stems with Demucs (drums, bass, vocals, guitar, piano, other)."""
    return T.separate_stems(ws, project, model)


@mcp.tool()
def list_stems(project: str) -> str:
    """List separated stems, their sizes, and whether each is transcribed."""
    return T.list_stems(ws, project)


@mcp.tool()
def transcribe_stem(project: str, stem: str, method: str = "auto") -> str:
    """Turn one stem into notes and a .mid. method: auto|pitched|piano|drums."""
    return T.transcribe_stem(ws, project, stem, method)


@mcp.tool()
def transcribe_all(project: str, skip_vocals: bool = True) -> str:
    """Transcribe every stem using the engine that suits each one."""
    return T.transcribe_all(ws, project, skip_vocals)


@mcp.tool()
def get_notes(project: str, stem: str, start_bar: int = 1, end_bar: int = 0,
              limit: int = 120) -> str:
    """Read a stem's notes as text — bar position, note name, length, velocity."""
    return T.get_notes(ws, project, stem, start_bar, end_bar, limit)


@mcp.tool()
def export_midi(project: str, stem: str = "") -> str:
    """Report the path to the project's .mid files."""
    return T.export_midi(ws, project, stem)


@mcp.tool()
def build_session(project: str) -> str:
    """Write the combined multitrack MIDI and the FL Studio session sheet."""
    return T.build_session(ws, project)


@mcp.tool()
def open_project_folder(project: str) -> str:
    """Open the project's folder in the OS file browser."""
    return T.open_project_folder(ws, project)


if __name__ == "__main__":
    mcp.run()
