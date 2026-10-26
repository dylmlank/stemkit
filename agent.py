r"""StemKit standalone agent — describe a song in your terminal and it pulls the
track apart into stems, notes, and MIDI for FL Studio.

    # in E:\stemkit\.env (or the environment):
    #   ANTHROPIC_API_KEY=sk-ant-...
    #   STEMKIT_MODEL=claude-opus-5              (optional; default is claude-opus-5)
    python agent.py                  # interactive
    python agent.py "pull apart D:\music\track.wav and show me the bassline"

Generation runs through the native Anthropic API (core.llm). The tool set comes
from core.tools.TOOLS; results are dispatched via core.tools.dispatch — one
definition, shared with mcp_server.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _load_env() -> None:
    env = Path(__file__).parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()

from core import Workspace, TOOLS, dispatch  # noqa: E402
from core.llm import make_client, run_loop  # noqa: E402

SYSTEM = (
    "You are StemKit, a music analysis assistant that pulls songs apart into "
    "stems, notes, and MIDI the user can open in FL Studio. The pipeline is "
    "ordered: import_audio, then analyze_audio, then separate_stems, then "
    "transcribe_all, then build_session. Call project_status when you need to "
    "know how far along a project is rather than guessing.\n\n"
    "Separation and transcription are slow — a few minutes each on a GPU, much "
    "longer on CPU. Say so before starting one, and don't kick off a stage the "
    "user didn't ask for.\n\n"
    "Be straight about quality. Bass, drums, and piano transcribe well; the "
    "'other' stem is strings, brass, pads, and synths mixed together and its "
    "notes are a rough sketch. When the user asks what was played, call "
    "get_notes and show them the actual notes rather than describing them. "
    "Keep replies short; act with the tools rather than narrating what you "
    "would do."
)


def main() -> int:
    ws = Workspace()
    client = make_client()
    history: list = []

    def drive(text: str) -> None:
        history.append({"role": "user", "content": text})
        run_loop(client, TOOLS, lambda n, a: dispatch(ws, n, a),
                 SYSTEM, history, print, max_tokens=16000)

    if len(sys.argv) > 1:
        drive(" ".join(sys.argv[1:]))
        return 0

    print("StemKit agent — point it at a song (Ctrl+C to quit).")
    try:
        while True:
            text = input("\n> ").strip()
            if text:
                drive(text)
    except (KeyboardInterrupt, EOFError):
        print("\nbye")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
