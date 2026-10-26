"""Project storage for StemKit.

One project = one song. Everything derived from that song lives under a single
directory so a run can be resumed, inspected, or thrown away wholesale:

    projects/<slug>/
        project.json     state written by each stage (tempo, key, stems, ...)
        source/          the normalized source audio
        stems/           separated stems as WAV
        midi/            per-stem .mid plus the combined multitrack
        session/         session sheet + anything meant for the DAW

`project.json` is the single source of truth about what has already been done;
tools read it to decide whether a stage can run yet.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from config import PROJECTS_DIR

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """Filesystem-safe project id. 'Miles Davis - So What' -> 'miles-davis-so-what'."""
    slug = _SLUG_STRIP.sub("-", name.lower()).strip("-")
    return slug or "untitled"


class Workspace:
    """Locates projects on disk and reads/writes their state."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or PROJECTS_DIR)

    # -- paths ------------------------------------------------------------

    def path(self, project: str) -> Path:
        return self.root / slugify(project)

    def subdir(self, project: str, name: str) -> Path:
        """Return (and create) a subdirectory of the project."""
        d = self.path(project) / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def exists(self, project: str) -> bool:
        return (self.path(project) / "project.json").exists()

    def list_projects(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(
            p.name for p in self.root.iterdir()
            if p.is_dir() and (p / "project.json").exists()
        )

    # -- state ------------------------------------------------------------

    def create(self, project: str, title: str | None = None) -> dict:
        """Create the project directory tree and its initial state."""
        slug = slugify(project)
        base = self.root / slug
        for sub in ("source", "stems", "midi", "session"):
            (base / sub).mkdir(parents=True, exist_ok=True)
        state = {
            "slug": slug,
            "title": title or project,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": None,     # {"path", "duration", "sample_rate", "origin"}
            "analysis": None,   # {"bpm", "key", "beats", "chords", ...}
            "stems": {},        # stem name -> relative wav path
            "midi": {},         # stem name -> relative mid path
            "notes": {},        # stem name -> transcription summary
        }
        self.save(slug, state)
        return state

    def state(self, project: str) -> dict:
        """Load a project's state, or raise if it doesn't exist."""
        f = self.path(project) / "project.json"
        if not f.exists():
            raise FileNotFoundError(
                f"No project {slugify(project)!r}. Run import_audio first."
            )
        return json.loads(f.read_text(encoding="utf-8"))

    def save(self, project: str, state: dict) -> None:
        f = self.path(project) / "project.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def update(self, project: str, **fields) -> dict:
        """Merge top-level fields into the state and persist it."""
        state = self.state(project)
        state.update(fields)
        self.save(project, state)
        return state

    # -- convenience ------------------------------------------------------

    def source_wav(self, project: str) -> Path:
        """Path to the normalized source audio; raises if not imported yet."""
        state = self.state(project)
        src = state.get("source")
        if not src:
            raise RuntimeError(
                f"{slugify(project)!r} has no audio yet. Run import_audio first."
            )
        p = self.path(project) / src["path"]
        if not p.exists():
            raise FileNotFoundError(f"Source audio missing from disk: {p}")
        return p

    def rel(self, project: str, path: Path) -> str:
        """Store paths relative to the project dir so projects stay movable."""
        return str(Path(path).relative_to(self.path(project))).replace("\\", "/")
