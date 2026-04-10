"""Repository .env loading — public config then secrets (override).

``configs/envs/.env`` — safe defaults & non-secret options (committed template:
``.env.example``).

``configs/secrets/.env`` — credentials (gitignored). Same KEY=VALUE format.
"""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Directory that contains ``configs/envs`` (repository root)."""
    override = os.environ.get("SMART_TRADER_ROOT")
    if override:
        return Path(override).resolve()
    here = Path(__file__).resolve()
    for p in (here.parent, *here.parents):
        if (p / "configs" / "envs").is_dir():
            return p
    raise RuntimeError("Could not locate repository root (configs/envs missing)")


def _parse_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    key, _, val = line.partition("=")
    key = key.strip()
    val = val.strip().strip("'\"")
    if "#" in val and not val.startswith(("[", "{")):
        val = val[: val.index("#")].strip()
    if not key:
        return None
    return key, val


def apply_env_file(path: Path, *, override: bool) -> None:
    """Parse a dotenv file into ``os.environ``."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_line(line)
        if not parsed:
            continue
        k, v = parsed
        if override:
            os.environ[k] = v
        else:
            os.environ.setdefault(k, v)


def load_repo_env_files() -> None:
    """Load ``configs/envs/.env`` then ``configs/secrets/.env`` (secrets win)."""
    root = repo_root()
    apply_env_file(root / "configs" / "envs" / ".env", override=False)
    apply_env_file(root / "configs" / "secrets" / ".env", override=True)
