"""Pristine process-environment guard for the forge LLM layer.

Several sibling CLI modules call ``dotenv.load_dotenv()`` at import time, which
injects ``.env`` values into ``os.environ`` for the whole ``swe-forge`` process.
That would repopulate forge credentials even after an explicit ``env -u`` and
defeat fail-fast on missing credentials.

This module snapshots the real (exec-time) environment when it is imported.
Because the entrypoint imports it before those sibling modules, the snapshot is
clean. :func:`scrub_injected_forge_env` then removes any forge credential var
that was *not* present at startup but appears later (i.e. was injected by an
implicit ``.env`` load), so credential resolution reflects only what was
actually exported. Vars used by other commands are left untouched.
"""

from __future__ import annotations

import os

_PRISTINE_ENV: dict[str, str] = dict(os.environ)

_FORGE_ENV_KEYS = (
    "TEACHER_LLM_BASE_URL",
    "TEACHER_LLM_API_KEY",
    "TEACHER_LLM_MODEL",
    "TEACHER_LLM_PROVIDER",
    "PANEL_LLM_BASE_URL",
    "PANEL_LLM_API_KEY",
)


def scrub_injected_forge_env() -> None:
    """Drop forge credential vars injected into ``os.environ`` after startup."""
    for key in _FORGE_ENV_KEYS:
        if key not in _PRISTINE_ENV and os.environ.get(key) is not None:
            del os.environ[key]
