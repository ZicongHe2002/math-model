"""Named hardware profiles shipped with the package; loading never queries a device or the network."""
from __future__ import annotations

import json
from importlib.resources import files

from .errors import MetadataError
from .hardware import HardwareProfile


DEVICE_ALIASES = {
    "tpu-v6-e": "tpu-v6e",
    "tpu-v6e": "tpu-v6e",
    "v6e": "tpu-v6e",
    "trillium": "tpu-v6e",
}

_PROFILE_FILES = {"tpu-v6e": "tpu_v6e.json"}


def load_device_profile(name: str) -> HardwareProfile:
    """Return a fresh profile for a supported device name (case-insensitive).

    The bundled data records source URLs and a retrieval date. Physical capacity,
    compiler allocation budgets and calibration evidence remain separate fields.
    """
    if not isinstance(name, str) or not name.strip():
        raise MetadataError("device must be a non-empty name, such as 'tpu-v6e'")
    canonical = DEVICE_ALIASES.get(name.strip().lower())
    if canonical not in _PROFILE_FILES:
        raise MetadataError(f"unknown device {name!r}; supported names: {', '.join(sorted(DEVICE_ALIASES))}")
    resource = files("mla_model").joinpath("profiles", _PROFILE_FILES[canonical])
    # Parse afresh so caller edits cannot alter a later invocation's device data.
    return HardwareProfile.from_source(json.loads(resource.read_text(encoding="utf-8")))
