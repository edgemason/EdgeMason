import copy
import re
import tempfile

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    r"|^[0-9a-f]{32}$",
    re.IGNORECASE,
)
_VERSION_RE = re.compile(r"^[a-zA-Z0-9_.-]+==\d+\.\d+\.\d+")
_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$"
)


def _is_path(value):
    if value.startswith("/"):
        return True
    tmp = tempfile.gettempdir()
    return bool(tmp) and tmp in value


def _replace(obj):
    if isinstance(obj, dict):
        return {
            key: "<SYSTEM_INFO>" if key == "system_info" else _replace(value)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [_replace(item) for item in obj]
    if isinstance(obj, str):
        if _is_path(obj):
            return "<PATH>"
        if _UUID_RE.match(obj):
            return "<UUID>"
        if _VERSION_RE.match(obj):
            return "<VERSION>"
        if _TIMESTAMP_RE.match(obj):
            return "<TIMESTAMP>"
        return obj
    return obj


def normalise(obj):
    """Deep-copy obj with run-specific values replaced by placeholders."""
    return _replace(copy.deepcopy(obj))