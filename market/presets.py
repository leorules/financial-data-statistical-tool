import json

from market.config import PRESETS_PATH


def load_all() -> dict[str, dict]:
    return json.loads(PRESETS_PATH.read_text()) if PRESETS_PATH.exists() else {}


def save(name: str, preset: dict) -> None:
    PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRESETS_PATH.write_text(json.dumps(load_all() | {name: preset}, indent=2, default=str))


def delete(name: str) -> None:
    PRESETS_PATH.write_text(json.dumps({k: v for k, v in load_all().items() if k != name}, indent=2))
