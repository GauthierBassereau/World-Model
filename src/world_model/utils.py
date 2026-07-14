from typing import Any


def unwrap_world_model(model: Any) -> Any:
    """Unwrap common containers around the world model."""
    unwrapped = model
    seen: set[int] = set()
    while True:
        model_id = id(unwrapped)
        if model_id in seen:
            return unwrapped
        seen.add(model_id)

        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
            continue
        if hasattr(unwrapped, "_orig_mod"):
            unwrapped = unwrapped._orig_mod
            continue
        return unwrapped
