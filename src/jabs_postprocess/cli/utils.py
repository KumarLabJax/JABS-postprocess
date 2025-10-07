import json
from pathlib import Path
from typing import Dict, Any, List


def load_json(source: Path | str | None) -> Dict[str, Any] | List[Any] | None:
    """Load JSON from a file path, JSON string, or return None.

    Args:
        source: Can be:
            - Path object pointing to a JSON file
            - String containing a file path to a JSON file
            - String containing raw JSON content
            - None (returns None)

    Returns:
        Parsed JSON (dict, list, or other JSON-serializable type), or None if input is None

    Raises:
        ValueError: If source cannot be parsed as JSON or loaded from file
        TypeError: If source is not a supported type
    """
    if source is None:
        return None

    if isinstance(source, Path):
        # It's a Path object, read from file
        try:
            with open(source, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise ValueError(f"File not found: {source}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {source}: {e}")

    elif isinstance(source, str):
        # It's a string - could be a file path or JSON content
        stripped = source.strip()

        # Check if it looks like JSON (starts with {, [, or is a quoted string)
        if stripped and stripped[0] in '{["' or stripped in ("true", "false", "null"):
            # Try as JSON first
            try:
                return json.loads(source)
            except json.JSONDecodeError:
                # If it fails, try as file path
                try:
                    with open(source, "r", encoding="utf-8") as f:
                        return json.load(f)
                except FileNotFoundError:
                    # Original JSON parse error is more relevant
                    raise ValueError(
                        f"Invalid JSON: {source[:100]}{'...' if len(source) > 100 else ''}"
                    )
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in file {source}: {e}")
        else:
            # Doesn't look like JSON, try as file path first
            try:
                with open(source, "r", encoding="utf-8") as f:
                    return json.load(f)
            except FileNotFoundError:
                # Last attempt: maybe it's JSON without typical start chars
                try:
                    return json.loads(source)
                except json.JSONDecodeError:
                    raise ValueError(
                        f"'{source}' is neither a valid file path nor valid JSON"
                    )
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in file {source}: {e}")
    else:
        raise TypeError(
            f"source must be Path, str, or None, got {type(source).__name__}"
        )
