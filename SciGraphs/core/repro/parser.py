# Pipeline parser for YAML/JSON with canonicalization and hashing
#
# Handles loading, parsing, normalizing and hashing pipeline files.

import json
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from .schema import PipelineSchema, ValidationError, validate_pipeline, parse_spec

# Try to import YAML support (optional dependency)
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class ParseError(Exception):
    """Raised when pipeline parsing fails."""
    pass


def _parse_scalar(value: str) -> Any:
    """Parse a small YAML scalar subset used by pipeline files."""
    value = value.strip()

    if value == "":
        return None
    if value == "{}":
        return {}
    if value == "[]":
        return []
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in {"null", "none", "~"}:
        return None

    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def _split_yaml_key_value(text: str) -> Tuple[str, str]:
    """Split a YAML key/value line on the first colon."""
    if ":" not in text:
        raise ParseError(f"Invalid YAML line: {text}")
    key, value = text.split(":", 1)
    key = key.strip()
    if not key:
        raise ParseError(f"Invalid YAML key in line: {text}")
    return key, value.strip()


def _minimal_yaml_load(content: str) -> Dict[str, Any]:
    """
    Parse the pipeline YAML subset without external dependencies.

    This intentionally supports the human-authored structures used by
    SciGraphs pipelines: nested mappings, simple lists, inline arrays,
    booleans, numbers, strings, and empty dict/list literals.
    """
    lines = []
    for raw_line in content.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, raw_line.strip()))

    if not lines:
        raise ParseError("YAML content is empty")

    def parse_block(index: int, indent: int) -> Tuple[Any, int]:
        if index >= len(lines):
            return {}, index

        is_list = lines[index][1].startswith("- ")
        container: Any = [] if is_list else {}

        while index < len(lines):
            line_indent, text = lines[index]
            if line_indent < indent:
                break
            if line_indent > indent:
                raise ParseError(f"Unexpected indentation near: {text}")

            if is_list:
                if not text.startswith("- "):
                    break
                item_text = text[2:].strip()
                index += 1

                if not item_text:
                    if index < len(lines) and lines[index][0] > line_indent:
                        item, index = parse_block(index, lines[index][0])
                    else:
                        item = None
                    container.append(item)
                    continue

                if ":" in item_text and not item_text.startswith(("http://", "https://")):
                    key, value = _split_yaml_key_value(item_text)
                    item = {key: _parse_scalar(value)} if value else {key: None}
                    if index < len(lines) and lines[index][0] > line_indent:
                        child, index = parse_block(index, lines[index][0])
                        if isinstance(child, dict):
                            item.update(child)
                        elif value == "":
                            item[key] = child
                    container.append(item)
                    continue

                container.append(_parse_scalar(item_text))
                continue

            if text.startswith("- "):
                break

            key, value = _split_yaml_key_value(text)
            index += 1

            if value:
                container[key] = _parse_scalar(value)
            elif index < len(lines) and lines[index][0] > line_indent:
                child, index = parse_block(index, lines[index][0])
                container[key] = child
            else:
                container[key] = None

        return container, index

    parsed, final_index = parse_block(0, lines[0][0])
    if final_index != len(lines):
        raise ParseError("Could not parse complete YAML document")
    if not isinstance(parsed, dict):
        raise ParseError("YAML content must be a mapping/object")
    return parsed


def _resolve_path(filepath: str, base_dir: Optional[str] = None) -> str:
    """
    Resolve a filepath, handling Blender-style // relative paths.

    Args:
        filepath: Path that may be absolute, relative, or //-prefixed
        base_dir: Base directory for resolving // paths (typically blend file dir)

    Returns:
        Resolved absolute path
    """
    if filepath.startswith("//"):
        # Blender-style relative path
        if base_dir:
            return os.path.join(base_dir, filepath[2:])
        else:
            return filepath[2:]  # Fall back to current dir relative
    elif os.path.isabs(filepath):
        return filepath
    else:
        if base_dir:
            return os.path.join(base_dir, filepath)
        return os.path.abspath(filepath)


def load_file(filepath: str) -> str:
    """
    Load a pipeline file and return its contents.

    Args:
        filepath: Path to YAML or JSON file

    Returns:
        File contents as string

    Raises:
        ParseError: If file cannot be read
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise ParseError(f"Pipeline file not found: {filepath}")
    except IOError as e:
        raise ParseError(f"Error reading pipeline file: {e}")


def parse_content(content: str, format_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse pipeline content from string.

    Args:
        content: YAML or JSON string
        format_hint: Optional format hint ('yaml', 'json')

    Returns:
        Parsed dictionary

    Raises:
        ParseError: If parsing fails
    """
    # Try JSON first (always available, stricter syntax)
    if format_hint == "json" or not format_hint:
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            if format_hint == "json":
                raise ParseError(f"Invalid JSON: {e}")
            # Fall through to try YAML

    # Try YAML
    if HAS_YAML:
        try:
            data = yaml.safe_load(content)
            if not isinstance(data, dict):
                raise ParseError("YAML content must be a mapping/object")
            return data
        except yaml.YAMLError as e:
            raise ParseError(f"Invalid YAML: {e}")
    elif format_hint == "yaml":
        return _minimal_yaml_load(content)

    try:
        return _minimal_yaml_load(content)
    except ParseError as e:
        raise ParseError(f"Could not parse content as JSON or YAML: {e}")


def detect_format(filepath: str) -> str:
    """Detect file format from extension."""
    ext = Path(filepath).suffix.lower()
    if ext in ('.yaml', '.yml'):
        return 'yaml'
    elif ext == '.json':
        return 'json'
    else:
        return 'auto'


def parse_pipeline(
    source: Union[str, Path, Dict[str, Any]],
    base_dir: Optional[str] = None,
) -> Tuple[PipelineSchema, Dict[str, Any], str]:
    """
    Parse a pipeline from file path, string content, or dictionary.

    Args:
        source: File path, content string, or already-parsed dict
        base_dir: Base directory for resolving relative paths

    Returns:
        Tuple of (PipelineSchema, raw_dict, canonical_hash)

    Raises:
        ParseError: If parsing fails
        ValidationError: If validation fails
    """
    raw_dict: Dict[str, Any]

    if isinstance(source, dict):
        raw_dict = source
    elif isinstance(source, Path):
        source = str(source)
        content = load_file(source)
        format_hint = detect_format(source)
        raw_dict = parse_content(content, format_hint)
        if base_dir is None:
            base_dir = str(Path(source).parent)
    elif isinstance(source, str):
        if os.path.isfile(source):
            content = load_file(source)
            format_hint = detect_format(source)
            raw_dict = parse_content(content, format_hint)
            if base_dir is None:
                base_dir = str(Path(source).parent)
        else:
            # Assume it's content
            raw_dict = parse_content(source)

    # Validate
    validate_pipeline(raw_dict)

    # Resolve output_dir path
    if "meta" in raw_dict and "output_dir" in raw_dict["meta"]:
        raw_dict["meta"]["output_dir"] = _resolve_path(
            raw_dict["meta"]["output_dir"],
            base_dir
        )

    if "dataset" in raw_dict and isinstance(raw_dict["dataset"], dict):
        dataset = raw_dict["dataset"]
        if "filepath" in dataset and dataset["filepath"]:
            dataset["filepath"] = _resolve_path(dataset["filepath"], base_dir)

    # Parse to typed schema
    schema = parse_spec(raw_dict)

    # Compute canonical hash
    canonical_hash = compute_pipeline_hash(raw_dict)

    return schema, raw_dict, canonical_hash


def canonicalize_pipeline(data: Dict[str, Any]) -> str:
    """
    Convert pipeline dictionary to canonical JSON string.

    Uses sorted keys and consistent formatting for reproducible hashing.

    Args:
        data: Pipeline dictionary

    Returns:
        Canonical JSON string
    """
    return json.dumps(
        data,
        sort_keys=True,
        indent=None,
        separators=(',', ':'),
        ensure_ascii=True,
    )


def compute_pipeline_hash(data: Dict[str, Any]) -> str:
    """
    Compute SHA256 hash of canonical pipeline representation.

    Args:
        data: Pipeline dictionary

    Returns:
        Hex digest of SHA256 hash
    """
    canonical = canonicalize_pipeline(data)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def save_pipeline(
    schema: PipelineSchema,
    filepath: str,
    format: str = "json",
) -> None:
    """
    Save a pipeline schema to file.

    Args:
        schema: Pipeline schema to save
        filepath: Output file path
        format: 'json' or 'yaml'
    """
    data = schema.to_dict()

    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        if format == "yaml" and HAS_YAML:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        else:
            json.dump(data, f, indent=2, sort_keys=False)


def save_canonical(data: Dict[str, Any], filepath: str) -> str:
    """
    Save canonical (normalized) pipeline JSON and return hash.

    Args:
        data: Pipeline dictionary
        filepath: Output file path

    Returns:
        Pipeline hash
    """
    canonical = canonicalize_pipeline(data)
    pipeline_hash = hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        # Write pretty-printed for readability, but hash is from canonical
        json.dump(json.loads(canonical), f, indent=2, sort_keys=True)

    return pipeline_hash
