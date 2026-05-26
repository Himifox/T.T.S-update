from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import pathlib
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.file_utils import atomic_write_json
from utils.frontend_utils import find_model_directory, find_workshop_item_by_id
from utils.url_utils import encode_url_path


OVERLAY_FILENAME = "live2d_emotion_overrides.json"
_overlay_lock = threading.RLock()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Live2DContext:
    requested_name: str
    source: str
    item_id: str
    model_dir: str
    actual_model_dir: str
    model_config_file: str
    model_config_path: str
    model_config_url: str
    model_identity: str
    fingerprint: str
    subdir_name: str | None = None


def locate_model_config(model_dir: str) -> tuple[str | None, str | None, str | None]:
    if not model_dir or not os.path.isdir(model_dir):
        return None, None, None

    for file_name in os.listdir(model_dir):
        if file_name.endswith(".model3.json"):
            return model_dir, file_name, None

    for subdir in os.listdir(model_dir):
        subdir_path = os.path.join(model_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for file_name in os.listdir(subdir_path):
            if file_name.endswith(".model3.json"):
                return subdir_path, file_name, subdir

    return None, None, None


def normalize_relative_asset_path(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.replace("\\", "/")
    path = pathlib.PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        return None
    return str(path)


def read_model_config(model_config_path: str) -> dict[str, Any]:
    with open(model_config_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def normalized_file_references(config_data: dict[str, Any]) -> dict[str, Any]:
    file_refs = copy.deepcopy(config_data.get("FileReferences") or {})
    if not isinstance(file_refs, dict):
        file_refs = {}
    if not isinstance(file_refs.get("Motions"), dict):
        file_refs["Motions"] = {}
    if not isinstance(file_refs.get("Expressions"), list):
        file_refs["Expressions"] = []
    return file_refs


def scan_live2d_assets(actual_model_dir: str) -> dict[str, list[str]]:
    expression_files: list[str] = []
    motion_files: list[str] = []

    for root, _, files in os.walk(actual_model_dir):
        for file_name in files:
            if not (file_name.endswith(".exp3.json") or file_name.endswith(".motion3.json")):
                continue
            full_path = os.path.join(root, file_name)
            rel_path = os.path.relpath(full_path, actual_model_dir).replace("\\", "/")
            if file_name.endswith(".exp3.json"):
                expression_files.append(rel_path)
            else:
                motion_files.append(rel_path)

    expression_files.sort()
    motion_files.sort()
    return {"expression_files": expression_files, "motion_files": motion_files}


def _walk_json_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _walk_json_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_json_strings(item)


def load_vtube_expression_refs(actual_model_dir: str) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()

    for root, _, files in os.walk(actual_model_dir):
        for file_name in files:
            if not file_name.endswith(".vtube.json"):
                continue
            vtube_path = os.path.join(root, file_name)
            try:
                with open(vtube_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
            except Exception:
                continue
            for raw in _walk_json_strings(data):
                if not raw.endswith(".exp3.json"):
                    continue
                normalized = normalize_relative_asset_path(raw)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    refs.append(normalized)

    return refs


def _resolve_expression_file(raw_file: str, scanned_expressions: list[str]) -> tuple[str, bool, bool]:
    normalized = normalize_relative_asset_path(raw_file) or raw_file.replace("\\", "/")
    scanned_set = set(scanned_expressions)
    if normalized in scanned_set:
        return normalized, True, False

    basename = os.path.basename(normalized)
    matches = [path for path in scanned_expressions if os.path.basename(path) == basename]
    if len(matches) == 1:
        return matches[0], True, False
    if len(matches) > 1:
        return normalized, False, True
    return normalized, False, False


def compose_expression_refs(
    config_data: dict[str, Any],
    scanned_expressions: list[str],
    vtube_refs: list[str] | None = None,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    registered_files: set[str] = set()

    expressions = normalized_file_references(config_data).get("Expressions") or []
    for item in expressions:
        if not isinstance(item, dict):
            continue
        raw_file = item.get("File")
        if not raw_file:
            continue
        file_path, exists, ambiguous = _resolve_expression_file(str(raw_file), scanned_expressions)
        registered_files.add(file_path)
        refs.append(
            {
                "name": item.get("Name") or os.path.basename(file_path).replace(".exp3.json", ""),
                "file": file_path,
                "source": "model",
                "exists": exists,
                "ambiguous": ambiguous,
            }
        )

    for raw_file in vtube_refs or []:
        file_path, exists, ambiguous = _resolve_expression_file(raw_file, scanned_expressions)
        if file_path in registered_files:
            continue
        registered_files.add(file_path)
        refs.append(
            {
                "name": os.path.basename(file_path).replace(".exp3.json", ""),
                "file": file_path,
                "source": "vtube",
                "exists": exists,
                "ambiguous": ambiguous,
            }
        )

    for file_path in scanned_expressions:
        if file_path in registered_files:
            continue
        refs.append(
            {
                "name": os.path.basename(file_path).replace(".exp3.json", ""),
                "file": file_path,
                "source": "disk",
                "exists": True,
                "ambiguous": False,
            }
        )

    return refs


def derive_emotion_mapping(config_data: dict[str, Any]) -> dict[str, Any]:
    derived = {"motions": {}, "expressions": {}}
    file_refs = normalized_file_references(config_data)

    for group_name, items in (file_refs.get("Motions") or {}).items():
        files: list[str] = []
        for item in items or []:
            file_path = item.get("File") if isinstance(item, dict) else None
            normalized = normalize_relative_asset_path(file_path)
            if normalized:
                files.append(normalized)
        derived["motions"][group_name] = files

    for item in file_refs.get("Expressions") or []:
        if not isinstance(item, dict):
            continue
        file_path = normalize_relative_asset_path(item.get("File"))
        if not file_path:
            continue
        name = str(item.get("Name") or "")
        group = name.split("_", 1)[0] if "_" in name else "neutral"
        derived["expressions"].setdefault(group, []).append(file_path)

    return derived


def sanitize_emotion_mapping(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"motions": {}, "expressions": {}, "hotkeys": {}}

    result = {"motions": {}, "expressions": {}, "hotkeys": {}}

    motions = data.get("motions") if isinstance(data.get("motions"), dict) else {}
    for group_name, files in motions.items():
        if group_name == "常驻":
            continue
        clean_files = []
        for file_path in files or []:
            normalized = normalize_relative_asset_path(file_path)
            if normalized:
                clean_files.append(normalized)
        if clean_files:
            result["motions"][str(group_name)] = clean_files

    expressions = data.get("expressions") if isinstance(data.get("expressions"), dict) else {}
    for group_name, files in expressions.items():
        clean_files = []
        for file_path in files or []:
            normalized = normalize_relative_asset_path(file_path)
            if normalized:
                clean_files.append(normalized)
        if clean_files:
            result["expressions"][str(group_name)] = clean_files

    hotkeys = data.get("hotkeys") if isinstance(data.get("hotkeys"), dict) else {}
    result["hotkeys"] = copy.deepcopy(hotkeys)
    return result


def is_empty_emotion_mapping(mapping: dict[str, Any]) -> bool:
    return not any(
        mapping.get(key)
        for key in ("motions", "expressions", "hotkeys")
    )


def _source_from_url_prefix(url_prefix: str | None) -> str:
    if url_prefix == "/workshop":
        return "steam"
    if url_prefix == "/static":
        return "static"
    if url_prefix == "/user_mods":
        return "user_mods"
    return "documents"


def _context_item_id(model_dir: str, actual_model_dir: str, item_id: str | None, source: str) -> str:
    if item_id:
        return str(item_id)
    if source != "steam":
        return ""
    model_path = Path(model_dir)
    actual_path = Path(actual_model_dir)
    if actual_path.parent == model_path:
        return model_path.name
    return model_path.name


def _relative_config_path(model_dir: str, actual_model_dir: str, model_config_file: str) -> str:
    path = os.path.join(actual_model_dir, model_config_file)
    return os.path.relpath(path, model_dir).replace("\\", "/")


def _build_model_config_url(
    requested_name: str,
    url_prefix: str | None,
    model_dir: str,
    actual_model_dir: str,
    model_config_file: str,
    subdir_name: str | None,
    item_id: str,
) -> str:
    if not url_prefix:
        return ""
    if url_prefix == "/workshop":
        if subdir_name:
            return encode_url_path(f"{url_prefix}/{item_id}/{subdir_name}/{model_config_file}")
        return encode_url_path(f"{url_prefix}/{item_id}/{model_config_file}")
    if subdir_name:
        return encode_url_path(f"{url_prefix}/{requested_name}/{subdir_name}/{model_config_file}")
    return encode_url_path(f"{url_prefix}/{requested_name}/{model_config_file}")


def _fingerprint(model_config_path: str, actual_model_dir: str) -> str:
    digest = hashlib.sha256()
    with open(model_config_path, "rb") as file:
        digest.update(file.read())
    assets = scan_live2d_assets(actual_model_dir)
    for key in ("expression_files", "motion_files"):
        for rel_path in assets[key]:
            digest.update(b"\0")
            digest.update(rel_path.encode("utf-8", errors="surrogatepass"))
    return digest.hexdigest()


def resolve_live2d_context(
    model_name: str | None = None,
    item_id: str | None = None,
) -> Live2DContext:
    requested_name = str(model_name or item_id or "")
    if not requested_name or "/" in requested_name or "\\" in requested_name or ".." in requested_name:
        raise FileNotFoundError("invalid model name")
    if item_id and ("/" in str(item_id) or "\\" in str(item_id) or ".." in str(item_id)):
        raise FileNotFoundError("invalid item id")

    if item_id:
        model_dir, url_prefix = find_workshop_item_by_id(str(item_id))
        if (not model_dir or not os.path.exists(model_dir)) and requested_name == str(item_id):
            model_dir, url_prefix = find_model_directory(requested_name)
    else:
        model_dir, url_prefix = find_model_directory(requested_name)

    if not model_dir or not os.path.exists(model_dir):
        raise FileNotFoundError("model directory not found")

    actual_model_dir, model_config_file, subdir_name = locate_model_config(model_dir)
    if not actual_model_dir or not model_config_file:
        raise FileNotFoundError("model3.json not found")

    source = _source_from_url_prefix(url_prefix)
    resolved_item_id = _context_item_id(model_dir, actual_model_dir, item_id, source)
    model_config_path = os.path.join(actual_model_dir, model_config_file)
    rel_config = _relative_config_path(model_dir, actual_model_dir, model_config_file)
    folder_name = os.path.basename(actual_model_dir)
    identity_source = "steam" if source == "steam" else source
    identity_name = resolved_item_id if source == "steam" else folder_name
    model_identity = f"{identity_source}:{identity_name}:{rel_config}"
    model_config_url = _build_model_config_url(
        requested_name,
        url_prefix,
        model_dir,
        actual_model_dir,
        model_config_file,
        subdir_name,
        resolved_item_id,
    )

    return Live2DContext(
        requested_name=requested_name,
        source=source,
        item_id=resolved_item_id,
        model_dir=model_dir,
        actual_model_dir=actual_model_dir,
        model_config_file=model_config_file,
        model_config_path=model_config_path,
        model_config_url=model_config_url,
        model_identity=model_identity,
        fingerprint=_fingerprint(model_config_path, actual_model_dir),
        subdir_name=subdir_name,
    )


def overlay_path(config_mgr: Any) -> Path:
    return Path(config_mgr.config_dir) / OVERLAY_FILENAME


def load_emotion_overlay(config_mgr: Any) -> dict[str, Any]:
    path = overlay_path(config_mgr)
    if not path.exists():
        return {"version": 1, "models": {}}
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as e:
        logger.warning("Failed to read Live2D emotion overlay %s: %s", path, e)
        return {"version": 1, "models": {}}
    if not isinstance(data, dict):
        return {"version": 1, "models": {}}
    if not isinstance(data.get("models"), dict):
        data["models"] = {}
    data["version"] = 1
    return data


def save_emotion_overlay(config_mgr: Any, overlay: dict[str, Any]) -> None:
    with _overlay_lock:
        atomic_write_json(overlay_path(config_mgr), overlay, ensure_ascii=False, indent=2)


def get_overlay_entry(config_mgr: Any, context: Live2DContext) -> tuple[dict[str, Any] | None, str]:
    with _overlay_lock:
        overlay = load_emotion_overlay(config_mgr)
        models = overlay.get("models") or {}
        entry = models.get(context.model_identity)
        if isinstance(entry, dict):
            return entry, "overlay"

        matches = [
            (identity, item)
            for identity, item in models.items()
            if isinstance(item, dict) and item.get("fingerprint") == context.fingerprint
        ]
        if len(matches) == 1:
            old_identity, old_entry = matches[0]
            models[context.model_identity] = old_entry
            models.pop(old_identity, None)
            save_emotion_overlay(config_mgr, overlay)
            return old_entry, "overlay_migrated"

    return None, "missing"


def save_overlay_mapping(config_mgr: Any, context: Live2DContext, mapping: dict[str, Any]) -> bool:
    with _overlay_lock:
        overlay = load_emotion_overlay(config_mgr)
        models = overlay.setdefault("models", {})
        changed = False
        if is_empty_emotion_mapping(mapping):
            changed = context.model_identity in models
            models.pop(context.model_identity, None)
        else:
            models[context.model_identity] = {
                "display_name": context.requested_name,
                "source": context.source,
                "item_id": context.item_id,
                "model_config_file": context.model_config_file,
                "fingerprint": context.fingerprint,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "mapping": mapping,
            }
            changed = True
        if changed:
            save_emotion_overlay(config_mgr, overlay)
        return changed
