"""Registro persistente de modelos-base disponiveis para a Fase 3."""

from __future__ import annotations

import json
import os
import platform
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT_ROOT / ".cache" / "fase3-model-registry.json"
ALIAS_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")

BUILTIN_MODELS: dict[str, dict[str, Any]] = {
    "qwen2.5-0.5b": {
        "alias": "qwen2.5-0.5b",
        "label": "Qwen2.5 0.5B Instruct",
        "source_type": "huggingface",
        "source": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": None,
        "builtin": True,
    },
    "qwen2.5-1.5b": {
        "alias": "qwen2.5-1.5b",
        "label": "Qwen2.5 1.5B Instruct",
        "source_type": "huggingface",
        "source": "Qwen/Qwen2.5-1.5B-Instruct",
        "revision": None,
        "builtin": True,
    },
}


def _read_custom_models() -> dict[str, dict[str, Any]]:
    try:
        content = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    models = content.get("models", {}) if isinstance(content, dict) else {}
    return models if isinstance(models, dict) else {}


def list_models() -> list[dict[str, Any]]:
    models = {**BUILTIN_MODELS, **_read_custom_models()}
    return [models[alias] for alias in sorted(models)]


def get_model(alias: str) -> dict[str, Any]:
    if not ALIAS_PATTERN.fullmatch(alias):
        raise ValueError("Alias de modelo invalido.")
    models = {item["alias"]: item for item in list_models()}
    try:
        return models[alias]
    except KeyError as exc:
        raise KeyError(f"Modelo {alias!r} nao cadastrado.") from exc


def _allowed_model_roots() -> list[Path]:
    configured = os.getenv("FASE3_MODEL_ROOTS", "")
    roots = [Path(item).expanduser().resolve() for item in configured.split(os.pathsep) if item]
    roots.append((PROJECT_ROOT / "modelos").resolve())
    return roots


def allowed_model_roots() -> list[str]:
    return [str(path) for path in _allowed_model_roots()]


def _validate_local_source(source: str) -> str:
    path = Path(source).expanduser().resolve()
    if not any(path == root or root in path.parents for root in _allowed_model_roots()):
        raise ValueError("Modelo local fora das raizes permitidas por FASE3_MODEL_ROOTS.")
    if not path.is_dir():
        raise ValueError("Diretorio do modelo local nao encontrado.")
    return str(path)


def register_model(payload: dict[str, Any]) -> dict[str, Any]:
    alias = str(payload.get("alias", "")).strip().lower()
    if not ALIAS_PATTERN.fullmatch(alias):
        raise ValueError("Alias deve usar 2 a 64 letras minusculas, numeros, ponto, hifen ou underscore.")
    if alias in BUILTIN_MODELS:
        raise ValueError("Um modelo predefinido nao pode ser substituido.")
    source_type = str(payload.get("source_type", "")).strip().lower()
    source = str(payload.get("source", "")).strip()
    if source_type not in {"huggingface", "local"}:
        raise ValueError("source_type deve ser huggingface ou local.")
    if source_type == "local":
        source = _validate_local_source(source)
    elif not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", source):
        raise ValueError("Informe um repo ID do Hugging Face no formato organizacao/modelo.")
    revision = str(payload.get("revision") or "").strip() or None
    trust_remote_code = bool(payload.get("trust_remote_code", False))
    if trust_remote_code and (source_type != "huggingface" or not revision):
        raise ValueError("trust_remote_code exige repo Hugging Face com revisao fixa.")
    target_modules = payload.get("target_modules") or []
    if not isinstance(target_modules, list) or not all(
        isinstance(item, str) and re.fullmatch(r"[A-Za-z0-9._-]+", item)
        for item in target_modules
    ):
        raise ValueError("target_modules deve conter apenas nomes de modulos validos.")
    model = {
        "alias": alias,
        "label": str(payload.get("label") or alias).strip()[:100],
        "source_type": source_type,
        "source": source,
        "revision": revision,
        "target_modules": target_modules,
        "trust_remote_code": trust_remote_code,
        "builtin": False,
    }
    custom = _read_custom_models()
    custom[alias] = model
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps({"models": custom}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return model


def unregister_model(alias: str) -> None:
    if alias in BUILTIN_MODELS:
        raise ValueError("Modelos predefinidos nao podem ser removidos.")
    custom = _read_custom_models()
    if alias not in custom:
        raise KeyError(f"Modelo {alias!r} nao cadastrado.")
    del custom[alias]
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps({"models": custom}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def model_reference(model: dict[str, Any]) -> str:
    return str(model["source"])


def model_installed(model: dict[str, Any]) -> bool:
    if model["source_type"] == "local":
        return Path(model["source"]).is_dir()
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=model["source"],
            revision=model.get("revision"),
            local_files_only=True,
        )
        return True
    except Exception:
        # O huggingface_hub varia a classe da falha de cache entre versoes.
        # Este teste e somente de disponibilidade local e nunca deve baixar.
        return False


def model_capabilities() -> list[dict[str, Any]]:
    return [{**model, "installed": model_installed(model)} for model in list_models()]


def hardware_capabilities() -> dict[str, Any]:
    result: dict[str, Any] = {
        "device": "cpu",
        "device_name": "CPU",
        "cuda_available": False,
        "bf16_supported": False,
        "quantization": ["auto", "fp32"],
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
    }
    try:
        import psutil

        result["system_memory_bytes"] = int(psutil.virtual_memory().total)
    except ImportError:
        if hasattr(os, "sysconf"):
            try:
                result["system_memory_bytes"] = int(
                    os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                )
            except (OSError, ValueError):
                pass
    try:
        import torch

        result["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            result.update(
                {
                    "device": "cuda",
                    "device_name": torch.cuda.get_device_name(0),
                    "cuda_available": True,
                    "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                    "quantization": ["auto", "fp32", "fp16", "bf16"],
                    "memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
                    "gpu_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
                }
            )
            try:
                import bitsandbytes  # noqa: F401

                result["quantization"].append("nf4")
            except ImportError:
                pass
    except ImportError:
        result["torch_version"] = None
    return result
