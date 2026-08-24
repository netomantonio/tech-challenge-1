"""CLI portatil para diagnostico e preparacao da infraestrutura da Fase 3."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

from fase3.llm_backend import get_promoted_local_config
from fase3.model_manager import install
from fase3.model_registry import hardware_capabilities, model_capabilities


def private_addresses() -> list[str]:
    addresses: set[str] = set()
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            if item[4][0] != "127.0.0.1":
                addresses.add(item[4][0])
    except OSError:
        pass
    return sorted(addresses)


def doctor(require_runtime: bool = False) -> dict:
    promoted = get_promoted_local_config()
    models = model_capabilities()
    matching = next(
        (
            model
            for model in models
            if model["alias"] == promoted.get("model_alias")
            or model["source"] == promoted.get("base_model")
        ),
        None,
    )
    adapter_ready = Path(promoted["adapter_path"], "adapter_model.safetensors").is_file()
    model_ready = bool(matching and matching["installed"])
    result = {
        "ready": adapter_ready and model_ready,
        "hardware": hardware_capabilities(),
        "private_addresses": private_addresses(),
        "promoted": promoted,
        "model": matching,
        "adapter_ready": adapter_ready,
        "model_ready": model_ready,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if require_runtime and not result["ready"]:
        raise SystemExit(
            "Runtime incompleto. Instale o modelo promovido pela interface ou execute "
            "python -m fase3.manage install --alias <alias>."
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--require-runtime", action="store_true")
    install_parser = subparsers.add_parser("install")
    install_parser.add_argument("--alias", required=True)
    subparsers.add_parser("models")
    args = parser.parse_args()
    if args.command == "doctor":
        doctor(args.require_runtime)
    elif args.command == "install":
        install(args.alias)
    else:
        print(json.dumps(model_capabilities(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
