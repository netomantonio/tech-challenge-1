"""Comandos seguros para preparar modelos cadastrados no backend."""

from __future__ import annotations

import argparse
import json

from fase3.model_registry import get_model, model_installed


def install(alias: str) -> dict:
    model = get_model(alias)
    if model["source_type"] == "local":
        return {"alias": alias, "installed": model_installed(model), "source": model["source"]}
    from huggingface_hub import snapshot_download

    print(f"Preparando download de {model['source']} ({alias})", flush=True)
    path = snapshot_download(
        repo_id=model["source"],
        revision=model.get("revision"),
    )
    result = {"alias": alias, "installed": True, "source": model["source"], "path": path}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["install"])
    parser.add_argument("--alias", required=True)
    args = parser.parse_args()
    if args.command == "install":
        install(args.alias)


if __name__ == "__main__":
    main()
