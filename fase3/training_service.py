"""Orquestracao local e segura dos jobs de dados, treino e avaliacao."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fase3.evaluation_contract import evaluation_contract_matches
from fase3.llm_backend import (
    DEFAULT_LOCAL_BASE_MODEL,
    PROJECT_ROOT,
    PROMOTED_MODEL_CONFIG_PATH,
)
from fase3.model_registry import (
    get_model,
    hardware_capabilities,
    list_models,
    model_capabilities,
    model_reference,
)

DATA_DIR = PROJECT_ROOT / "fase3" / "data"
FINETUNING_DIR = PROJECT_ROOT / "resultados" / "fase3" / "finetuning"
CALIBRATION_PATH = PROJECT_ROOT / "resultados" / "fase3" / "calibracao_adapter.json"
VERSION_PATTERN = re.compile(r"^(?P<alias>[a-z0-9][a-z0-9._-]{1,63})-v(?P<number>[1-9][0-9]*)$")


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ler_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _relativo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def resolver_adapter(version: str) -> Path:
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError("Versao invalida. Use o formato alias-do-modelo-vN.")
    path = (FINETUNING_DIR / version / "lora_adapter").resolve()
    if FINETUNING_DIR.resolve() not in path.parents:
        raise ValueError("Caminho de adapter fora do diretorio permitido.")
    return path


def _adapter_metadata(adapter: Path) -> dict[str, Any]:
    manifest = _ler_json(adapter / "adapter_manifest.json") or {}
    summary = _ler_json(adapter.parent / "training_summary.json") or {}
    base_model = manifest.get("base_model") or summary.get("base_model") or DEFAULT_LOCAL_BASE_MODEL
    return {
        **summary,
        **manifest,
        "base_model": base_model,
        "model_alias": manifest.get("model_alias") or summary.get("model_alias") or "qwen2.5-1.5b",
    }


class TrainingJobManager:
    """Executa somente pipelines conhecidos e captura seu estado para a UI."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: Optional[subprocess.Popen[str]] = None
        self._job: Optional[dict[str, Any]] = None

    @property
    def ativo(self) -> bool:
        with self._lock:
            return bool(self._job and self._job["status"] in {"aguardando", "executando", "cancelando"})

    def job(self) -> Optional[dict[str, Any]]:
        with self._lock:
            if not self._job:
                return None
            snapshot = {key: value for key, value in self._job.items() if key != "_logs"}
            snapshot["logs"] = list(self._job["_logs"])
            return snapshot

    def _ambiente(self, online: bool = False) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        env["PYTHONUNBUFFERED"] = "1"
        env["HF_HUB_OFFLINE"] = "0" if online else "1"
        return env

    def _atualizar_progresso(self, linha: str) -> None:
        if not self._job:
            return
        if self._job["tipo"] == "treino":
            match = re.search(r"['\"]epoch['\"]\s*:\s*([0-9.]+)", linha)
            if match:
                epoca = float(match.group(1))
                total = float(self._job["metadados"]["epochs"])
                self._job["progresso"] = min(0.98, max(0.02, epoca / total))
                self._job["etapa"] = f"Treinando epoca {min(epoca, total):g} de {total:g}"
        elif self._job["tipo"] == "instalacao":
            match = re.search(r"\b([0-9]{1,3})%", linha)
            if match:
                percent = min(100, int(match.group(1)))
                self._job["progresso"] = min(0.98, max(0.02, percent / 100))
                self._job["etapa"] = f"Baixando arquivos do modelo: {percent}%"

    def _executar(self, command: list[str], online: bool = False) -> None:
        with self._lock:
            assert self._job is not None
            if self._job["status"] == "cancelando":
                self._job["status"] = "cancelado"
                self._job["finalizado_em"] = _agora()
                self._job["etapa"] = "Cancelado"
                return
            self._job["status"] = "executando"
            self._job["iniciado_em"] = _agora()
            self._job["progresso"] = 0.02
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                self._process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    env=self._ambiente(online=online),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=creationflags,
                )
            except OSError as exc:
                self._job["_logs"].append(f"Falha ao iniciar processo: {exc}")
                self._job["status"] = "falhou"
                self._job["codigo_saida"] = -1
                self._job["finalizado_em"] = _agora()
                self._job["etapa"] = "Falha ao iniciar processo"
                self._process = None
                return

        assert self._process.stdout is not None
        for linha in self._process.stdout:
            linha = linha.rstrip()
            with self._lock:
                if self._job is not None and linha:
                    self._job["_logs"].append(linha)
                    self._atualizar_progresso(linha)

        codigo = self._process.wait()
        with self._lock:
            assert self._job is not None
            cancelado = self._job["status"] == "cancelando"
            self._job["status"] = "cancelado" if cancelado else ("concluido" if codigo == 0 else "falhou")
            self._job["codigo_saida"] = codigo
            self._job["finalizado_em"] = _agora()
            self._job["progresso"] = 1.0 if codigo == 0 else self._job["progresso"]
            self._job["etapa"] = (
                "Concluido" if codigo == 0 else ("Cancelado" if cancelado else "Falha na execucao")
            )
            self._process = None

    def _iniciar(
        self,
        tipo: str,
        titulo: str,
        command: list[str],
        metadados: Optional[dict[str, Any]] = None,
        online: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            if self.ativo:
                raise RuntimeError("Ja existe uma operacao em andamento.")
            self._job = {
                "id": uuid.uuid4().hex[:12],
                "tipo": tipo,
                "titulo": titulo,
                "status": "aguardando",
                "etapa": "Preparando processo",
                "progresso": 0.0,
                "criado_em": _agora(),
                "iniciado_em": None,
                "finalizado_em": None,
                "codigo_saida": None,
                "metadados": metadados or {},
                "comando": command[1:],
                "_logs": deque(maxlen=600),
            }
            thread = threading.Thread(target=self._executar, args=(command, online), daemon=True)
            thread.start()
            return self.job() or {}

    def iniciar_dataset(self) -> dict[str, Any]:
        return self._iniciar(
            "dataset",
            "Preparar dataset clinico",
            [sys.executable, "-m", "fase3.data.build_finetuning_dataset"],
        )

    def iniciar_treino(self, config: dict[str, Any]) -> dict[str, Any]:
        version = config["version"]
        match = VERSION_PATTERN.fullmatch(version)
        if not match:
            raise ValueError("Versao invalida. Use o formato alias-do-modelo-vN.")
        model_alias = config.get("model_alias") or match.group("alias")
        if match.group("alias") != model_alias:
            raise ValueError("A versao deve comecar com o alias do modelo selecionado.")
        model = get_model(model_alias)
        base_model = model_reference(model)
        adapter = resolver_adapter(version)
        output_dir = adapter.parent
        if output_dir.exists():
            raise FileExistsError(f"A versao {version} ja existe e nao sera sobrescrita.")
        command = [
            sys.executable,
            "-m",
            "fase3.finetuning.train_lora",
            "--base-model",
            base_model,
            "--model-alias",
            model_alias,
            "--precision",
            str(config.get("precision", "auto")),
            "--epochs",
            str(config["epochs"]),
            "--batch-size",
            str(config["batch_size"]),
            "--gradient-accumulation-steps",
            str(config["gradient_accumulation_steps"]),
            "--learning-rate",
            str(config["learning_rate"]),
            "--max-length",
            str(config["max_length"]),
            "--seed",
            str(config["seed"]),
            "--lora-r",
            str(config["lora_r"]),
            "--lora-alpha",
            str(config["lora_alpha"]),
            "--lora-dropout",
            str(config["lora_dropout"]),
            "--output-dir",
            str(output_dir),
        ]
        if config.get("gradient_checkpointing"):
            command.append("--gradient-checkpointing")
        target_modules = config.get("target_modules") or model.get("target_modules") or []
        if target_modules:
            command.extend(["--lora-target-modules", ",".join(target_modules)])
        if model.get("revision"):
            command.extend(["--model-revision", str(model["revision"])])
        trust_remote_code = bool(config.get("trust_remote_code") and model.get("trust_remote_code"))
        if config.get("trust_remote_code") and not trust_remote_code:
            raise ValueError("trust_remote_code nao foi autorizado no cadastro deste modelo.")
        if trust_remote_code:
            command.append("--trust-remote-code")
        return self._iniciar(
            "treino",
            f"Treinar {version}",
            command,
            {
                **config,
                "model_alias": model_alias,
                "base_model": base_model,
                "trust_remote_code": trust_remote_code,
                "output_dir": _relativo(output_dir),
            },
        )

    def iniciar_instalacao(self, alias: str) -> dict[str, Any]:
        model = get_model(alias)
        return self._iniciar(
            "instalacao",
            f"Instalar {model['label']}",
            [sys.executable, "-m", "fase3.model_manager", "install", "--alias", alias],
            {"model_alias": alias, "source": model["source"]},
            online=True,
        )

    def iniciar_loss(self, version: str) -> dict[str, Any]:
        adapter = resolver_adapter(version)
        if not (adapter / "adapter_model.safetensors").exists():
            raise FileNotFoundError(f"Adapter {version} nao encontrado.")
        metadata = _adapter_metadata(adapter)
        return self._iniciar(
            "loss",
            f"Reavaliar loss de {version}",
            [
                sys.executable,
                "-m",
                "fase3.finetuning.evaluate_adapter_loss",
                "--adapter-path",
                str(adapter),
                "--base-model",
                str(metadata["base_model"]),
            ],
            {"version": version, "adapter_path": _relativo(adapter)},
        )

    def iniciar_calibracao(self, version: str, scales: list[float]) -> dict[str, Any]:
        adapter = resolver_adapter(version)
        if not (adapter / "adapter_model.safetensors").exists():
            raise FileNotFoundError(f"Adapter {version} nao encontrado.")
        metadata = _adapter_metadata(adapter)
        return self._iniciar(
            "calibracao",
            f"Calibrar e validar {version}",
            [
                sys.executable,
                "-m",
                "fase3.calibrate_adapter",
                "--adapter-path",
                str(adapter),
                "--base-model",
                str(metadata["base_model"]),
                "--scales",
                ",".join(str(scale) for scale in scales),
                "--enforce-gates",
            ],
            {"version": version, "adapter_path": _relativo(adapter), "scales": scales},
        )

    def cancelar(self) -> dict[str, Any]:
        with self._lock:
            if not self.ativo or self._job is None:
                raise RuntimeError("Nao existe operacao ativa para cancelar.")
            self._job["status"] = "cancelando"
            self._job["etapa"] = "Encerrando processo"
            if self._process is not None:
                self._process.terminate()
            return self.job() or {}

    def promover(self, version: str) -> dict[str, Any]:
        if self.ativo:
            raise RuntimeError("Aguarde a operacao atual terminar antes de promover.")
        adapter = resolver_adapter(version)
        calibracao = _ler_json(CALIBRATION_PATH)
        if not evaluation_contract_matches(calibracao):
            raise ValueError(
                "A calibracao esta desatualizada para o prompt ou os casos atuais. "
                "Execute Calibrar e validar novamente."
            )
        selecionado = (calibracao or {}).get("selecionado", {})
        caminho_calibrado = Path(str(selecionado.get("adapter_path", "")))
        if not caminho_calibrado.is_absolute():
            caminho_calibrado = PROJECT_ROOT / caminho_calibrado
        if caminho_calibrado.resolve() != adapter.resolve():
            raise ValueError("A calibracao mais recente nao pertence ao adapter selecionado.")
        if not selecionado.get("aprovado"):
            raise ValueError("O adapter nao atingiu todos os gates e nao pode ser promovido.")
        metadata = _adapter_metadata(adapter)
        config = {
            "base_model": metadata["base_model"],
            "model_alias": metadata["model_alias"],
            "revision": metadata.get("revision") or metadata.get("model_revision"),
            "precision": metadata.get("precision", "auto"),
            "trust_remote_code": bool(metadata.get("trust_remote_code", False)),
            "adapter_path": _relativo(adapter),
            "lora_scale": selecionado["lora_scale"],
            "promovido_em": _agora(),
            "metricas": selecionado,
        }
        PROMOTED_MODEL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROMOTED_MODEL_CONFIG_PATH.write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return config

    def overview(self) -> dict[str, Any]:
        train = DATA_DIR / "finetuning_train.jsonl"
        val = DATA_DIR / "finetuning_val.jsonl"

        def contar(path: Path) -> int:
            try:
                return sum(1 for linha in path.read_text(encoding="utf-8").splitlines() if linha.strip())
            except OSError:
                return 0

        adapters = []
        if FINETUNING_DIR.exists():
            for directory in FINETUNING_DIR.iterdir():
                match = VERSION_PATTERN.fullmatch(directory.name)
                adapter = directory / "lora_adapter" / "adapter_model.safetensors"
                if not match or not adapter.exists():
                    continue
                summary = _ler_json(directory / "training_summary.json") or {}
                adapters.append(
                    {
                        "version": directory.name,
                        "numero": int(match.group("number")),
                        "model_alias": match.group("alias"),
                        "base_model": summary.get("base_model"),
                        "precision": summary.get("precision", "fp16" if summary.get("fp16") else "fp32"),
                        "adapter_path": _relativo(adapter.parent),
                        "epochs": summary.get("epochs"),
                        "learning_rate": summary.get("learning_rate"),
                        "validation_loss": (
                            summary.get("metricas_reavaliadas", {}).get("validacao", {}).get("loss")
                            or summary.get("final_eval_loss")
                        ),
                        "created_at": datetime.fromtimestamp(adapter.stat().st_mtime, timezone.utc).isoformat(),
                    }
                )
        adapters.sort(key=lambda item: item["created_at"], reverse=True)
        next_versions = {}
        for model in list_models():
            numbers = [
                item["numero"] for item in adapters if item["model_alias"] == model["alias"]
            ]
            next_versions[model["alias"]] = f"{model['alias']}-v{max(numbers or [0]) + 1}"
        promoted = _ler_json(PROMOTED_MODEL_CONFIG_PATH)
        if not promoted:
            promoted = {
                "base_model": DEFAULT_LOCAL_BASE_MODEL,
                "adapter_path": "resultados/fase3/finetuning/qwen2.5-1.5b-v4/lora_adapter",
                "lora_scale": 0.75,
                "promovido_em": None,
            }
        calibration = _ler_json(CALIBRATION_PATH)
        latest_calibration = (calibration or {}).get("selecionado")
        if latest_calibration and not evaluation_contract_matches(calibration):
            latest_calibration = {
                **latest_calibration,
                "aprovado": False,
                "stale": True,
                "stale_reason": (
                    "O prompt, o split de validacao ou os casos finais mudaram. "
                    "Execute uma nova calibracao."
                ),
            }
        return {
            "dataset": {"train": contar(train), "validation": contar(val), "ready": train.exists() and val.exists()},
            "adapters": adapters,
            "next_version": next_versions.get("qwen2.5-1.5b", "qwen2.5-1.5b-v1"),
            "next_versions": next_versions,
            "models": model_capabilities(),
            "hardware": hardware_capabilities(),
            "promoted": promoted,
            "latest_calibration": latest_calibration,
            "job": self.job(),
        }
