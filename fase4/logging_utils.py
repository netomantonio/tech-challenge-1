"""Auditoria estruturada do monitoramento multimodal (Fase 4).

Mesma convencao da Fase 3 (`fase3/logging_utils.py`): um logger proprio com
`propagate = False` e um evento JSON por linha, gravado tambem em arquivo
`.jsonl`. Aqui a auditoria registra o que cada modalidade detectou e qual
alerta foi emitido, porque o enunciado pede alertas automaticos para a
equipe medica e isso precisa ser rastreavel depois.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fase4.config import DIR_RESULTADOS

DEFAULT_AUDIT_LOG_PATH = DIR_RESULTADOS / "auditoria_monitoramento.jsonl"

_logger = logging.getLogger("fase4_monitoramento_multimodal")
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


def _audit_log_path() -> Path:
    return Path(os.getenv("FASE4_AUDIT_LOG_PATH", str(DEFAULT_AUDIT_LOG_PATH)))


def registrar_evento(evento: str, **campos: Any) -> dict:
    """Grava um evento de auditoria (stdout em JSON + arquivo `.jsonl`) e o retorna."""
    registro = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": evento,
        **campos,
    }
    linha = json.dumps(registro, ensure_ascii=False, default=str)

    if os.getenv("FASE4_LOG_SILENCIOSO") != "1":
        _logger.info(linha)

    path = _audit_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(linha + "\n")

    return registro
