"""Logging estruturado de auditoria do assistente medico (Fase 3).

Segue a convencao ja usada em ``src/api.py``: um logger nomeado com
``StreamHandler`` proprio (``propagate = False`` para nao duplicar linhas) e
payloads em JSON de uma linha por evento. Alem do stdout, cada evento e
tambem persistido em um arquivo `.jsonl` de auditoria, ja que a Fase 3 pede
"logging detalhado para rastreamento e auditoria" (nao so observabilidade
operacional).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_AUDIT_LOG_PATH = (
    Path(__file__).resolve().parent.parent / "resultados" / "fase3" / "auditoria.jsonl"
)

_logger = logging.getLogger("fase3_assistente_medico")
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


def _audit_log_path() -> Path:
    return Path(os.getenv("FASE3_AUDIT_LOG_PATH", str(DEFAULT_AUDIT_LOG_PATH)))


def registrar_interacao(evento: str, **campos: Any) -> dict:
    """Grava um evento de auditoria (stdout em JSON + arquivo `.jsonl`) e o retorna."""
    registro = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": evento,
        **campos,
    }
    linha = json.dumps(registro, ensure_ascii=False)
    _logger.info(linha)

    path = _audit_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(linha + "\n")

    return registro
