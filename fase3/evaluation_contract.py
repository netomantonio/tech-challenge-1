"""Fingerprint do contrato usado para calibrar e promover adapters."""

from __future__ import annotations

import hashlib
from pathlib import Path

from fase3.prompting import SYSTEM_PROMPT_CLINICO, USER_PROMPT_TEMPLATE

ROOT = Path(__file__).resolve().parent.parent
EVALUATION_CASES_PATH = ROOT / "fase3" / "data" / "assistant_evaluation_cases.json"
VALIDATION_DATA_PATH = ROOT / "fase3" / "data" / "finetuning_val.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_evaluation_contract() -> dict[str, str | int]:
    return {
        "schema_version": 1,
        "prompt_sha256": hashlib.sha256(
            (SYSTEM_PROMPT_CLINICO + "\n" + USER_PROMPT_TEMPLATE).encode("utf-8")
        ).hexdigest(),
        "validation_data_sha256": _sha256(VALIDATION_DATA_PATH),
        "evaluation_cases_sha256": _sha256(EVALUATION_CASES_PATH),
    }


def evaluation_contract_matches(calibration: dict | None) -> bool:
    return bool(
        calibration
        and calibration.get("evaluation_contract") == current_evaluation_contract()
    )
