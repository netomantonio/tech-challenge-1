"""Acesso a base de dados estruturada de prontuarios (mock de EHR) da Fase 3.

Usa SQLite (biblioteca padrao) semeado a partir de
``fase3/data/pacientes_sinteticos.json`` — todos os pacientes sao ficticios,
identificados apenas por codigo interno (``PAC-000x``), nunca por nome.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from langchain_core.tools import tool

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_PACIENTES_JSON = DATA_DIR / "pacientes_sinteticos.json"
DEFAULT_DB_PATH = DATA_DIR / "prontuarios.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pacientes (
    paciente_id TEXT PRIMARY KEY,
    idade INTEGER,
    sexo TEXT,
    diagnostico TEXT,
    estagio TEXT,
    exames_pendentes TEXT,
    exames_realizados TEXT,
    alertas_ativos TEXT,
    observacoes TEXT
);
"""


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    # `sqlite3.Connection` como context manager so cuida de commit/rollback,
    # nao fecha a conexao — por isso o `close()` explicito no `finally`
    # (necessario no Windows, onde um arquivo com conexao aberta nao pode
    # ser removido, como em `tempfile.TemporaryDirectory.cleanup()`).
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            conn.execute(_SCHEMA)
    finally:
        conn.close()


def seed_db(db_path: Path = DEFAULT_DB_PATH, pacientes_json: Path = DEFAULT_PACIENTES_JSON) -> int:
    """(Re)popula a tabela de pacientes a partir do JSON sintetico. Retorna quantos foram inseridos."""
    init_db(db_path)
    pacientes = json.loads(pacientes_json.read_text(encoding="utf-8"))
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            conn.execute("DELETE FROM pacientes")
            conn.executemany(
                """
                INSERT INTO pacientes
                    (paciente_id, idade, sexo, diagnostico, estagio,
                     exames_pendentes, exames_realizados, alertas_ativos, observacoes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        p["paciente_id"],
                        p["idade"],
                        p["sexo"],
                        p["diagnostico"],
                        p["estagio"],
                        json.dumps(p["exames_pendentes"], ensure_ascii=False),
                        json.dumps(p["exames_realizados"], ensure_ascii=False),
                        json.dumps(p["alertas_ativos"], ensure_ascii=False),
                        p["observacoes"],
                    )
                    for p in pacientes
                ],
            )
    finally:
        conn.close()
    return len(pacientes)


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    if not db_path.exists():
        seed_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def get_paciente(paciente_id: str, db_path: Path = DEFAULT_DB_PATH) -> Optional[dict]:
    with _connect(db_path) as conn:
        linha = conn.execute(
            "SELECT * FROM pacientes WHERE paciente_id = ?", (paciente_id,)
        ).fetchone()
    if linha is None:
        return None
    registro = dict(linha)
    for campo in ("exames_pendentes", "exames_realizados", "alertas_ativos"):
        registro[campo] = json.loads(registro[campo])
    return registro


def get_exames_pendentes(paciente_id: str, db_path: Path = DEFAULT_DB_PATH) -> list[str]:
    paciente = get_paciente(paciente_id, db_path)
    return paciente["exames_pendentes"] if paciente else []


def get_alertas_ativos(paciente_id: str, db_path: Path = DEFAULT_DB_PATH) -> list[str]:
    paciente = get_paciente(paciente_id, db_path)
    return paciente["alertas_ativos"] if paciente else []


@tool
def consulta_prontuario_tool(paciente_id: str) -> str:
    """Consulta o prontuario estruturado de um paciente pelo codigo interno (ex.: PAC-0001)."""
    paciente = get_paciente(paciente_id)
    if paciente is None:
        return f"Nenhum prontuario encontrado para o codigo {paciente_id}."
    return json.dumps(paciente, ensure_ascii=False)
