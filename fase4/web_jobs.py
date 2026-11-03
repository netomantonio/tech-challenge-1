"""Jobs em segundo plano da interface web da Fase 4.

Os backends reais (MediaPipe, YOLOv8, faster-whisper) e o download do
PhysioNet levam minutos. O `JobManager` roda uma funcao Python numa thread
daemon, um job por vez, e expoe um retrato do estado para a interface fazer
polling - o mesmo vocabulario de estados do `TrainingJobManager` da Fase 3
(`aguardando`, `executando`, `cancelando`, `concluido`, `falhou`, `cancelado`).

O cancelamento e cooperativo: a funcao recebe `cancelado()` e decide quando
checar. Uma passada do MediaPipe sobre um video nao e interrompida no meio;
o job para entre etapas.
"""

from __future__ import annotations

import threading
import traceback
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable

Informar = Callable[..., None]
Cancelado = Callable[[], bool]
FuncaoJob = Callable[[Informar, Cancelado], Any]

ESTADOS_ATIVOS = {"aguardando", "executando", "cancelando"}


class JobOcupadoError(RuntimeError):
    """Ja existe um job em andamento (ou nao ha job para cancelar)."""


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    """Executa uma funcao por vez numa thread e guarda progresso, logs e resultado."""

    def __init__(self, maximo_logs: int = 200) -> None:
        self._lock = threading.RLock()
        self._job: dict[str, Any] | None = None
        self._thread: threading.Thread | None = None
        self._cancelar = threading.Event()
        self._contador = 0
        self._maximo_logs = maximo_logs

    # --- consulta -----------------------------------------------------------

    @property
    def ativo(self) -> bool:
        with self._lock:
            return bool(self._job and self._job["estado"] in ESTADOS_ATIVOS)

    def snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._job:
                return None
            retrato = {chave: valor for chave, valor in self._job.items() if chave != "_logs"}
            retrato["logs"] = list(self._job["_logs"])
            return retrato

    # --- controle -----------------------------------------------------------

    def iniciar(
        self,
        tipo: str,
        titulo: str,
        funcao: FuncaoJob,
        metadados: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self.ativo:
                raise JobOcupadoError("Ja existe uma execucao em andamento. Aguarde ou cancele.")
            self._contador += 1
            self._cancelar = threading.Event()
            self._job = {
                "id": f"job-{self._contador:04d}",
                "tipo": tipo,
                "titulo": titulo,
                "estado": "aguardando",
                "progresso": 0.0,
                "etapa": "Na fila",
                "metadados": dict(metadados or {}),
                "criado_em": _agora(),
                "iniciado_em": None,
                "finalizado_em": None,
                "resultado": None,
                "erro": None,
                "_logs": deque(maxlen=self._maximo_logs),
            }
            self._thread = threading.Thread(
                target=self._executar, args=(funcao,), name=self._job["id"], daemon=True
            )
            self._thread.start()
            return self.snapshot() or {}

    def cancelar(self) -> dict[str, Any]:
        with self._lock:
            if not self.ativo or not self._job:
                raise JobOcupadoError("Nao ha execucao em andamento para cancelar.")
            self._job["estado"] = "cancelando"
            self._job["etapa"] = "Cancelamento solicitado; aguardando a etapa atual terminar"
            self._registrar("cancelamento solicitado pela interface")
            self._cancelar.set()
            return self.snapshot() or {}

    def aguardar(self, timeout_s: float | None = None) -> bool:
        """Bloqueia ate o job terminar (uso em testes). True se terminou."""
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout_s)
        return not thread.is_alive()

    # --- execucao -----------------------------------------------------------

    def _registrar(self, mensagem: str) -> None:
        if self._job is not None:
            marca = datetime.now().strftime("%H:%M:%S")
            self._job["_logs"].append(f"[{marca}] {mensagem}")

    def _informar(self, progresso: float | None = None, etapa: str | None = None, log: str | None = None) -> None:
        with self._lock:
            if self._job is None:
                return
            if progresso is not None:
                self._job["progresso"] = float(min(1.0, max(0.0, progresso)))
            if etapa:
                self._job["etapa"] = etapa
                self._registrar(etapa)
            if log:
                self._registrar(log)

    def _executar(self, funcao: FuncaoJob) -> None:
        with self._lock:
            if self._job is None:
                return
            self._job["estado"] = "executando"
            self._job["iniciado_em"] = _agora()
            self._job["etapa"] = "Iniciando"
            self._registrar(f"job {self._job['id']} iniciado: {self._job['titulo']}")

        try:
            resultado = funcao(self._informar, self._cancelar.is_set)
        except Exception as erro:  # noqa: BLE001 - o erro vira estado do job
            with self._lock:
                if self._job is not None:
                    self._job["estado"] = "falhou"
                    self._job["erro"] = f"{type(erro).__name__}: {erro}"
                    self._job["etapa"] = "Falhou"
                    for linha in traceback.format_exc().strip().splitlines()[-6:]:
                        self._registrar(linha)
                    self._job["finalizado_em"] = _agora()
            return

        with self._lock:
            if self._job is None:
                return
            cancelado = self._cancelar.is_set()
            self._job["estado"] = "cancelado" if cancelado else "concluido"
            self._job["etapa"] = "Cancelado" if cancelado else "Concluido"
            self._job["progresso"] = self._job["progresso"] if cancelado else 1.0
            self._job["resultado"] = resultado
            self._job["finalizado_em"] = _agora()
            self._registrar("job encerrado" if cancelado else "job concluido")
