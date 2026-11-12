"""Receptor de demonstracao para o webhook de alertas da Fase 4.

Sobe um servidor HTTP minimo que imprime cada alerta recebido - faz o papel
do painel da equipe (ou do sistema hospitalar) na gravacao do video. Em um
terminal:

    python -m fase4.receptor_webhook_demo --porta 8765

Em outro, com o webhook apontado para ele:

    $env:FASE4_WEBHOOK_URL = "http://127.0.0.1:8765/alertas"
    python -m fase4.cli_demo --paciente-id PAC-0006 --tempo-real --atraso 0.01

Cada POST e respondido com 202 e gravado tambem em `--arquivo`, se informado.
Nao ha autenticacao: e um receptor local de demonstracao, nao um endpoint de
producao.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class ReceptorAlertas(BaseHTTPRequestHandler):
    arquivo: Path | None = None
    recebidos = 0
    silencioso = False

    def do_POST(self) -> None:  # noqa: N802 - nome exigido pela http.server
        tamanho = int(self.headers.get("Content-Length", "0") or 0)
        corpo = self.rfile.read(tamanho) if tamanho else b""
        try:
            alerta = json.loads(corpo.decode("utf-8")) if corpo else {}
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            return

        type(self).recebidos += 1
        if not self.silencioso:
            agora = datetime.now().strftime("%H:%M:%S")
            print(
                f"[{agora}] #{type(self).recebidos} {alerta.get('prioridade', '?').upper():8s} "
                f"{alerta.get('paciente_id', '?')}  {alerta.get('titulo', '')}"
            )
            print(f"           -> {alerta.get('destino', '')}")
            if alerta.get("acao_recomendada"):
                print(f"           conduta: {alerta['acao_recomendada']}")

        if self.arquivo is not None:
            with self.arquivo.open("a", encoding="utf-8") as f:
                f.write(json.dumps(alerta, ensure_ascii=False) + "\n")

        resposta = json.dumps({"recebido": True, "alerta_id": alerta.get("alerta_id")}).encode("utf-8")
        self.send_response(202)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resposta)))
        self.end_headers()
        self.wfile.write(resposta)

    def do_GET(self) -> None:  # noqa: N802
        corpo = json.dumps({"servico": "receptor de alertas fase4", "recebidos": type(self).recebidos}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def log_message(self, formato: str, *args) -> None:  # silencia o log padrao
        return


def criar_servidor(
    host: str, porta: int, arquivo: Path | None = None, silencioso: bool = False
) -> ThreadingHTTPServer:
    ReceptorAlertas.arquivo = arquivo
    ReceptorAlertas.recebidos = 0
    ReceptorAlertas.silencioso = silencioso
    return ThreadingHTTPServer((host, porta), ReceptorAlertas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--porta", type=int, default=8765)
    parser.add_argument("--arquivo", type=Path, default=None, help="Grava os alertas recebidos em JSONL.")
    args = parser.parse_args()

    servidor = criar_servidor(args.host, args.porta, args.arquivo)
    print(f"Receptor de alertas ouvindo em http://{args.host}:{args.porta}/alertas (Ctrl+C para sair)")
    print(f'Aponte o pipeline com: $env:FASE4_WEBHOOK_URL = "http://{args.host}:{args.porta}/alertas"')
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print(f"\n{ReceptorAlertas.recebidos} alertas recebidos.")
    finally:
        servidor.server_close()


if __name__ == "__main__":
    main()
