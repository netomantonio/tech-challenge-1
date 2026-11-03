"""Interface web da Fase 4: API, SSE, jobs, uploads e seguranca de caminho.

Tudo roda com `TestClient`, sem rede e sem os pacotes opcionais. As amostras
sao geradas num diretorio temporario e todas as pastas da aplicacao apontam
para ele via `Deps`, para nada ser gravado em `resultados/fase4/`.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from fase4.data import gerar_dados_sinteticos as gerador
from fase4.markdown_simples import markdown_para_html
from fase4.web_app import create_app
from fase4.web_jobs import JobManager, JobOcupadoError
from fase4.web_service import BarramentoAlertas, Deps


def _ler_sse(resposta, maximo: int | None = None):
    """Converte as linhas de uma resposta SSE em (evento, id, dados)."""
    eventos = []
    atual: dict[str, str] = {}
    for linha in resposta.iter_lines():
        if isinstance(linha, bytes):
            linha = linha.decode("utf-8")
        if linha == "":
            if "data" in atual:
                eventos.append((atual.get("event", "message"), atual.get("id"), json.loads(atual["data"])))
                if maximo is not None and len(eventos) >= maximo:
                    return eventos
            atual = {}
            continue
        if linha.startswith(":"):
            continue
        chave, _, valor = linha.partition(":")
        atual[chave] = valor.strip()
    return eventos


class BaseWeb(unittest.TestCase):
    diretorio: tempfile.TemporaryDirectory
    raiz: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.diretorio = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.raiz = Path(cls.diretorio.name)
        cls.amostras = cls.raiz / "amostras"
        cls.enviados = cls.raiz / "enviados"
        cls.physionet = cls.raiz / "physionet"
        cls.resultados = cls.raiz / "resultados"
        for pasta in (cls.amostras, cls.enviados, cls.physionet, cls.resultados):
            pasta.mkdir(parents=True, exist_ok=True)
        for paciente in gerador.PACIENTES_PADRAO:
            gerador.gerar_amostras_paciente(paciente, seed=42, destino_raiz=cls.amostras)

        # Registro "PhysioNet" falso: 200 minutos do paciente estavel, sem temperatura.
        registro = cls.physionet / "p099999-2112-01-01-00-00n"
        registro.mkdir()
        df = pd.read_csv(cls.amostras / "PAC-0002" / "sinais_vitais.csv").drop(columns=["temperatura"]).head(200)
        df.to_csv(registro / "sinais_vitais.csv", index=False)
        (registro / "fonte.json").write_text(json.dumps({
            "dataset": "teste", "slug": "t", "versao": "0", "registro": registro.name, "licenca": "ODbL",
            "canais": {"frequencia_cardiaca": {"fonte": "HR", "unidade": "bpm", "ffill_max_min": 5}},
            "canais_ausentes": ["temperatura"], "fs_hz": 1, "amostras_lidas": 12000,
            "minutos_exportados": 200, "minutos_descartados": 0,
        }), encoding="utf-8")

        cls.ambiente = patch.dict("os.environ", {
            "FASE4_AUDIT_LOG_PATH": str(cls.raiz / "auditoria.jsonl"),
            "FASE4_LOG_SILENCIOSO": "1",
            "FASE4_WEBHOOK_URL": "",
            "FASE4_FILA_ALERTAS": "",
        })
        cls.ambiente.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.ambiente.stop()
        cls.diretorio.cleanup()

    def deps(self, **extras) -> Deps:
        return Deps(
            dir_amostras=self.amostras, dir_enviados=self.enviados, dir_physionet=self.physionet,
            dir_resultados=self.resultados, sse_keepalive_s=0.5, ffmpeg=None, **extras,
        )

    def cliente(self, **extras) -> TestClient:
        return TestClient(create_app(self.deps(**extras)))


# ---------------------------------------------------------------------------
# Esqueleto, listagem e seguranca
# ---------------------------------------------------------------------------


class EsqueletoTests(BaseWeb):
    def test_interface_status_e_cabecalhos_de_seguranca(self) -> None:
        client = self.cliente()
        pagina = client.get("/")
        self.assertEqual(pagina.status_code, 200)
        self.assertIn("Monitoramento multimodal", pagina.text)
        self.assertIn("default-src 'self'", pagina.headers["content-security-policy"])
        self.assertEqual(pagina.headers["x-content-type-options"], "nosniff")

        status = client.get("/api/status").json()
        self.assertIn("speech_to_text", status["servicos"])
        self.assertIn("mediapipe", status["backends"])
        self.assertEqual(len(status["faixas"]), 6)
        self.assertIsNone(status["job"])

    def test_lista_pacientes_das_tres_origens(self) -> None:
        pacientes = {p["paciente_id"]: p for p in self.cliente().get("/api/pacientes").json()}
        self.assertEqual(pacientes["PAC-0006"]["origem"], "sinteticos")
        self.assertIn("video", pacientes["PAC-0006"]["modalidades"])
        self.assertIn("consulta.wav", pacientes["PAC-0006"]["audios"])
        self.assertNotIn("video", pacientes["PAC-0002"]["modalidades"])
        self.assertEqual(pacientes["p099999-2112-01-01-00-00n"]["origem"], "physionet")
        self.assertFalse(pacientes["PAC-0006"]["resultado_disponivel"])

        detalhe = self.cliente().get("/api/pacientes/PAC-0006").json()
        self.assertEqual(detalhe["video"]["largura"], gerador.LARGURA_VIDEO)
        self.assertIn("sinais_vitais", detalhe["eventos_plantados"])
        detalhe_pn = self.cliente().get("/api/pacientes/p099999-2112-01-01-00-00n").json()
        self.assertEqual(detalhe_pn["fonte"]["licenca"], "ODbL")

    def test_rejeita_id_invalido_e_path_traversal(self) -> None:
        client = self.cliente()
        self.assertEqual(client.get("/api/pacientes/PAC 0006").status_code, 422)
        # O roteador decodifica %2F e nao casa a rota (404); sem decodificar, o id e rejeitado (422).
        self.assertIn(client.get("/api/pacientes/..%2F..%2Fconfig").status_code, (404, 422))
        self.assertEqual(client.get("/api/pacientes/..config").status_code, 422)
        self.assertEqual(client.get("/api/pacientes/PAC-9999").status_code, 404)
        self.assertEqual(client.get("/api/pacientes/PAC-0006/midia/..%5Cconfig.py").status_code, 404)
        self.assertEqual(client.get("/api/pacientes/PAC-0006/midia/sinais_vitais.csv").status_code, 404)
        self.assertEqual(client.get("/api/pacientes/PAC-0006/relatorios/..%2Fx").status_code, 404)


class MarkdownTests(unittest.TestCase):
    def test_converte_o_subconjunto_dos_relatorios(self) -> None:
        texto = (
            "# Titulo\n\nParagrafo com **negrito**, `codigo` e *enfase*.  \nSegunda linha.\n\n"
            "## Tabela\n\n| Canal | Valor |\n| --- | ---: |\n| fc | 93.5 |\n\n"
            "- item um\n- item dois\n\n> aviso\n\n```\nx = 1 < 2\n```\n\nVeja https://physionet.org/x?a=1&b=2 agora.\n"
            "<script>alert(1)</script>\n"
        )
        html = markdown_para_html(texto)
        self.assertIn("<h1>Titulo</h1>", html)
        self.assertIn("<strong>negrito</strong>", html)
        self.assertIn("<code>codigo</code>", html)
        self.assertIn("<em>enfase</em>", html)
        self.assertIn("<br>", html)
        self.assertIn('<th class="num">Valor</th>', html)
        self.assertIn('<td class="num">93.5</td>', html)
        self.assertIn("<ul><li>item um</li><li>item dois</li></ul>", html)
        self.assertIn("<blockquote><p>aviso</p></blockquote>", html)
        self.assertIn("<pre><code>x = 1 &lt; 2</code></pre>", html)
        self.assertIn('<a href="https://physionet.org/x?a=1&amp;b=2" rel="noopener" target="_blank">', html)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)


# ---------------------------------------------------------------------------
# Processamento, relatorios e midia
# ---------------------------------------------------------------------------


class ProcessamentoTests(BaseWeb):
    def test_processa_sintetico_offline_sincrono(self) -> None:
        client = self.cliente()
        resposta = client.post(
            "/api/pacientes/PAC-0006/processar",
            json={"provedor_azure": "offline", "gravar_video_anotado": False},
        )
        self.assertEqual(resposta.status_code, 200, resposta.text)
        corpo = resposta.json()
        self.assertEqual(corpo["modo"], "sincrono")
        resultado = corpo["resultado"]
        self.assertEqual(resultado["origem"], "sinteticos")
        self.assertGreater(resultado["fusao"]["risco"], 0)
        self.assertTrue(resultado["alertas"])
        for alerta in resultado["alertas"]:
            self.assertEqual(alerta["canais_entrega"], ["fila_jsonl", "painel_web"])
        self.assertIn("relatorio_monitoramento", resultado["relatorios"])
        self.assertEqual(resultado["audio"]["transcricao"]["provedor"], "offline")
        self.assertIsNotNone(resultado["video"])
        self.assertIsNone(resultado["midias"]["video_anotado"])
        self.assertTrue(resultado["midias"]["audio"].endswith("consulta.wav"))

        gravado = client.get("/api/pacientes/PAC-0006/resultado").json()
        self.assertEqual(gravado["total_alertas"], resultado["total_alertas"])
        self.assertTrue((self.resultados / "fila_alertas.jsonl").exists())

        alertas = client.get("/api/alertas").json()
        self.assertEqual(len(alertas["alertas"]), len(resultado["alertas"]))
        self.assertTrue(all(a["origem"] == "canal_memoria" for a in alertas["alertas"]))
        self.assertEqual(alertas["seq_atual"], len(resultado["alertas"]))

        listagem = {p["paciente_id"]: p for p in client.get("/api/pacientes").json()}
        self.assertTrue(listagem["PAC-0006"]["resultado_disponivel"])

    def test_relatorio_renderizado_e_404_antes_de_processar(self) -> None:
        client = self.cliente()
        self.assertEqual(client.get("/api/pacientes/PAC-0002/resultado").status_code, 404)
        self.assertEqual(client.get("/api/pacientes/PAC-0002/relatorios/relatorio_audio").status_code, 404)
        client.post("/api/pacientes/PAC-0002/processar", json={"provedor_azure": "offline"})
        relatorio = client.get("/api/pacientes/PAC-0002/relatorios/relatorio_monitoramento").json()
        self.assertIn("<h1>", relatorio["html"])
        self.assertIn("<table>", relatorio["html"])
        self.assertIn("PAC-0002", relatorio["markdown"])
        self.assertEqual(client.get("/api/pacientes/PAC-0002/relatorios/relatorio_video").status_code, 404)

    def test_midia_com_range(self) -> None:
        client = self.cliente()
        resposta = client.get("/api/pacientes/PAC-0006/midia/consulta.wav", headers={"Range": "bytes=0-99"})
        self.assertEqual(resposta.status_code, 206)
        self.assertEqual(resposta.headers["content-length"], "100")
        self.assertEqual(resposta.headers["accept-ranges"], "bytes")
        self.assertTrue(resposta.headers["content-type"].startswith("audio/wav"))
        self.assertEqual(resposta.content[:4], b"RIFF")
        self.assertEqual(client.get("/api/pacientes/PAC-0006/midia/fisioterapia.mp4").status_code, 200)
        self.assertEqual(client.get("/api/pacientes/PAC-0002/midia/fisioterapia.mp4").status_code, 404)

    def test_quadro_jpeg_info_e_roi(self) -> None:
        client = self.cliente()
        info = client.get("/api/pacientes/PAC-0006/video/info").json()
        self.assertEqual((info["largura"], info["altura"]), (gerador.LARGURA_VIDEO, gerador.ALTURA_VIDEO))
        self.assertAlmostEqual(info["fps"], gerador.FPS_VIDEO, places=2)
        self.assertFalse(info["reproduzivel_no_navegador"])

        quadro = client.get("/api/pacientes/PAC-0006/quadro.jpg", params={"t": 5})
        self.assertEqual(quadro.status_code, 200)
        self.assertEqual(quadro.headers["content-type"], "image/jpeg")
        self.assertEqual(quadro.content[:2], b"\xff\xd8")
        self.assertEqual(quadro.headers["x-quadro-largura"], str(gerador.LARGURA_VIDEO))
        self.assertEqual(client.get("/api/pacientes/PAC-0006/quadro.jpg", params={"t": 999}).status_code, 422)
        self.assertEqual(client.get("/api/pacientes/PAC-0002/quadro.jpg").status_code, 404)

        invalida = client.post(
            "/api/pacientes/PAC-0006/processar",
            json={"provedor_azure": "offline", "area_critica": [10, 10, 5, 5], "gravar_video_anotado": False},
        )
        self.assertEqual(invalida.status_code, 422)
        fora = client.post(
            "/api/pacientes/PAC-0006/processar",
            json={"provedor_azure": "offline", "area_critica": [0, 0, 9999, 10], "gravar_video_anotado": False},
        )
        self.assertEqual(fora.status_code, 422)
        valida = client.post(
            "/api/pacientes/PAC-0006/processar",
            json={"provedor_azure": "offline", "area_critica": list(gerador.AREA_CRITICA_DRENO), "gravar_video_anotado": False},
        )
        self.assertEqual(valida.status_code, 200, valida.text)
        self.assertEqual(valida.json()["resultado"]["video"]["origem_area_critica"], "manual")


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


class JobManagerTests(unittest.TestCase):
    def test_job_lento_com_progresso_e_cancelamento(self) -> None:
        jobs = JobManager()
        liberar = threading.Event()

        def tarefa(informar, cancelado):
            informar(progresso=0.3, etapa="meio do caminho")
            liberar.wait(5)
            return {"ok": True, "cancelado": cancelado()}

        job = jobs.iniciar("teste", "Tarefa lenta", tarefa, metadados={"x": 1})
        self.assertTrue(jobs.ativo)
        self.assertEqual(job["metadados"], {"x": 1})
        with self.assertRaises(JobOcupadoError):
            jobs.iniciar("teste", "Outra", tarefa)
        for _ in range(50):
            if jobs.snapshot()["progresso"] >= 0.3:
                break
            threading.Event().wait(0.02)
        retrato = jobs.snapshot()
        self.assertEqual(retrato["estado"], "executando")
        self.assertEqual(retrato["etapa"], "meio do caminho")
        self.assertTrue(any("meio do caminho" in l for l in retrato["logs"]))

        liberar.set()
        self.assertTrue(jobs.aguardar(5))
        retrato = jobs.snapshot()
        self.assertEqual(retrato["estado"], "concluido")
        self.assertEqual(retrato["resultado"], {"ok": True, "cancelado": False})
        self.assertEqual(retrato["progresso"], 1.0)
        self.assertFalse(jobs.ativo)

        # Cancelamento cooperativo.
        liberar.clear()
        jobs.iniciar("teste", "Cancelavel", tarefa)
        jobs.cancelar()
        self.assertEqual(jobs.snapshot()["estado"], "cancelando")
        liberar.set()
        self.assertTrue(jobs.aguardar(5))
        self.assertEqual(jobs.snapshot()["estado"], "cancelado")
        self.assertTrue(jobs.snapshot()["resultado"]["cancelado"])
        with self.assertRaises(JobOcupadoError):
            jobs.cancelar()

    def test_excecao_vira_estado_falhou(self) -> None:
        jobs = JobManager()

        def quebra(informar, cancelado):
            raise RuntimeError("deu ruim")

        jobs.iniciar("teste", "Quebra", quebra)
        self.assertTrue(jobs.aguardar(5))
        retrato = jobs.snapshot()
        self.assertEqual(retrato["estado"], "falhou")
        self.assertIn("deu ruim", retrato["erro"])
        self.assertTrue(any("RuntimeError" in l for l in retrato["logs"]))


class JobsNaApiTests(BaseWeb):
    def test_backend_real_vira_job_e_409_quando_ocupado(self) -> None:
        liberar = threading.Event()
        chamadas: list[dict] = []

        def processar_falso(paciente_id, **kwargs):
            chamadas.append({"paciente_id": paciente_id, **kwargs})
            liberar.wait(5)
            from fase4.pipeline import processar_paciente
            return processar_paciente(
                paciente_id, raiz_amostras=kwargs["raiz_amostras"], dir_saida=kwargs["dir_saida"],
                provedor_azure="offline", gravar_video_anotado=False, canais=kwargs["canais"],
            )

        with patch("fase4.web_service.backends_disponiveis", return_value={
            "cv2": True, "mediapipe": True, "yolov8": True, "whisper": True, "wfdb": True, "ffmpeg": False,
        }), patch("fase4.web_app.backends_disponiveis", return_value={
            "cv2": True, "mediapipe": True, "yolov8": True, "whisper": True, "wfdb": True, "ffmpeg": False,
        }):
            client = self.cliente(processar=processar_falso)
            resposta = client.post("/api/pacientes/PAC-0006/processar", json={"backend_pose": "mediapipe"})
            self.assertEqual(resposta.status_code, 202, resposta.text)
            self.assertEqual(resposta.json()["modo"], "job")
            self.assertEqual(client.post("/api/pacientes/PAC-0006/processar", json={"backend_pose": "mediapipe"}).status_code, 409)
            self.assertEqual(client.post("/api/pacientes/PAC-0002/processar", json={"provedor_azure": "offline"}).status_code, 409)
            self.assertEqual(client.get("/api/jobs/atual").json()["job"]["estado"], "executando")
            liberar.set()
            self.assertTrue(client.app.state.deps.jobs.aguardar(20))
            job = client.get("/api/jobs/atual").json()["job"]
        self.assertEqual(job["estado"], "concluido", job.get("erro"))
        self.assertEqual(chamadas[0]["backend_pose"], "mediapipe")
        self.assertEqual(job["resultado"]["paciente_id"], "PAC-0006")
        self.assertTrue(job["resultado"]["alertas"])

    def test_backend_ausente_e_503_antes_de_criar_job(self) -> None:
        with patch("fase4.web_app.backends_disponiveis", return_value={
            "cv2": True, "mediapipe": False, "yolov8": True, "whisper": False, "wfdb": True, "ffmpeg": False,
        }):
            client = self.cliente()
            resposta = client.post("/api/pacientes/PAC-0006/processar", json={"backend_pose": "mediapipe"})
            self.assertEqual(resposta.status_code, 503)
            self.assertIn("mediapipe", resposta.json()["detail"])
            self.assertIsNone(client.get("/api/jobs/atual").json()["job"])
            self.assertEqual(client.post("/api/jobs/cancelar").status_code, 409)

    def test_physionet_processa_como_job_com_relatorio(self) -> None:
        client = self.cliente()
        resposta = client.post("/api/pacientes/p099999-2112-01-01-00-00n/processar", json={"provedor_azure": "offline"})
        self.assertEqual(resposta.status_code, 202, resposta.text)
        self.assertTrue(client.app.state.deps.jobs.aguardar(30))
        job = client.get("/api/jobs/atual").json()["job"]
        self.assertEqual(job["estado"], "concluido", job.get("erro"))
        self.assertIn("relatorio_physionet", job["resultado"]["relatorios"])
        relatorio = client.get("/api/pacientes/p099999-2112-01-01-00-00n/relatorios/relatorio_physionet").json()
        self.assertIn("Validação", relatorio["markdown"])

    def test_download_physionet_usa_funcao_injetada(self) -> None:
        def baixar_falso(registro, destino, minutos):
            nome = registro.rsplit("/", 1)[-1]
            pasta = destino / nome
            pasta.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.physionet / "p099999-2112-01-01-00-00n" / "sinais_vitais.csv", pasta / "sinais_vitais.csv")
            return {"minutos_exportados": 200}

        client = self.cliente(baixar_physionet=baixar_falso)
        resposta = client.post("/api/physionet/baixar", json={"registro": "p01/p010101/p010101-2100-01-01-00-00n", "minutos": 100})
        self.assertEqual(resposta.status_code, 202, resposta.text)
        self.assertTrue(client.app.state.deps.jobs.aguardar(30))
        job = client.get("/api/jobs/atual").json()["job"]
        self.assertEqual(job["estado"], "concluido", job.get("erro"))
        self.assertEqual(job["resultado"]["download"], {"minutos_exportados": 200})
        self.assertIn("p010101-2100-01-01-00-00n", {p["paciente_id"] for p in client.get("/api/pacientes").json()})


# ---------------------------------------------------------------------------
# Streams: tempo real, alertas e webhook
# ---------------------------------------------------------------------------


class StreamsTests(BaseWeb):
    def test_tempo_real_emite_todas_as_leituras(self) -> None:
        client = self.cliente()
        with client.stream(
            "GET", "/api/pacientes/PAC-0006/tempo-real", params={"ms_por_minuto": 0, "limite_minutos": 125}
        ) as resposta:
            self.assertEqual(resposta.status_code, 200)
            self.assertTrue(resposta.headers["content-type"].startswith("text/event-stream"))
            eventos = _ler_sse(resposta)
        self.assertEqual(eventos[0][0], "inicio")
        inicio = eventos[0][2]
        self.assertEqual(inicio["total_minutos"], 720)
        self.assertEqual(len(inicio["faixas"]), 6)
        leituras = [e for e in eventos if e[0] == "leitura"]
        self.assertEqual(len(leituras), 125)
        self.assertEqual(leituras[0][2]["minuto"], 0)
        self.assertEqual(set(leituras[0][2]["valores"]), set(inicio["canais"]))
        self.assertEqual(leituras[0][2]["achados"], [])
        self.assertEqual(eventos[-1][0], "fim")
        self.assertEqual(eventos[-1][2]["minutos"], 125)

    def test_tempo_real_gera_alertas_e_entrega_no_painel(self) -> None:
        client = self.cliente()
        with client.stream(
            "GET", "/api/pacientes/PAC-0006/tempo-real", params={"ms_por_minuto": 0, "multivariado": 0}
        ) as resposta:
            eventos = _ler_sse(resposta)
        alertas = [a for e in eventos if e[0] == "leitura" for a in e[2]["alertas"]]
        self.assertGreaterEqual(len(alertas), 4)
        self.assertTrue(all("painel_web" in a["canais_entrega"] for a in alertas))
        self.assertFalse(any(a["tipo"] == "anomalia_multivariada" for e in eventos if e[0] == "leitura" for a in e[2]["achados"]))
        fim = eventos[-1][2]
        self.assertEqual(fim["total_alertas"], len(alertas))
        historico = client.get("/api/alertas").json()["alertas"]
        self.assertEqual(len(historico), len(alertas))

    def test_tempo_real_sem_serie_e_404(self) -> None:
        pasta = self.enviados / "PAC-SEM-SERIE"
        pasta.mkdir(exist_ok=True)
        try:
            self.assertEqual(self.cliente().get("/api/pacientes/PAC-SEM-SERIE/tempo-real").status_code, 404)
        finally:
            shutil.rmtree(pasta, ignore_errors=True)

    def test_webhook_recebe_e_stream_reentrega(self) -> None:
        client = self.cliente()
        alerta = {
            "alerta_id": "ALERTA-PAC-X-001", "paciente_id": "PAC-X", "prioridade": "vermelho",
            "titulo": "Teste de webhook", "destino": "Equipe", "acao_recomendada": "Olhar", "canais_entrega": ["webhook"],
        }
        resposta = client.post("/api/webhook/alertas", json=alerta)
        self.assertEqual(resposta.status_code, 202, resposta.text)
        self.assertEqual(resposta.json()["seq"], 1)
        historico = client.get("/api/alertas").json()["alertas"]
        self.assertEqual(historico[0]["origem"], "webhook")
        self.assertEqual(historico[0]["titulo"], "Teste de webhook")

        # `maximo=1` encerra o stream apos o primeiro alerta: com o TestClient a
        # desconexao do cliente nao chega ao gerador, e o stream ficaria aberto.
        with client.stream("GET", "/api/alertas/stream", params={"desde": 0, "maximo": 1}) as stream:
            eventos = _ler_sse(stream)
        self.assertEqual(len(eventos), 1)
        self.assertEqual(eventos[0][0], "alerta")
        self.assertEqual(eventos[0][1], "1")
        self.assertEqual(eventos[0][2]["alerta_id"], "ALERTA-PAC-X-001")

        self.assertEqual(client.post("/api/webhook/alertas", json={"titulo": "sem id"}).status_code, 422)
        self.assertEqual(client.post("/api/webhook/alertas", content=b"nao e json").status_code, 422)
        remoto = TestClient(client.app, client=("192.168.1.5", 1))
        self.assertEqual(remoto.post("/api/webhook/alertas", json=alerta).status_code, 403)

    def test_barramento_marca_canais_na_ordem(self) -> None:
        barramento = BarramentoAlertas()
        from fase4.alertas import Achado, MotorDeAlertas
        motor = MotorDeAlertas(canais=[barramento])
        alertas = motor.gerar("PAC-Y", [Achado("video", "profissional_ausente", "x", "atencao", 0.6)])
        self.assertEqual(alertas[0].canais_entrega, ["painel_web"])
        self.assertEqual(barramento.historico()[0]["canais_entrega"], ["painel_web"])
        self.assertEqual(barramento.seq_atual, 1)


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------


class UploadTests(BaseWeb):
    def _put(self, client, paciente, nome, dados: bytes, **kwargs):
        return client.put(
            f"/api/pacientes/enviados/{paciente}/arquivos/{nome}", content=dados,
            headers={"Content-Length": str(len(dados))}, **kwargs,
        )

    def test_upload_roundtrip(self) -> None:
        client = self.cliente()
        criado = client.post("/api/pacientes/enviados", json={"paciente_id": "PAC-WEB1"})
        self.assertEqual(criado.status_code, 201, criado.text)
        self.assertEqual(criado.json()["origem"], "enviados")
        origem = self.amostras / "PAC-0002"
        for nome in ("sinais_vitais.csv", "prescricoes.json", "movimentacao.csv", "consulta.wav", "consulta.transcricao.json"):
            resposta = self._put(client, "PAC-WEB1", nome, (origem / nome).read_bytes())
            self.assertEqual(resposta.status_code, 201, f"{nome}: {resposta.text}")
            self.assertEqual(resposta.json()["validacao"]["tipo"], nome.rsplit(".", 1)[-1])
        self.assertEqual(self._put(client, "PAC-WEB1", "consulta.wav", (origem / "consulta.wav").read_bytes()).json()["validacao"]["taxa_hz"], 16000)

        listagem = {p["paciente_id"]: p for p in client.get("/api/pacientes").json()}
        self.assertEqual(listagem["PAC-WEB1"]["origem"], "enviados")
        self.assertIn("sinais_vitais", listagem["PAC-WEB1"]["modalidades"])

        processado = client.post("/api/pacientes/PAC-WEB1/processar", json={"provedor_azure": "offline"})
        self.assertEqual(processado.status_code, 200, processado.text)
        self.assertEqual(processado.json()["resultado"]["origem"], "enviados")
        self.assertTrue((self.resultados / "enviados" / "PAC-WEB1" / "relatorio_monitoramento.md").exists())

        self.assertEqual(client.delete("/api/pacientes/enviados/PAC-WEB1").json()["removido"], True)
        self.assertNotIn("PAC-WEB1", {p["paciente_id"] for p in client.get("/api/pacientes").json()})
        self.assertFalse((self.resultados / "enviados" / "PAC-WEB1").exists())

    def test_upload_rejeita_nome_tamanho_e_conteudo(self) -> None:
        client = self.cliente(limites_upload={"csv": 60, "json": 20 * 1024 * 1024, "wav": 1024, "mp4": 1024})
        client.post("/api/pacientes/enviados", json={"paciente_id": "PAC-WEB2"})
        try:
            self.assertEqual(client.post("/api/pacientes/enviados", json={"paciente_id": "PAC-0006"}).status_code, 409)
            self.assertEqual(client.post("/api/pacientes/enviados", json={"paciente_id": "PAC-WEB2"}).status_code, 409)
            self.assertEqual(client.post("/api/pacientes/enviados", json={"paciente_id": "../x"}).status_code, 422)
            self.assertEqual(self._put(client, "PAC-WEB2", "qualquer.exe", b"x").status_code, 404)
            self.assertEqual(self._put(client, "PAC-WEB2", "fisioterapia.mp4", b"isto nao e um mp4 de verdade").status_code, 422)
            self.assertEqual(self._put(client, "PAC-WEB2", "sinais_vitais.csv", b"timestamp,frequencia_cardiaca\n" * 5).status_code, 413)
            self.assertEqual(self._put(client, "PAC-WEB2", "prescricoes.json", b"{nao json").status_code, 422)
            self.assertEqual(self._put(client, "PAC-WEB2", "prescricoes.json", b'{"a": 1}').status_code, 422)
            self.assertEqual(self._put(client, "PAC-WEB2", "movimentacao.csv", b"timestamp,x\n1,2\n").status_code, 422)
            self.assertEqual(self._put(client, "PAC-WEB2", "consulta.wav", b"RIFFxxxxWAVEfmt ").status_code, 422)
            sem_tamanho = client.put(
                "/api/pacientes/enviados/PAC-WEB2/arquivos/prescricoes.json",
                content=iter([b"[]"]), headers={"Transfer-Encoding": "chunked"},
            )
            self.assertEqual(sem_tamanho.status_code, 411)
            self.assertEqual(self._put(client, "PAC-0006", "prescricoes.json", b"[]").status_code, 403)
            self.assertFalse(any(self.enviados.joinpath("PAC-WEB2").glob("*.parte")))
            remoto = TestClient(client.app, client=("10.0.0.9", 1))
            self.assertEqual(remoto.delete("/api/pacientes/enviados/PAC-WEB2").status_code, 403)
        finally:
            client.delete("/api/pacientes/enviados/PAC-WEB2")


if __name__ == "__main__":
    unittest.main()
