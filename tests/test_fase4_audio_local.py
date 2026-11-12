"""Audio da Fase 4 sem nuvem: disartria, WER, provedor `local` e TTS.

Nenhum teste aqui usa rede, microfone, alto-falante ou o faster-whisper de
verdade: o modulo `faster_whisper` e substituido em `sys.modules` (mesmo
padrao de `tests/test_fase4_azure.py`) e o TTS e exercitado apenas nas
funcoes puras de montagem do WAV.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from fase4 import audio_analysis
from fase4.audio_analysis import taxa_erro_palavras
from fase4.azure_services import (
    ServicoAzureIndisponivelError,
    analisar_texto,
    status_servicos,
    transcrever_audio,
)
from fase4.config import LIMIARES_AUDIO
from fase4.data import gerar_dados_sinteticos as gerador
from fase4.data import gerar_fala_tts as tts


class AmostrasTemporarias(unittest.TestCase):
    diretorio: tempfile.TemporaryDirectory
    raiz: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.diretorio = tempfile.TemporaryDirectory()
        cls.raiz = Path(cls.diretorio.name)
        for paciente in gerador.PACIENTES_PADRAO:
            gerador.gerar_amostras_paciente(paciente, seed=42, destino_raiz=cls.raiz)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.diretorio.cleanup()


# ---------------------------------------------------------------------------
# Disartria
# ---------------------------------------------------------------------------


class PadraoDisartricoTests(AmostrasTemporarias):
    def test_voz_com_piora_dispara_padrao_disartrico(self) -> None:
        atributos = audio_analysis.extrair_atributos(self.raiz / "PAC-0006" / "consulta.wav")
        achados = audio_analysis.avaliar_alteracoes_vocais(atributos)
        tipos = {a.tipo for a in achados}
        self.assertIn("padrao_disartrico", tipos)
        achado = next(a for a in achados if a.tipo == "padrao_disartrico")
        self.assertIn("disartria", achado.descricao)
        self.assertIn("taxa de fala reduzida", achado.descricao)
        self.assertEqual(achado.severidade, "atencao")

    def test_voz_estavel_nao_dispara(self) -> None:
        atributos = audio_analysis.extrair_atributos(self.raiz / "PAC-0002" / "consulta.wav")
        tipos = {a.tipo for a in audio_analysis.avaliar_alteracoes_vocais(atributos)}
        self.assertNotIn("padrao_disartrico", tipos)
        self.assertNotIn("instabilidade_vocal", tipos)

    def test_monopitch_sozinho_nao_basta(self) -> None:
        # Voz monotona mas com taxa de fala normal: nao e padrao disartrico.
        atributos = audio_analysis.extrair_atributos(self.raiz / "PAC-0002" / "consulta.wav")
        self.assertLessEqual(atributos.f0_desvio_hz, LIMIARES_AUDIO.monopitch_f0_desvio_hz)
        self.assertGreater(atributos.taxa_fala_silabas_s, LIMIARES_AUDIO.taxa_fala_lenta)
        tipos = {a.tipo for a in audio_analysis.avaliar_alteracoes_vocais(atributos)}
        self.assertNotIn("padrao_disartrico", tipos)

    def test_fala_lenta_e_monotona_dispara_mesmo_sem_jitter(self) -> None:
        atributos = audio_analysis.extrair_atributos(self.raiz / "PAC-0002" / "consulta.wav")
        lenta = replace(atributos, taxa_fala_silabas_s=1.5)
        achados = audio_analysis.avaliar_alteracoes_vocais(lenta)
        achado = next(a for a in achados if a.tipo == "padrao_disartrico")
        self.assertIn("voz monotona", achado.descricao)
        self.assertNotIn("F0 instavel", achado.descricao)

    def test_descricao_da_instabilidade_cita_disartria(self) -> None:
        atributos = audio_analysis.extrair_atributos(self.raiz / "PAC-0006" / "consulta.wav")
        achado = next(
            a for a in audio_analysis.avaliar_alteracoes_vocais(atributos)
            if a.tipo == "instabilidade_vocal"
        )
        self.assertIn("disartria", achado.descricao)


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------


class TaxaErroPalavrasTests(unittest.TestCase):
    def test_identico_e_zero_e_ignora_acentos_e_pontuacao(self) -> None:
        self.assertEqual(taxa_erro_palavras("Estou com falta de ar.", "estou com falta de ar"), 0.0)
        self.assertEqual(taxa_erro_palavras("Não aguento, doutora!", "nao aguento doutora"), 0.0)

    def test_conta_substituicao_insercao_e_remocao(self) -> None:
        # 1 substituicao em 5 palavras.
        self.assertAlmostEqual(taxa_erro_palavras("a b c d e", "a b x d e"), 0.2)
        # 1 insercao em 4.
        self.assertAlmostEqual(taxa_erro_palavras("a b c d", "a b c d e"), 0.25)
        # 1 remocao em 4.
        self.assertAlmostEqual(taxa_erro_palavras("a b c d", "a b d"), 0.25)

    def test_referencia_vazia(self) -> None:
        self.assertEqual(taxa_erro_palavras("", ""), 0.0)
        self.assertEqual(taxa_erro_palavras("", "algo"), 1.0)

    def test_hipotese_vazia_e_erro_total(self) -> None:
        self.assertEqual(taxa_erro_palavras("a b c", ""), 1.0)


# ---------------------------------------------------------------------------
# Provedor local (faster-whisper falso)
# ---------------------------------------------------------------------------


def _modulo_whisper_falso(segmentos, registro: dict):
    class ModeloFalso:
        def __init__(self, nome, device, compute_type, download_root):
            registro.update(nome=nome, device=device, compute_type=compute_type, download_root=download_root)

        def transcribe(self, caminho, language, beam_size, vad_filter):
            registro.update(caminho=caminho, language=language)
            return iter(segmentos), SimpleNamespace(language=language)

    return SimpleNamespace(WhisperModel=ModeloFalso)


class ProvedorLocalTests(AmostrasTemporarias):
    def _wav(self) -> Path:
        return self.raiz / "PAC-0006" / "consulta.wav"

    def test_local_devolve_segmentos_com_confianca_de_logprob(self) -> None:
        segmentos = [
            SimpleNamespace(start=0.5, end=2.0, text=" Estou muito cansada. ", avg_logprob=-0.1),
            SimpleNamespace(start=2.5, end=3.5, text=" Sem ar. ", avg_logprob=-0.7),
            SimpleNamespace(start=3.6, end=3.7, text="   ", avg_logprob=-3.0),
        ]
        registro: dict = {}
        with patch.dict("sys.modules", {"faster_whisper": _modulo_whisper_falso(segmentos, registro)}):
            resultado = transcrever_audio(self._wav(), provedor="local")
        self.assertEqual(resultado.provedor, "local")
        self.assertEqual(resultado.texto, "Estou muito cansada. Sem ar.")
        self.assertEqual(len(resultado.segmentos), 2)  # o segmento vazio e descartado
        self.assertAlmostEqual(resultado.segmentos[0].confianca, np.exp(-0.1), places=4)
        # Confianca media ponderada pela duracao (1.5 s e 1.0 s).
        esperada = (np.exp(-0.1) * 1.5 + np.exp(-0.7) * 1.0) / 2.5
        self.assertAlmostEqual(resultado.confianca, esperada, places=4)
        self.assertEqual(registro["device"], "cpu")
        self.assertEqual(registro["compute_type"], "int8")
        self.assertEqual(registro["language"], "pt")
        self.assertEqual(registro["nome"], "small")

    def test_modelo_pode_ser_trocado_por_variavel_de_ambiente(self) -> None:
        registro: dict = {}
        segmentos = [SimpleNamespace(start=0.0, end=1.0, text="ola", avg_logprob=-0.2)]
        with patch.dict("sys.modules", {"faster_whisper": _modulo_whisper_falso(segmentos, registro)}), \
                patch.dict("os.environ", {"FASE4_WHISPER_MODELO": "base"}):
            transcrever_audio(self._wav(), provedor="local")
            self.assertEqual(status_servicos()["speech_to_text_local"]["modelo"], "base")
        self.assertEqual(registro["nome"], "base")

    def test_sem_fala_reconhecida_falha_explicitamente(self) -> None:
        with patch.dict("sys.modules", {"faster_whisper": _modulo_whisper_falso([], {})}):
            with self.assertRaisesRegex(ServicoAzureIndisponivelError, "nao reconheceu"):
                transcrever_audio(self._wav(), provedor="local")

    def test_sem_biblioteca_instalada_falha_explicitamente(self) -> None:
        with patch.dict("sys.modules", {"faster_whisper": None}):
            with self.assertRaisesRegex(ServicoAzureIndisponivelError, "faster-whisper"):
                transcrever_audio(self._wav(), provedor="local")
            self.assertEqual(status_servicos()["speech_to_text_local"]["provedor_efetivo"], "indisponivel")

    def test_auto_nunca_escolhe_o_local(self) -> None:
        registro: dict = {}
        segmentos = [SimpleNamespace(start=0.0, end=1.0, text="ola", avg_logprob=-0.2)]
        with patch.dict("sys.modules", {"faster_whisper": _modulo_whisper_falso(segmentos, registro)}), \
                patch("fase4.azure_services.azure_speech_configurada", return_value=False):
            resultado = transcrever_audio(self._wav(), provedor="auto")
        self.assertEqual(resultado.provedor, "offline")
        self.assertEqual(registro, {})

    def test_analise_de_texto_local_usa_o_lexico(self) -> None:
        analise = analisar_texto("Estou sem ar e com febre.", provedor="local")
        self.assertEqual(analise.provedor, "offline")
        self.assertIn("respiratorio", analise.categorias())

    def test_pipeline_de_audio_calcula_wer_contra_a_referencia(self) -> None:
        with (self.raiz / "PAC-0006" / "consulta.transcricao.json").open(encoding="utf-8") as f:
            referencia = json.load(f)["segmentos"]
        # Reconhecedor "quase perfeito": troca uma palavra da primeira frase.
        segmentos = []
        for indice, item in enumerate(referencia):
            texto = item["texto"].replace("cansada", "cansado") if indice == 0 else item["texto"]
            segmentos.append(SimpleNamespace(start=item["inicio_s"], end=item["fim_s"], text=texto, avg_logprob=-0.05))
        with patch.dict("sys.modules", {"faster_whisper": _modulo_whisper_falso(segmentos, {})}):
            resultado = audio_analysis.analisar_consulta(self._wav(), "PAC-0006", provedor_azure="local")
        self.assertEqual(resultado.transcricao.provedor, "local")
        self.assertIsNotNone(resultado.wer_referencia)
        total = sum(len(audio_analysis._tokens_para_wer(s["texto"])) for s in referencia)
        self.assertAlmostEqual(resultado.wer_referencia, 1 / total, places=4)
        self.assertIn("taxa de erro de palavras", audio_analysis.relatorio_markdown(resultado))
        # Os termos criticos continuam vindo da transcricao real.
        self.assertTrue(resultado.analise_texto.termos_criticos)

    def test_modo_offline_nao_calcula_wer(self) -> None:
        resultado = audio_analysis.analisar_consulta(self._wav(), "PAC-0006", provedor_azure="offline")
        self.assertIsNone(resultado.wer_referencia)


# ---------------------------------------------------------------------------
# TTS: funcoes puras
# ---------------------------------------------------------------------------


class MontagemTtsTests(unittest.TestCase):
    def test_reamostragem_preserva_duracao(self) -> None:
        sinal = np.sin(np.linspace(0, 2 * np.pi * 440, 22050)).astype(np.float32)
        novo = tts.reamostrar(sinal, 22050, 16000)
        self.assertEqual(novo.size, 16000)
        self.assertIs(tts.reamostrar(sinal, 16000, 16000).dtype.type, np.float32)

    def test_apara_silencio_das_bordas(self) -> None:
        taxa = 16000
        sinal = np.concatenate([np.zeros(taxa), np.full(taxa, 0.5, dtype=np.float32), np.zeros(taxa)])
        aparado = tts.aparar_silencio(sinal, taxa, margem_s=0.0)
        self.assertEqual(aparado.size, taxa)
        self.assertEqual(tts.aparar_silencio(np.zeros(100), taxa).size, 100)

    def test_montagem_respeita_pausas_e_registra_instantes(self) -> None:
        taxa = 1000
        trechos = [np.full(500, 0.5, dtype=np.float32), np.full(250, 0.5, dtype=np.float32)]
        sinal, instantes = tts.montar_consulta(trechos, [0.4, 0.8], taxa)
        self.assertEqual(len(instantes), 2)
        self.assertAlmostEqual(instantes[0][0], tts.SILENCIO_INICIAL_S)
        self.assertAlmostEqual(instantes[0][1], tts.SILENCIO_INICIAL_S + 0.5)
        self.assertAlmostEqual(instantes[1][0], tts.SILENCIO_INICIAL_S + 0.5 + 0.4)
        esperado = int((tts.SILENCIO_INICIAL_S + 0.5 + 0.4 + 0.25 + 0.8) * taxa)
        self.assertEqual(sinal.size, esperado)
        self.assertLessEqual(float(np.max(np.abs(sinal))), 0.9 + 1e-6)

    def test_pausas_da_referencia(self) -> None:
        segmentos = [
            {"inicio_s": 0.5, "fim_s": 1.5},
            {"inicio_s": 2.5, "fim_s": 3.0},   # pausa de 1.0 s
            {"inicio_s": 3.0, "fim_s": 4.0},   # sem pausa -> padrao
        ]
        pausas = tts.pausas_da_referencia(segmentos)
        self.assertEqual(len(pausas), 3)
        self.assertAlmostEqual(pausas[0], 1.0)
        self.assertAlmostEqual(pausas[1], tts.PAUSA_PADRAO_S)
        self.assertAlmostEqual(pausas[2], tts.SILENCIO_FINAL_S)

    def test_escolha_de_voz_prefere_pt_br(self) -> None:
        vozes = [
            SimpleNamespace(id="TTS_MS_EN-US_ZIRA_11.0", name="Microsoft Zira Desktop", languages=["en-US"]),
            SimpleNamespace(id="TTS_MS_PT-BR_MARIA_11.0", name="Microsoft Maria Desktop", languages=["pt-BR"]),
        ]
        engine = SimpleNamespace(getProperty=lambda _nome: vozes)
        self.assertEqual(tts.escolher_voz_pyttsx3(engine).name, "Microsoft Maria Desktop")
        self.assertEqual(tts.escolher_voz_pyttsx3(engine, preferida="zira").name, "Microsoft Zira Desktop")
        with self.assertRaises(tts.SinteseIndisponivelError):
            tts.escolher_voz_pyttsx3(engine, preferida="inexistente")
        so_ingles = SimpleNamespace(getProperty=lambda _nome: vozes[:1])
        self.assertIsNone(tts.escolher_voz_pyttsx3(so_ingles))

    def test_wav_gravado_e_lido_de_volta(self) -> None:
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "x.wav"
            sinal = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
            tts.salvar_wav(sinal, caminho, 16000)
            lido, taxa = tts.ler_wav_mono(caminho)
        self.assertEqual(taxa, 16000)
        self.assertEqual(lido.size, 1600)
        self.assertAlmostEqual(float(lido[-1]), 0.5, places=3)


if __name__ == "__main__":
    unittest.main()
