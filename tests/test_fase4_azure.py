"""Servicos Azure da Fase 4 sem chamadas de rede: SDKs simulados em sys.modules."""

import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fase4.azure_services import (
    CATEGORIA_SINTOMA_AZURE,
    LIMITE_CARACTERES_DOCUMENTO,
    PESO_SINTOMA_AZURE,
    ServicoAzureIndisponivelError,
    _transcrever_azure,
    analisar_texto,
    dividir_em_documentos,
)


class SpeechTests(unittest.TestCase):
    def executar(self, tipo):
        sdk = MagicMock()
        reconhecedor = sdk.SpeechRecognizer.return_value
        callbacks = {}
        for nome in ("recognized", "session_stopped", "canceled"):
            getattr(reconhecedor, nome).connect.side_effect = (
                lambda callback, chave=nome: callbacks.update({chave: callback})
            )

        def iniciar():
            if tipo == "erro":
                callbacks["canceled"](SimpleNamespace(cancellation_details=SimpleNamespace(
                    reason=sdk.CancellationReason.Error,
                )))
            elif tipo == "vazio":
                callbacks["session_stopped"](None)
            elif tipo == "sucesso":
                callbacks["recognized"](SimpleNamespace(result=SimpleNamespace(
                    reason=sdk.ResultReason.RecognizedSpeech, offset=0,
                    duration=10000000, text="Estou cansada.",
                    json='{"NBest": [{"Confidence": 0.9}]}',
                )))
                callbacks["session_stopped"](None)

        reconhecedor.start_continuous_recognition.side_effect = iniciar
        azure = MagicMock()
        azure.cognitiveservices.speech = sdk
        modules = {
            "azure": azure,
            "azure.cognitiveservices": azure.cognitiveservices,
            "azure.cognitiveservices.speech": sdk,
        }
        with patch.dict("sys.modules", modules), patch.dict("os.environ", {
            "AZURE_SPEECH_KEY": "teste", "AZURE_SPEECH_REGION": "teste",
        }), patch("fase4.azure_services.duracao_wav", return_value=2):
            try:
                if tipo == "timeout":
                    with patch("fase4.azure_services.Event") as evento:
                        evento.return_value.wait.return_value = False
                        return _transcrever_azure(Path("consulta.wav"), "pt-BR")
                return _transcrever_azure(Path("consulta.wav"), "pt-BR")
            finally:
                reconhecedor.stop_continuous_recognition.assert_called_once()

    def test_cancelamento_por_erro_nao_retorna_sucesso(self):
        with self.assertRaisesRegex(ServicoAzureIndisponivelError, "cancelou"):
            self.executar("erro")

    def test_ausencia_de_fala_nao_retorna_transcricao(self):
        with self.assertRaisesRegex(ServicoAzureIndisponivelError, "nao reconheceu"):
            self.executar("vazio")

    def test_timeout_interrompe_reconhecimento(self):
        with self.assertRaisesRegex(ServicoAzureIndisponivelError, "Tempo limite"):
            self.executar("timeout")

    def test_transcricao_preserva_texto_e_confianca(self):
        resultado = self.executar("sucesso")
        self.assertEqual(resultado.texto, "Estou cansada.")
        self.assertEqual(resultado.provedor, "azure")
        self.assertEqual(resultado.confianca, 0.9)


# ---------------------------------------------------------------------------
# Text Analytics
# ---------------------------------------------------------------------------


def _sentimento(rotulo, positivo, neutro, negativo):
    return SimpleNamespace(
        is_error=False, sentiment=rotulo,
        confidence_scores=SimpleNamespace(positive=positivo, neutral=neutro, negative=negativo),
    )


def _frases(*frases):
    return SimpleNamespace(is_error=False, key_phrases=list(frases))


def _entidade(texto, categoria, confianca=0.9):
    return SimpleNamespace(text=texto, category=categoria, confidence_score=confianca)


@contextmanager
def text_analytics_falso(
    sentimentos=None, frases=None, entidades_saude=None, entidades_genericas=None,
    saude_falha=False, erro_em_sentimento=False,
):
    """Cliente falso que registra os documentos recebidos em cada operacao."""
    chamadas = {"sentimento": [], "frases": [], "saude": [], "ner": []}

    class Cliente:
        def __init__(self, endpoint, credential):
            self.endpoint = endpoint

        def analyze_sentiment(self, documentos, language):
            chamadas["sentimento"].append(list(documentos))
            if erro_em_sentimento:
                return [SimpleNamespace(is_error=True) for _ in documentos]
            if sentimentos is not None:
                # Um sentimento por documento, na ordem; o ultimo repete se faltar.
                return [sentimentos[min(i, len(sentimentos) - 1)] for i in range(len(documentos))]
            return [_sentimento("negative", 0.1, 0.2, 0.7) for _ in documentos]

        def extract_key_phrases(self, documentos, language):
            chamadas["frases"].append(list(documentos))
            return [frases or _frases("falta de ar", "febre") for _ in documentos]

        def begin_analyze_healthcare_entities(self, documentos, language):
            chamadas["saude"].append(list(documentos))
            if saude_falha:
                raise RuntimeError("recurso sem Text Analytics for health")
            resultados = [SimpleNamespace(is_error=False, entities=entidades_saude or []) for _ in documentos]
            return SimpleNamespace(result=lambda: resultados)

        def recognize_entities(self, documentos, language):
            chamadas["ner"].append(list(documentos))
            return [SimpleNamespace(is_error=False, entities=entidades_genericas or []) for _ in documentos]

    textanalytics = SimpleNamespace(TextAnalyticsClient=Cliente)
    core = SimpleNamespace(credentials=SimpleNamespace(AzureKeyCredential=lambda chave: chave))
    azure = SimpleNamespace(ai=SimpleNamespace(textanalytics=textanalytics), core=core)
    modules = {
        "azure": azure,
        "azure.ai": azure.ai,
        "azure.ai.textanalytics": textanalytics,
        "azure.core": core,
        "azure.core.credentials": core.credentials,
    }
    env = {"AZURE_LANGUAGE_KEY": "teste", "AZURE_LANGUAGE_ENDPOINT": "https://teste.cognitiveservices.azure.com/"}
    with patch.dict("sys.modules", modules), patch.dict("os.environ", env):
        yield chamadas


class DivisaoEmDocumentosTests(unittest.TestCase):
    def test_texto_curto_e_um_documento(self):
        self.assertEqual(dividir_em_documentos("Estou bem."), ["Estou bem."])
        self.assertEqual(dividir_em_documentos("   "), [])

    def test_texto_longo_e_dividido_em_fim_de_frase_sem_perder_palavras(self):
        frase = "A paciente relata falta de ar ao subir escadas e febre a noite. "
        texto = frase * 300  # ~19.000 caracteres
        documentos = dividir_em_documentos(texto)
        self.assertGreaterEqual(len(documentos), 4)
        for documento in documentos:
            self.assertLessEqual(len(documento), LIMITE_CARACTERES_DOCUMENTO)
            self.assertTrue(documento.endswith("."), documento[-20:])
        self.assertEqual(" ".join(documentos).split(), texto.split())

    def test_sem_pontuacao_quebra_em_espaco(self):
        texto = " ".join(["palavra"] * 2000)
        documentos = dividir_em_documentos(texto, limite=100)
        self.assertTrue(all(len(d) <= 100 for d in documentos))
        self.assertEqual(" ".join(documentos).split(), texto.split())


class TextAnalyticsTests(unittest.TestCase):
    def test_texto_longo_vai_em_varios_documentos_e_lotes_de_dez(self):
        frase = "Hoje acordei com dor no peito e muito cansada, doutora. "
        texto = frase * 1100  # ~61.000 caracteres -> 13 documentos -> 2 lotes
        with text_analytics_falso() as chamadas:
            analise = analisar_texto(texto, provedor="azure")
        documentos = [d for lote in chamadas["sentimento"] for d in lote]
        self.assertEqual(analise.documentos_enviados, len(documentos))
        self.assertGreater(len(documentos), 10)
        self.assertEqual(len(chamadas["sentimento"]), 2)
        self.assertTrue(all(len(lote) <= 10 for lote in chamadas["sentimento"]))
        self.assertEqual(analise.provedor, "azure")
        self.assertEqual(analise.sentimento, "negativo")
        self.assertEqual(analise.frases_chave, ["falta de ar", "febre"])  # sem repeticao
        categorias = analise.categorias()
        self.assertIn("dor_toracica", categorias)
        self.assertIn("fadiga", categorias)

    def test_sentimento_e_media_ponderada_pelo_tamanho(self):
        # ~7.400 caracteres -> 2 documentos: o primeiro com ~5.000, o segundo
        # com ~2.400. O rotulo final segue o documento maior.
        texto = "Estou com muita dor e muito cansada. " * 200
        negativo = _sentimento("negative", 0.05, 0.05, 0.9)
        positivo = _sentimento("positive", 0.9, 0.05, 0.05)
        with text_analytics_falso(sentimentos=[negativo, positivo]) as chamadas:
            analise = analisar_texto(texto, provedor="azure")
        self.assertEqual(analise.documentos_enviados, 2)
        tamanhos = [len(d) for d in chamadas["sentimento"][0]]
        self.assertGreater(tamanhos[0], tamanhos[1])
        self.assertEqual(analise.sentimento, "negativo")
        peso_maior = tamanhos[0] / sum(tamanhos)
        esperado = 0.9 * peso_maior + 0.05 * (1 - peso_maior)
        self.assertAlmostEqual(analise.scores_sentimento["negativo"], esperado, places=3)

        with text_analytics_falso(sentimentos=[positivo, negativo]):
            analise = analisar_texto(texto, provedor="azure")
        self.assertEqual(analise.sentimento, "positivo")

    def test_entidade_do_lexico_herda_categoria_e_peso(self):
        texto = "A paciente diz que esta com chiado no peito desde ontem."
        with text_analytics_falso(entidades_saude=[_entidade("chiado no peito", "SymptomOrSign", 0.95)]):
            analise = analisar_texto(texto, provedor="azure")
        termo = next(t for t in analise.termos_criticos if t.termo == "chiado no peito")
        self.assertEqual(termo.categoria, "respiratorio")
        self.assertEqual(termo.peso, 0.9)
        # Nao duplica o que o lexico ja tinha encontrado sozinho.
        self.assertEqual(sum(1 for t in analise.termos_criticos if t.termo == "chiado no peito"), 1)
        self.assertEqual(analise.entidades[0]["origem"], "text_analytics_for_health")

    def test_sintoma_fora_do_lexico_vira_termo_informativo(self):
        texto = "Relata formigamento na mao direita e visao turva."
        entidades = [
            _entidade("formigamento", "SymptomOrSign", 0.92),
            _entidade("visao turva", "SymptomOrSign", 0.6),   # confianca baixa: ignorada
            _entidade("mao direita", "BodyStructure", 0.99),   # nao e sintoma: ignorada
        ]
        with text_analytics_falso(entidades_saude=entidades):
            analise = analisar_texto(texto, provedor="azure")
        termos = {t.termo: t for t in analise.termos_criticos}
        self.assertIn("formigamento", termos)
        self.assertEqual(termos["formigamento"].categoria, CATEGORIA_SINTOMA_AZURE)
        self.assertEqual(termos["formigamento"].peso, PESO_SINTOMA_AZURE)
        self.assertNotIn("visao turva", termos)
        self.assertNotIn("mao direita", termos)
        self.assertEqual(len(analise.entidades), 3)

    def test_sem_recurso_de_saude_cai_para_ner_generico_e_nao_falha(self):
        texto = "Tomei dipirona duas vezes ontem."
        with text_analytics_falso(saude_falha=True, entidades_genericas=[_entidade("ontem", "DateTime")]) as chamadas:
            analise = analisar_texto(texto, provedor="azure")
        self.assertTrue(chamadas["saude"])
        self.assertTrue(chamadas["ner"])
        self.assertEqual(analise.entidades[0]["origem"], "ner_generico")
        self.assertEqual(analise.termos_criticos, [])

    def test_erro_em_um_documento_falha_explicitamente(self):
        with text_analytics_falso(erro_em_sentimento=True):
            with self.assertRaises(ServicoAzureIndisponivelError):
                analisar_texto("qualquer texto", provedor="azure")


class DotenvTests(unittest.TestCase):
    def test_le_env_sem_sobrescrever_ambiente(self):
        import os
        import tempfile

        from fase4.config import carregar_dotenv

        conteudo = "\n".join(
            [
                "# comentario",
                'AZURE_SPEECH_KEY="chave"',
                "AZURE_SPEECH_REGION=brazilsouth # regiao",
                "export AZURE_LANGUAGE_KEY='outra'",
                "AZURE_LANGUAGE_ENDPOINT=https://x.cognitiveservices.azure.com/",
                "JA_DEFINIDA=do_arquivo",
                "linha sem igual",
            ]
        )
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / ".env"
            caminho.write_text(conteudo, encoding="utf-8")
            with patch.dict(os.environ, {"JA_DEFINIDA": "do_ambiente"}, clear=True):
                carregadas = carregar_dotenv(caminho)
                self.assertEqual(
                    carregadas,
                    {
                        "AZURE_SPEECH_KEY": "chave",
                        "AZURE_SPEECH_REGION": "brazilsouth",
                        "AZURE_LANGUAGE_KEY": "outra",
                        "AZURE_LANGUAGE_ENDPOINT": "https://x.cognitiveservices.azure.com/",
                    },
                )
                self.assertEqual(os.environ["JA_DEFINIDA"], "do_ambiente")
                self.assertEqual(os.environ["AZURE_SPEECH_REGION"], "brazilsouth")

    def test_arquivo_ausente_nao_falha(self):
        from fase4.config import carregar_dotenv

        self.assertEqual(carregar_dotenv(Path("nao/existe/.env")), {})
