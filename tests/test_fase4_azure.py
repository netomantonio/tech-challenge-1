"""Falhas de reconhecimento sem chamadas de rede."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fase4.azure_services import ServicoAzureIndisponivelError, _transcrever_azure


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
