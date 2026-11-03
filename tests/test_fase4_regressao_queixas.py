import unittest
from pathlib import Path
from unittest.mock import patch
from fase4.azure_services import _termos_a_partir_de_entidades, _buscar_termos_criticos
from fase4.audio_analysis import analisar_consulta, relatorio_markdown


class QueixasTests(unittest.TestCase):
    def test_negacao_e_expressoes_positivas(self):
        texto = "Estou bem melhor, sem dor. Estou animada e dormindo tranquila."
        entidades = [{"texto": t, "categoria": "SymptomOrSign", "confianca": .99}
                     for t in ("estou bem melhor", "dor", "animada", "dormindo tranquila")]
        self.assertEqual(_termos_a_partir_de_entidades(entidades, texto, []), [])

    def test_negacao_nao_apaga_queixa_posterior(self):
        self.assertEqual(_buscar_termos_criticos("Nego falta de ar."), [])
        self.assertTrue(_buscar_termos_criticos("Sem febre, mas agora tenho febre."))
        self.assertTrue(_buscar_termos_criticos("Estou sem ar e nao consigo dormir."))

    def test_tts_preserva_conteudo_sem_avaliar_voz(self):
        import tempfile
        from fase4.data.gerar_dados_sinteticos import gerar_amostras_paciente
        import json
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            gerar_amostras_paciente("PAC-0006", seed=42, destino_raiz=raiz)
            audio = raiz / "PAC-0006" / "consulta.wav"
            ref = audio.with_suffix(".transcricao.json")
            dados = json.loads(ref.read_text(encoding="utf-8"))
            dados["origem"] = "tts_powershell"
            ref.write_text(json.dumps(dados), encoding="utf-8")
            with patch("fase4.audio_analysis.avaliar_alteracoes_vocais") as avaliar:
                resultado = analisar_consulta(audio, "PAC-0006", provedor_azure="offline")
                avaliar.assert_not_called()
            self.assertTrue(resultado.achados)
            self.assertIn("TTS", relatorio_markdown(resultado))
