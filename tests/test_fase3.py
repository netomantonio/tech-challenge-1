"""Testes da Fase 3 (assistente medico virtual).

Segue o mesmo estilo de `tests/test_fase2.py`: `unittest.TestCase` puro,
sem rede e sem depender de `GROQ_API_KEY` — o LLM e sempre um `FakeLLM`
deterministico (`fase3.llm_backend.get_llm("fake", ...)`).
"""

from __future__ import annotations

import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from fase3 import ehr_tools
from fase3.assistant_chain import responder_pergunta_clinica
from fase3.clinical_flow_graph import executar_fluxo_clinico
from fase3.data.build_finetuning_dataset import (
    FinetuningExample,
    anonimizar_texto,
    curar,
    detectar_pii,
)
from fase3.guardrails import DISCLAIMER, aplicar_guardrails
from fase3.llm_backend import GroqLLM, LLMUnavailableError, get_llm
from fase3.retrieval import buscar_protocolos, construir_retriever


class AnonimizacaoTests(unittest.TestCase):
    def test_redige_cpf_telefone_email_e_nome(self) -> None:
        texto = (
            "Paciente: Joao da Silva, CPF 123.456.789-00, telefone (11) 91234-5678, "
            "email joao@example.com."
        )
        resultado = anonimizar_texto(texto)
        self.assertNotIn("Joao da Silva", resultado)
        self.assertNotIn("123.456.789-00", resultado)
        self.assertNotIn("91234-5678", resultado)
        self.assertNotIn("joao@example.com", resultado)
        self.assertTrue(detectar_pii(texto))
        self.assertFalse(detectar_pii(resultado))

    def test_texto_sem_pii_nao_e_alterado_por_deteccao(self) -> None:
        texto = "O protocolo PROT-004 recomenda reavaliacao a cada 4 horas."
        self.assertFalse(detectar_pii(texto))


class CuradoriaDatasetTests(unittest.TestCase):
    def test_remove_duplicatas_e_tamanhos_fora_da_faixa(self) -> None:
        exemplos = [
            FinetuningExample("pergunta A", "", "resposta valida com tamanho razoavel para passar", "faq", "1"),
            FinetuningExample("pergunta A", "", "resposta valida com tamanho razoavel para passar", "faq", "1"),
            FinetuningExample("pergunta B", "", "curta", "faq", "2"),
            FinetuningExample("pergunta C", "", "resposta valida de tamanho aceitavel para o teste", "faq", "3"),
        ]
        curados = curar(exemplos)
        self.assertEqual(len(curados), 2)
        ids = {e.source_id for e in curados}
        self.assertEqual(ids, {"1", "3"})


class GuardrailsTests(unittest.TestCase):
    def test_resposta_segura_recebe_disclaimer(self) -> None:
        resultado = aplicar_guardrails("O protocolo indica reavaliacao periodica.")
        self.assertFalse(resultado.bloqueado)
        self.assertIn(DISCLAIMER, resultado.resposta)

    def test_prescricao_direta_e_bloqueada(self) -> None:
        resultado = aplicar_guardrails("Tome 500mg de dipirona agora mesmo.")
        self.assertTrue(resultado.bloqueado)
        self.assertEqual(resultado.motivo, "prescricao_direta_bloqueada")

    def test_pii_e_bloqueada(self) -> None:
        resultado = aplicar_guardrails("Contate Joao da Silva no telefone (11) 91234-5678.")
        self.assertTrue(resultado.bloqueado)
        self.assertEqual(resultado.motivo, "pii_detectada")


class LLMBackendTests(unittest.TestCase):
    def test_fake_llm_retorna_respostas_em_ordem(self) -> None:
        llm = get_llm("fake", respostas=["primeira", "segunda"])
        self.assertEqual(llm.invoke("q1"), "primeira")
        self.assertEqual(llm.invoke("q2"), "segunda")
        self.assertEqual(llm.invoke("q3"), llm.resposta_padrao)

    def test_groq_sem_chave_levanta_erro_claro(self) -> None:
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GROQ_API_KEY", None)
            llm = GroqLLM()
            with self.assertRaises(LLMUnavailableError):
                llm.invoke("teste")

    def test_backend_desconhecido_levanta_value_error(self) -> None:
        with self.assertRaises(ValueError):
            get_llm("inexistente")


class RetrievalTests(unittest.TestCase):
    def test_busca_retorna_protocolo_relevante(self) -> None:
        retriever = construir_retriever(k=3)
        documentos = buscar_protocolos("dor persistente apos a cirurgia", retriever)
        ids = {doc.metadata["id"] for doc in documentos}
        self.assertIn("PROT-004", ids)


class EhrToolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls._tmpdir.name) / "prontuarios_teste.db"
        ehr_tools.seed_db(cls.db_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_seed_e_consulta_paciente_existente(self) -> None:
        paciente = ehr_tools.get_paciente("PAC-0001", self.db_path)
        self.assertIsNotNone(paciente)
        self.assertIn("ecocardiograma_basal", paciente["exames_pendentes"])

    def test_paciente_inexistente_retorna_none(self) -> None:
        self.assertIsNone(ehr_tools.get_paciente("PAC-0000", self.db_path))

    def test_alertas_ativos_do_paciente(self) -> None:
        alertas = ehr_tools.get_alertas_ativos("PAC-0005", self.db_path)
        self.assertTrue(any("Febre" in a for a in alertas))


class AssistantChainTests(unittest.TestCase):
    def test_resposta_inclui_fontes_e_disclaimer(self) -> None:
        llm = get_llm("fake", respostas=["Siga o protocolo institucional relevante."])
        resultado = responder_pergunta_clinica(
            "O que fazer com dor pos-operatoria persistente?", paciente_id="PAC-0006", llm=llm
        )
        self.assertGreater(len(resultado["fontes"]), 0)
        self.assertFalse(resultado["bloqueado"])
        self.assertIn(DISCLAIMER, resultado["resposta"])

    def test_prescricao_direta_do_llm_e_bloqueada_na_chain(self) -> None:
        llm = get_llm("fake", respostas=["Tome 500mg de dipirona agora mesmo."])
        resultado = responder_pergunta_clinica("Qual conduta seguir?", llm=llm)
        self.assertTrue(resultado["bloqueado"])
        self.assertEqual(resultado["motivo_bloqueio"], "prescricao_direta_bloqueada")


class ClinicalFlowGraphTests(unittest.TestCase):
    def test_fluxo_completo_paciente_com_exame_pendente(self) -> None:
        llm = get_llm("fake", respostas=["Siga o protocolo institucional relevante."])
        estado = executar_fluxo_clinico("PAC-0001", "Posso iniciar o tratamento?", llm=llm)
        self.assertTrue(estado["paciente_encontrado"])
        self.assertTrue(any("Exames pendentes" in a for a in estado["alertas"]))

    def test_fluxo_encerra_com_seguranca_se_paciente_nao_existe(self) -> None:
        llm = get_llm("fake", respostas=["nao deveria ser usado"])
        estado = executar_fluxo_clinico("PAC-0000", "Qual conduta?", llm=llm)
        self.assertFalse(estado["paciente_encontrado"])
        self.assertNotIn("sugestao", estado)

    def test_fluxo_escala_alerta_quando_guardrail_bloqueia(self) -> None:
        llm = get_llm("fake", respostas=["Tome 500mg de dipirona agora mesmo."])
        estado = executar_fluxo_clinico("PAC-0002", "Qual conduta seguir?", llm=llm)
        self.assertTrue(estado["bloqueado"])
        self.assertTrue(any("requer revisao manual" in a for a in estado["alertas"]))


if __name__ == "__main__":
    unittest.main()
