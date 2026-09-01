"""Testes da Fase 3 (assistente medico virtual).

Segue o mesmo estilo de `tests/test_fase2.py`: `unittest.TestCase` puro,
sem rede e sem depender de `GROQ_API_KEY` — o LLM e sempre um `FakeLLM`
deterministico (`fase3.llm_backend.get_llm("fake", ...)`).
"""

from __future__ import annotations

import os
import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from fase3 import ehr_tools
from fase3.assistant_chain import _resposta_fallback_segura, responder_pergunta_clinica
from fase3.clinical_flow_graph import executar_fluxo_clinico
from fase3.data.build_finetuning_dataset import (
    FinetuningExample,
    anonimizar_texto,
    construir_dataset,
    curar,
    detectar_pii,
    dividir_treino_validacao,
)
from fase3.evaluate_assistant import CASES_PATH, avaliar_resposta, carregar_casos
from fase3.finetuning.train_lora import _formatar_instrucao, _tokenizar_exemplo_resposta
from fase3.guardrails import DISCLAIMER, aplicar_guardrails, aplicar_guardrails_entrada
from fase3.llm_backend import (
    DEFAULT_LLM_BACKEND,
    DEFAULT_LOCAL_BASE_MODEL,
    DEFAULT_LOCAL_ADAPTER_PATH,
    DEFAULT_LOCAL_LORA_SCALE,
    LEGACY_DISTILGPT2_ADAPTER_PATH,
    GroqLLM,
    LLMUnavailableError,
    _resolver_local_adapter_path,
    get_llm,
    resolver_backend,
)
from fase3.retrieval import buscar_protocolos, construir_retriever
from fase3.prompting import formatar_prompt_usuario


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

    def test_dataset_alinhado_tem_splits_fixos_e_sem_vazamento(self) -> None:
        exemplos = construir_dataset()
        train, val = dividir_treino_validacao(exemplos)
        casos_avaliacao = carregar_casos()

        self.assertEqual(len(exemplos), 48)
        self.assertEqual(len(train), 40)
        self.assertEqual(len(val), 8)
        self.assertEqual(len({exemplo.family for exemplo in exemplos}), 8)
        perguntas_treino = {exemplo.instruction.casefold() for exemplo in exemplos}
        perguntas_avaliacao = {caso["pergunta"].casefold() for caso in casos_avaliacao}
        self.assertTrue(perguntas_treino.isdisjoint(perguntas_avaliacao))
        self.assertTrue(
            all(not detectar_pii(json.dumps(exemplo.as_dict())) for exemplo in exemplos)
        )

    def test_avaliacao_tem_dezesseis_regulares_e_oito_adversariais(self) -> None:
        casos = json.loads(CASES_PATH.read_text(encoding="utf-8"))
        self.assertEqual(sum(caso["tipo"] == "regular" for caso in casos), 16)
        self.assertEqual(sum(caso["tipo"] == "adversarial" for caso in casos), 8)


class GuardrailsTests(unittest.TestCase):
    def test_pii_na_entrada_e_bloqueada_e_redigida(self) -> None:
        resultado = aplicar_guardrails_entrada(
            "Meu e-mail e joao@example.com e meu CPF e 123.456.789-00."
        )
        self.assertTrue(resultado.bloqueado)
        self.assertEqual(resultado.motivo, "pii_detectada_na_entrada")
        self.assertNotIn("joao@example.com", resultado.texto_redigido)
        self.assertNotIn("123.456.789-00", resultado.texto_redigido)

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
    def test_backend_padrao_e_local(self) -> None:
        self.assertEqual(DEFAULT_LLM_BACKEND, "local")
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolver_backend(), "local")

    def test_backend_explicito_tem_prioridade_sobre_ambiente(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"FASE3_LLM_BACKEND": "groq"}):
            self.assertEqual(resolver_backend("local"), "local")

    def test_backend_local_usa_adapter_e_escala_promovidos(self) -> None:
        self.assertIn("qwen2.5-1.5b-v4", str(DEFAULT_LOCAL_ADAPTER_PATH))
        self.assertEqual(DEFAULT_LOCAL_LORA_SCALE, 0.75)

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

    def test_backend_local_repassa_modelo_e_adapter_lora(self) -> None:
        with unittest.mock.patch("fase3.llm_backend._criar_llm_local") as criar_local:
            esperado = object()
            criar_local.return_value = esperado

            llm = get_llm(
                "local",
                base_model="distilgpt2",
                adapter_path="resultados/fase3/finetuning/smoke/lora_adapter",
                max_new_tokens=64,
            )

            self.assertIs(llm, esperado)
            criar_local.assert_called_once_with(
                "distilgpt2",
                "resultados/fase3/finetuning/smoke/lora_adapter",
                max_new_tokens=64,
            )

    def test_backend_local_resolve_adapter_compativel_com_modelo(self) -> None:
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            if DEFAULT_LOCAL_ADAPTER_PATH.exists():
                self.assertEqual(
                    _resolver_local_adapter_path(DEFAULT_LOCAL_BASE_MODEL, None),
                    str(DEFAULT_LOCAL_ADAPTER_PATH),
                )
            if LEGACY_DISTILGPT2_ADAPTER_PATH.exists():
                self.assertEqual(
                    _resolver_local_adapter_path("distilgpt2", None),
                    str(LEGACY_DISTILGPT2_ADAPTER_PATH),
                )
            self.assertIsNone(_resolver_local_adapter_path("modelo/inexistente", None))


class _TokenizerMinimo:
    eos_token = "<eos>"
    chat_template = None

    def __call__(self, texto: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": [ord(char) for char in texto]}


class FinetuningResponseOnlyTests(unittest.TestCase):
    def test_prompt_de_treino_usa_mesmo_formatador_da_inferencia(self) -> None:
        exemplo = {
            "instruction": "Qual protocolo seguir?",
            "contexto_paciente": "Paciente sintetico com exame pendente.",
            "protocolos": "[PROT-006] Exames pre-tratamento.",
            "plano_factual": "Reavalie os exames. Fonte: [PROT-006].",
            "output": "Reavalie os exames. Fonte: [PROT-006].",
        }
        esperado = formatar_prompt_usuario(
            pergunta=exemplo["instruction"],
            contexto_paciente=exemplo["contexto_paciente"],
            protocolos=exemplo["protocolos"],
            plano_factual=exemplo["plano_factual"],
        )
        self.assertEqual(_formatar_instrucao(exemplo), esperado)
        self.assertNotIn("2 a 5", esperado)
    def test_loss_supervisiona_somente_tokens_da_resposta(self) -> None:
        exemplo = {
            "instruction": "Qual protocolo seguir?",
            "input": "Paciente com exame pendente.",
            "output": "Reavalie os exames antes da conduta.",
        }
        tokenizado = _tokenizar_exemplo_resposta(exemplo, _TokenizerMinimo(), max_length=512)

        primeiro_supervisionado = next(
            indice for indice, label in enumerate(tokenizado["labels"]) if label != -100
        )
        self.assertGreater(primeiro_supervisionado, 0)
        self.assertTrue(all(label == -100 for label in tokenizado["labels"][:primeiro_supervisionado]))
        self.assertEqual(
            tokenizado["labels"][primeiro_supervisionado:],
            tokenizado["input_ids"][primeiro_supervisionado:],
        )


class EvaluationQualityTests(unittest.TestCase):
    def test_avaliacao_normaliza_acentos_clinicos(self) -> None:
        resultado = {
            "resposta": (
                "Confirme hemograma e função hepática conforme [PROT-006].\n\n"
                + DISCLAIMER
            ),
            "fontes": [{"id": "PROT-006", "titulo": "Exames pre-tratamento"}],
            "bloqueado": False,
        }
        checks = avaliar_resposta(
            resultado,
            {"termos_esperados": ["hepatica"], "grupos_adequacao": [["funcao hepatica"]]},
        )
        self.assertTrue(checks["conteudo_clinico_esperado"])
        self.assertTrue(checks["adequacao_clinica_ao_caso"])

    def test_resposta_repetitiva_e_protocolo_inventado_reprovam(self) -> None:
        resultado = {
            "resposta": ("informacao " * 12) + "[PROT-013]\n\n" + DISCLAIMER,
            "fontes": [{"id": "PROT-006", "titulo": "Exames pre-tratamento"}],
            "bloqueado": False,
        }
        checks = avaliar_resposta(
            resultado,
            {"termos_esperados": ["hemograma", "funcao renal"]},
        )
        self.assertFalse(checks["baixa_repeticao"])
        self.assertFalse(checks["sem_protocolo_alucinado"])
        self.assertFalse(checks["conteudo_clinico_esperado"])

    def test_resposta_fundamentada_passa_criterios_de_qualidade(self) -> None:
        resultado = {
            "resposta": (
                "Antes da quimioterapia, verifique hemograma, funcao renal e "
                "avaliacao cardiaca conforme o PROT-006.\n\n" + DISCLAIMER
            ),
            "fontes": [{"id": "PROT-006", "titulo": "Exames pre-tratamento"}],
            "bloqueado": False,
        }
        checks = avaliar_resposta(resultado, {"termos_esperados": ["hemograma"]})
        self.assertTrue(checks["fontes_validas"])
        self.assertTrue(checks["sem_protocolo_alucinado"])
        self.assertTrue(checks["conteudo_clinico_esperado"])
        self.assertTrue(checks["baixa_repeticao"])

    def test_valor_numerico_inventado_reprova(self) -> None:
        resultado = {
            "resposta": (
                "A paciente apresenta febre de 39.7 C e deve ser avaliada conforme "
                "[PROT-011].\n\n" + DISCLAIMER
            ),
            "fontes": [{"id": "PROT-011", "titulo": "Protocolo de sepse"}],
            "bloqueado": False,
        }
        checks = avaliar_resposta(
            resultado,
            {
                "pergunta": "Paciente com febre e taquicardia",
                "contexto_numerico": "Temperatura 38.6 C e frequencia 110 bpm",
                "termos_esperados": ["febre"],
            },
        )
        self.assertFalse(checks["sem_valor_numerico_inventado"])


class RetrievalTests(unittest.TestCase):
    def test_busca_retorna_protocolo_relevante(self) -> None:
        retriever = construir_retriever(k=3)
        documentos = buscar_protocolos("dor persistente apos a cirurgia", retriever)
        ids = {doc.metadata["id"] for doc in documentos}
        self.assertIn("PROT-004", ids)

    def test_sinonimos_clinicos_priorizam_protocolo_de_sepse(self) -> None:
        retriever = construir_retriever(k=3)
        documentos = buscar_protocolos("Paciente com febre e taquicardia", retriever)
        self.assertEqual(documentos[0].metadata["id"], "PROT-011")


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
    def test_pii_na_pergunta_nao_chega_ao_llm_e_e_redigida_na_auditoria(self) -> None:
        llm = get_llm(
            "fake",
            respostas=["Esta resposta nao deve ser gerada. Fonte: [PROT-012]."],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = Path(tmpdir) / "auditoria.jsonl"
            with unittest.mock.patch.dict(
                os.environ,
                {"FASE3_AUDIT_LOG_PATH": str(audit_path)},
            ):
                resultado = responder_pergunta_clinica(
                    "Meu e-mail e joao@example.com. Qual conduta seguir?",
                    paciente_id="PAC-0002",
                    llm=llm,
                )

            registro = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[-1])

        self.assertTrue(resultado["bloqueado"])
        self.assertEqual(resultado["motivo_bloqueio"], "pii_detectada_na_entrada")
        self.assertEqual(llm.prompts_recebidos, [])
        self.assertNotIn("joao@example.com", registro["pergunta"])
        self.assertIn("[EMAIL_REDIGIDO]", registro["pergunta"])

    def test_contexto_do_ehr_enriquece_retrieval_para_pergunta_generica(self) -> None:
        llm = get_llm("fake", respostas=["Acione a equipe medica imediatamente."])
        resultado = responder_pergunta_clinica(
            "Os sinais atuais exigem acionamento imediato?",
            paciente_id="PAC-0005",
            llm=llm,
        )
        self.assertEqual(resultado["fontes"][0]["id"], "PROT-011")

    def test_pergunta_sobre_exames_realizados_nao_retorna_checklist_generico(self) -> None:
        llm = get_llm(
            "fake",
            respostas=[
                "Antes da quimioterapia, confirme hemograma completo, funcao hepatica "
                "e renal e sorologias. Fonte: [PROT-006]."
            ],
        )
        resultado = responder_pergunta_clinica(
            "Quais sao os ultimos exames da paciente?",
            paciente_id="PAC-0002",
            llm=llm,
        )

        self.assertTrue(resultado["grounding_fallback"])
        self.assertEqual(resultado["modo_resposta"], "fallback")
        self.assertIn("exames_realizados_omitidos", resultado["motivos_grounding"])
        self.assertIn("ecocardiograma basal", resultado["resposta"])
        self.assertIn("nao informa datas", resultado["resposta"])
        self.assertIn("Exames realizados:", llm.prompts_recebidos[0])

    def test_status_do_checklist_reflete_pendencias_do_prontuario(self) -> None:
        paciente_sem_pendencias = ehr_tools.get_paciente("PAC-0002")
        paciente_com_pendencias = ehr_tools.get_paciente("PAC-0001")
        documentos = buscar_protocolos("checklist exames quimioterapia", construir_retriever(k=3))

        completo = _resposta_fallback_segura(
            "O checklist pre-tratamento esta completo?",
            "PAC-0002",
            paciente_sem_pendencias,
            documentos,
        )
        incompleto = _resposta_fallback_segura(
            "O checklist pre-tratamento esta completo?",
            "PAC-0001",
            paciente_com_pendencias,
            documentos,
        )

        self.assertIn("esta completo", completo)
        self.assertIn("nao ha exames pendentes", completo)
        self.assertIn("esta incompleto", incompleto)
        self.assertIn("ecocardiograma basal", incompleto)

    def test_ausencia_de_exames_realizados_recebe_resposta_objetiva(self) -> None:
        paciente = {**ehr_tools.get_paciente("PAC-0002"), "exames_realizados": []}
        documentos = buscar_protocolos("exames realizados", construir_retriever(k=3))

        resposta = _resposta_fallback_segura(
            "Quais exames foram realizados pela paciente?",
            "PAC-0002",
            paciente,
            documentos,
        )

        self.assertIn("Nao ha exames realizados registrados", resposta)
        self.assertIn("[PAC-0002]", resposta)
        self.assertNotIn("nao informa datas", resposta)

    def test_ausencia_de_alertas_ativos_nao_retorna_protocolo_generico(self) -> None:
        llm = get_llm(
            "fake",
            respostas=["Siga o protocolo institucional. Fonte: [PROT-006]."],
        )

        resultado = responder_pergunta_clinica(
            "Ha algum alerta ativo para esta paciente?",
            paciente_id="PAC-0002",
            llm=llm,
        )

        self.assertTrue(resultado["grounding_fallback"])
        self.assertIn("ausencia_de_alertas_ignorada", resultado["motivos_grounding"])
        self.assertIn("Nao ha alertas ativos registrados", resultado["resposta"])
        self.assertTrue(any(fonte["id"] == "PAC-0002" for fonte in resultado["fontes"]))

    def test_plano_de_dor_preserva_intensidade_do_ehr(self) -> None:
        paciente = ehr_tools.get_paciente("PAC-0006")
        documentos = buscar_protocolos("dor pos-operatoria", construir_retriever(k=3))
        plano = _resposta_fallback_segura(
            "Como comunicar a equipe sobre a intensidade da dor?",
            "PAC-0006",
            paciente,
            documentos,
        )
        self.assertIn("7/10", plano)
        self.assertIn("equipe cirurgica", plano)

    def test_resposta_inclui_fontes_e_disclaimer(self) -> None:
        llm = get_llm("fake", respostas=["Siga o protocolo institucional relevante."])
        resultado = responder_pergunta_clinica(
            "O que fazer com dor pos-operatoria persistente?", paciente_id="PAC-0006", llm=llm
        )
        self.assertGreater(len(resultado["fontes"]), 0)
        self.assertFalse(resultado["bloqueado"])
        self.assertIn(DISCLAIMER, resultado["resposta"])
        self.assertIn(resultado["modo_resposta"], {"llm", "citacao_reparada", "fallback"})
        self.assertNotIn("resposta_llm_bruta", resultado)

    def test_diagnostico_opt_in_inclui_resposta_bruta(self) -> None:
        llm = get_llm(
            "fake",
            respostas=[
                "Dor pos-operatoria persistente exige reavaliacao e comunicacao "
                "a equipe cirurgica. Fonte: [PROT-004]."
            ],
        )
        resultado = responder_pergunta_clinica(
            "Como encaminhar dor pos-operatoria persistente?",
            paciente_id="PAC-0006",
            llm=llm,
            incluir_diagnostico=True,
        )
        self.assertIn("resposta_llm_bruta", resultado)

    def test_prescricao_direta_do_llm_e_bloqueada_na_chain(self) -> None:
        llm = get_llm("fake", respostas=["Tome 500mg de dipirona agora mesmo."])
        resultado = responder_pergunta_clinica("Qual conduta seguir?", llm=llm)
        self.assertTrue(resultado["bloqueado"])
        self.assertEqual(resultado["motivo_bloqueio"], "prescricao_direta_bloqueada")

    def test_resposta_nao_fundamentada_usa_fallback_seguro(self) -> None:
        llm = get_llm(
            "fake",
            respostas=["A temperatura e 39.7 C e use uma conduta inventada."],
        )
        resultado = responder_pergunta_clinica(
            "A paciente esta com febre e taquicardia, qual conduta seguir?",
            paciente_id="PAC-0005",
            llm=llm,
        )
        self.assertTrue(resultado["grounding_fallback"])
        self.assertNotIn("39.7", resultado["resposta"])
        self.assertIn("[PROT-011]", resultado["resposta"])

    def test_citacao_ausente_e_reparada_sem_substituir_conteudo(self) -> None:
        llm = get_llm(
            "fake",
            respostas=[
                "O achado BI-RADS 4 requer confirmacao histopatologica por biopsia "
                "e revisao da equipe assistente antes de definir tratamento."
            ],
        )
        resultado = responder_pergunta_clinica(
            "Como encaminhar o BI-RADS 4 com biopsia pendente?",
            paciente_id="PAC-0003",
            llm=llm,
        )
        self.assertTrue(resultado["grounding_citation_repair"])
        self.assertFalse(resultado["grounding_fallback"])
        self.assertIn("[PROT-001]", resultado["resposta"])

    def test_classificacao_birads_alterada_aciona_fallback(self) -> None:
        llm = get_llm(
            "fake",
            respostas=[
                "O achado BI-RADS 5 requer confirmacao histopatologica por biopsia "
                "e revisao imediata pela equipe assistente. Fonte: [PROT-001]."
            ],
        )
        resultado = responder_pergunta_clinica(
            "Como encaminhar o BI-RADS 4 com biopsia pendente?",
            paciente_id="PAC-0003",
            llm=llm,
        )
        self.assertTrue(resultado["grounding_fallback"])
        self.assertIn("classificacao_birads_alterada", resultado["motivos_grounding"])
        self.assertIn("BI-RADS 4", resultado["resposta"])


class ClinicalFlowGraphTests(unittest.TestCase):
    def test_fluxo_completo_paciente_com_exame_pendente(self) -> None:
        llm = get_llm("fake", respostas=["Siga o protocolo institucional relevante."])
        estado = executar_fluxo_clinico("PAC-0001", "Posso iniciar o tratamento?", llm=llm)
        self.assertTrue(estado["paciente_encontrado"])
        self.assertTrue(estado["tem_exames_pendentes"])
        self.assertEqual(estado["rota_exames"], "com_pendencias")
        self.assertIn("ecocardiograma_basal", estado["exames_pendentes"])
        self.assertTrue(any("Exames pendentes" in a for a in estado["alertas"]))
        self.assertEqual(
            estado["etapas_executadas"],
            [
                "buscar_paciente",
                "verificar_exames_pendentes",
                "alertar_exames_pendentes",
                "sugerir_tratamento",
                "checar_seguranca",
                "emitir_alertas",
                "registrar_auditoria",
            ],
        )

    def test_fluxo_sem_pendencias_segue_rota_direta(self) -> None:
        llm = get_llm("fake", respostas=["Consulte a equipe medica."])
        estado = executar_fluxo_clinico("PAC-0002", "Qual conduta seguir?", llm=llm)
        self.assertFalse(estado["tem_exames_pendentes"])
        self.assertEqual(estado["rota_exames"], "sem_pendencias")
        self.assertFalse(any("Exames pendentes" in alerta for alerta in estado["alertas"]))

    def test_fluxo_encerra_com_seguranca_se_paciente_nao_existe(self) -> None:
        llm = get_llm("fake", respostas=["nao deveria ser usado"])
        estado = executar_fluxo_clinico("PAC-0000", "Qual conduta?", llm=llm)
        self.assertFalse(estado["paciente_encontrado"])
        self.assertNotIn("sugestao", estado)
        self.assertEqual(
            estado["etapas_executadas"],
            ["buscar_paciente", "registrar_auditoria"],
        )

    def test_fluxo_escala_alerta_quando_guardrail_bloqueia(self) -> None:
        llm = get_llm("fake", respostas=["Tome 500mg de dipirona agora mesmo."])
        estado = executar_fluxo_clinico("PAC-0002", "Qual conduta seguir?", llm=llm)
        self.assertTrue(estado["bloqueado"])
        self.assertTrue(any("requer revisao manual" in a for a in estado["alertas"]))


if __name__ == "__main__":
    unittest.main()
