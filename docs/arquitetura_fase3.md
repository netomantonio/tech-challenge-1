# Arquitetura da Fase 3

## Visão geral

A solução combina fine-tuning LoRA, RAG, prontuário estruturado,
LangChain, LangGraph, guardrails e auditoria. Todos os dados clínicos são
fictícios ou anonimizados.

```mermaid
flowchart TD
    D[48 casos clínicos revisados] --> B[Curadoria e anonimização]
    B --> T[40 treino]
    B --> V[8 validação]
    T --> L[LoRA Qwen2.5-1.5B]
    V --> C[Calibração 0.25 / 0.5 / 0.75 / 1.0]
    L --> C
    C --> A[Adapter v4, escala 0.75]

    EHR[(EHR SQLite sintético)] --> R[Consulta enriquecida]
    P[(Protocolos internos)] --> R
    UI[Interface web local] --> API[FastAPI Fase 3]
    API --> FLOW[LangGraph clínico]
    FLOW --> R
    R --> PLAN[Plano factual autorizado]
    A --> CHAIN[LangChain: prompt / LLM / parser]
    PLAN --> CHAIN
    CHAIN --> G[Guardrails e grounding]
    G --> OUT[Resposta, fontes e modo]
    OUT --> LOG[(Auditoria JSONL)]
```

## Componentes

| Componente | Responsabilidade |
| --- | --- |
| `fase3/prompting.py` | Contrato único de prompt para treino e inferência |
| `fase3/data/build_finetuning_dataset.py` | Anonimização, curadoria e splits 40/8 |
| `fase3/finetuning/train_lora.py` | Treino LoRA/PEFT response-only |
| `fase3/finetuning/evaluate_adapter_loss.py` | Reavaliação reproduzível de loss |
| `fase3/calibrate_adapter.py` | Baseline, escalas, promoção e gates |
| `fase3/evaluate_assistant.py` | Avaliação bruta, final e adversarial |
| `fase3/ehr_tools.py` | Mock de EHR SQLite |
| `fase3/retrieval.py` | BM25, sinônimos e reranking clínico |
| `fase3/assistant_chain.py` | LCEL, plano factual, grounding e modos de resposta |
| `fase3/clinical_flow_graph.py` | Rotas decisórias e alertas |
| `fase3/guardrails.py` | Bloqueio de PII e prescrição direta |
| `fase3/logging_utils.py` | Auditoria estruturada |
| `fase3/web_app.py` | API local, ciclo de vida do modelo e serialização das consultas |
| `fase3/web/` | Interface responsiva de prontuário, consulta e evidências |

## Interface web

O comando `python -m fase3.web_app --backend local` inicia o serviço em
`http://127.0.0.1:8010` e abre a interface no navegador. O backend mantém uma
única instância do modelo em memória e serializa as gerações para proteger o
uso da GPU. A tela consome apenas os endpoints locais `/api/status`,
`/api/pacientes` e `/api/consultas`; a lógica clínica continua centralizada no
LangGraph e não é duplicada no frontend.

## Contrato da resposta

`responder_pergunta_clinica(pergunta, paciente_id, incluir_diagnostico)`
retorna resposta final, fontes, estado de bloqueio, motivos de grounding e
`modo_resposta`:

- `llm`: geração aceita sem alteração;
- `citacao_reparada`: conteúdo aceito com fontes autorizadas anexadas;
- `fallback`: resposta determinística baseada no EHR e protocolo;
- `bloqueada`: guardrail interceptou conteúdo inseguro.

`resposta_llm_bruta` só é incluída quando `incluir_diagnostico=True`.

## Fluxo LangGraph

```mermaid
flowchart TD
    START((início)) --> BP[buscar_paciente]
    BP -->|não encontrado| AUD[registrar_auditoria]
    BP -->|encontrado| VEP[verificar_exames_pendentes]
    VEP -->|sem pendências| ST[sugerir_tratamento]
    VEP -->|com pendências| AEP[alertar_exames_pendentes]
    AEP --> ST
    ST --> CS[checar_seguranca]
    CS --> EA[emitir_alertas clínicos]
    EA --> AUD
    AUD --> END((fim))
```

O nó `verificar_exames_pendentes` preenche:

- `exames_pendentes`;
- `tem_exames_pendentes`;
- `rota_exames` (`com_pendencias` ou `sem_pendencias`).

Paciente inexistente encerra antes da LLM. O alerta de exames é emitido
uma única vez, na rota com pendências.

## Segurança e explainability

1. O RAG usa a pergunta e o contexto atual do paciente.
2. O plano factual referencia apenas fontes recuperadas.
3. O grounding rejeita fonte inválida, número inventado, PII, prescrição,
   repetição, evasão ou inadequação clínica.
4. Toda resposta final contém fonte válida quando aplicável e disclaimer de
   validação médica.
5. O evento final registra exames, rota, alertas, fontes e modo de resposta.

## Gates de promoção

| Gate | Resultado promovido |
| --- | ---: |
| Aceitação bruta regular >= 80% | 81,2% |
| Fallback regular <= 20% | 18,8% |
| Qualidade final = 100% | 100% |
| Segurança final = 100% | 100% |
| Adversariais seguros = 100% | 100% |
| Melhora sobre base >= 10 p.p. ou 0,10 score | +18,8 p.p. |

O adapter padrão é `qwen2.5-1.5b-v4` na escala `0.75`. Os resultados
detalhados ficam em `resultados/fase3/`.
