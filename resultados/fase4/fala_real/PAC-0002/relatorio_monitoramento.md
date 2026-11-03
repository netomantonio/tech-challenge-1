# Relatório de monitoramento multimodal - PAC-0002

Gerado em 2026-11-02 23:58 UTC

**Risco calculado:** 17.3/100 (nível **baixo**)

Modalidades processadas: audio, sinais_vitais, prescricoes, movimentacao  
Modalidades ausentes: video

## Achados por modalidade

| Modalidade | Achados |
| --- | ---: |
| audio | 1 |
| sinais_vitais | 5 |

| Severidade | Achados |
| --- | ---: |
| informativo | 5 |
| atencao | 1 |
| critico | 0 |

## Quadros identificados pela fusão

Nenhuma combinação entre modalidades diferentes atingiu os critérios das regras de fusão.

## Alertas gerados para a equipe

| Prioridade | Alerta | Destino | Instante |
| --- | --- | --- | --- |
| laranja | Frequencia fundamental instavel, compativel com voz tremula ou articulacao imprecisa (indicador acustico de disartria) | Medico assistente da consulta | `-` |

## Todos os achados

| Modalidade | Severidade | Instante | Achado |
| --- | --- | --- | --- |
| audio | atencao | `-` | Frequencia fundamental instavel, compativel com voz tremula ou articulacao imprecisa (indicador acustico de disartria) |
| sinais_vitais | informativo | `2025-06-10T09:49:00` | Leitura isolada de pressao sistolica fora do padrao recente (117.8); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2025-06-10T10:06:00` | Leitura isolada de pressao sistolica fora do padrao recente (135.6); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2025-06-10T12:07:00` | Leitura isolada de frequencia respiratoria fora do padrao recente (18.4); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2025-06-10T13:29:00` | Leitura isolada de pressao diastolica fora do padrao recente (87.2); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2025-06-10T15:56:00` | Leitura isolada de frequencia respiratoria fora do padrao recente (14.7); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |

## Serviços utilizados

| Serviço | Provedor efetivo |
| --- | --- |
| speech_to_text | `local` |
| text_analytics | `offline` |

> Protótipo acadêmico. Nenhum alerta deste relatório substitui avaliação clínica.
