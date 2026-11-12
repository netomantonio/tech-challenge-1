# Relatório de monitoramento multimodal - p044083-2112-05-04-19-50n

Gerado em 2026-11-03 01:34 UTC

**Risco calculado:** 98.6/100 (nível **alto**)

Modalidades processadas: sinais_vitais  
Modalidades ausentes: video, audio, prescricoes, movimentacao

## Achados por modalidade

| Modalidade | Achados |
| --- | ---: |
| sinais_vitais | 48 |

| Severidade | Achados |
| --- | ---: |
| informativo | 23 |
| atencao | 15 |
| critico | 10 |

## Quadros identificados pela fusão

Nenhuma combinação entre modalidades diferentes atingiu os critérios das regras de fusão.

## Alertas gerados para a equipe

| Prioridade | Alerta | Destino | Instante | Entregue via |
| --- | --- | --- | --- | --- |
| vermelho | 6 achados de sinal vital fora do limite critico | Equipe de plantao do leito | `2112-05-04T20:05:00` | fila_jsonl, painel_web |
| vermelho | 4 achados de anomalia multivariada | Equipe de plantao do leito | `2112-05-05T02:40:00` | fila_jsonl, painel_web |
| laranja | 7 achados de mudanca abrupta de sinal vital | Equipe de plantao do leito | `2112-05-05T06:35:00` | fila_jsonl, painel_web |
| laranja | 8 achados de tendencia sustentada em sinais vitais | Equipe de plantao do leito | `2112-05-05T02:22:00` | fila_jsonl, painel_web |

## Todos os achados

| Modalidade | Severidade | Instante | Achado |
| --- | --- | --- | --- |
| sinais_vitais | critico | `2112-05-04T20:05:00` | Saturacao oxigenio abaixo do limite critico: 71 % (limite 90) |
| sinais_vitais | informativo | `2112-05-04T21:59:00` | Leitura isolada de frequencia respiratoria fora do padrao recente (9); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2112-05-04T23:06:00` | Leitura isolada de pressao sistolica fora do padrao recente (88.5); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2112-05-04T23:09:00` | Leitura isolada de frequencia respiratoria fora do padrao recente (22); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | atencao | `2112-05-04T23:11:00` | Mudanca abrupta de pressao sistolica confirmada em 3 leituras seguidas (139) |
| sinais_vitais | atencao | `2112-05-04T23:59:00` | Tendencia de queda sustentada de pressao diastolica: 53.0 contra 65.0 do basal |
| sinais_vitais | critico | `2112-05-05T00:22:00` | Pressao diastolica abaixo do limite critico: 49 mmHg (limite 50) |
| sinais_vitais | atencao | `2112-05-05T00:26:00` | Tendencia de queda sustentada de pressao sistolica: 113.0 contra 137.0 do basal |
| sinais_vitais | informativo | `2112-05-05T00:28:00` | Leitura isolada de pressao sistolica fora do padrao recente (103); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2112-05-05T00:39:00` | Leitura isolada de frequencia cardiaca fora do padrao recente (92); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2112-05-05T00:44:00` | Leitura isolada de saturacao oxigenio fora do padrao recente (97.5); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2112-05-05T00:55:00` | Leitura isolada de frequencia respiratoria fora do padrao recente (10); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | atencao | `2112-05-05T01:04:00` | Mudanca abrupta de frequencia respiratoria confirmada em 3 leituras seguidas (24) |
| sinais_vitais | informativo | `2112-05-05T01:37:00` | Leitura isolada de pressao diastolica fora do padrao recente (65); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2112-05-05T01:44:00` | Leitura isolada de frequencia cardiaca fora do padrao recente (82); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2112-05-05T02:04:00` | Leitura isolada de frequencia respiratoria fora do padrao recente (5); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2112-05-05T02:07:00` | Leitura isolada de frequencia cardiaca fora do padrao recente (82); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | atencao | `2112-05-05T02:12:00` | Mudanca abrupta de frequencia cardiaca confirmada em 3 leituras seguidas (81) |
| sinais_vitais | atencao | `2112-05-05T02:22:00` | Tendencia de queda sustentada de frequencia respiratoria: 12.0 contra 16.0 do basal |
| sinais_vitais | informativo | `2112-05-05T02:44:00` | Leitura isolada de pressao diastolica fora do padrao recente (56); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | critico | `2112-05-05T02:53:00` | Frequencia respiratoria acima do limite critico: 29 irpm (limite 24) |
| sinais_vitais | informativo | `2112-05-05T03:19:00` | Leitura isolada de pressao sistolica fora do padrao recente (92); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2112-05-05T03:40:00` | Leitura isolada de saturacao oxigenio fora do padrao recente (98); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2112-05-05T04:24:00` | Leitura isolada de pressao diastolica fora do padrao recente (55); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2112-05-05T04:25:00` | Leitura isolada de frequencia respiratoria fora do padrao recente (32); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2112-05-05T04:27:00` | Leitura isolada de frequencia respiratoria fora do padrao recente (30.5); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2112-05-05T04:34:00` | Leitura isolada de pressao diastolica fora do padrao recente (53); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | atencao | `2112-05-05T04:35:00` | Tendencia de alta sustentada de frequencia respiratoria: 19.0 contra 16.0 do basal |
| sinais_vitais | atencao | `2112-05-05T04:40:00` | Mudanca abrupta de pressao diastolica confirmada em 3 leituras seguidas (57) |
| sinais_vitais | informativo | `2112-05-05T04:49:00` | Leitura isolada de frequencia cardiaca fora do padrao recente (90); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | critico | `2112-05-05T04:56:00` | Pressao diastolica abaixo do limite critico: 49 mmHg (limite 50) |
| sinais_vitais | atencao | `2112-05-05T05:11:00` | Tendencia de queda sustentada de saturacao oxigenio: 98.0 contra 100.0 do basal |
| sinais_vitais | critico | `2112-05-05T05:26:00` | Pressao sistolica abaixo do limite critico: 87 mmHg (limite 90) |
| sinais_vitais | informativo | `2112-05-05T05:34:00` | Leitura isolada de frequencia respiratoria fora do padrao recente (23); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2112-05-05T05:39:00` | Leitura isolada de frequencia respiratoria fora do padrao recente (21); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2112-05-05T05:42:00` | Leitura isolada de frequencia respiratoria fora do padrao recente (21); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | atencao | `2112-05-05T05:52:00` | Tendencia de queda sustentada de frequencia cardiaca: 81.0 contra 91.0 do basal |
| sinais_vitais | atencao | `2112-05-05T06:35:00` | Mudanca abrupta de pressao diastolica confirmada em 3 leituras seguidas (50) |
| sinais_vitais | atencao | `2112-05-05T06:38:00` | Mudanca abrupta de pressao sistolica confirmada em 3 leituras seguidas (101) |
| sinais_vitais | informativo | `2112-05-05T06:59:00` | Leitura isolada de frequencia cardiaca fora do padrao recente (89); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | atencao | `2112-05-05T07:05:00` | Mudanca abrupta de frequencia respiratoria confirmada em 3 leituras seguidas (30) |
| sinais_vitais | critico | `2112-05-05T07:05:00` | Frequencia respiratoria acima do limite critico: 30 irpm (limite 24) |
| sinais_vitais | atencao | `2112-05-05T07:22:00` | Tendencia de alta sustentada de frequencia respiratoria: 20.0 contra 16.0 do basal |
| sinais_vitais | atencao | `2112-05-05T07:50:00` | Tendencia de queda sustentada de saturacao oxigenio: 98.0 contra 100.0 do basal |
| sinais_vitais | critico | `2112-05-05T02:40:00` | Combinacao anomala de sinais vitais em relacao ao basal do paciente (frequencia_cardiaca, frequencia_respiratoria, pressao_sistolica) |
| sinais_vitais | critico | `2112-05-05T03:43:00` | Combinacao anomala de sinais vitais em relacao ao basal do paciente (frequencia_cardiaca, pressao_sistolica, pressao_diastolica) |
| sinais_vitais | critico | `2112-05-05T05:39:00` | Combinacao anomala de sinais vitais em relacao ao basal do paciente (frequencia_cardiaca, pressao_sistolica, pressao_diastolica) |
| sinais_vitais | critico | `2112-05-05T07:24:00` | Combinacao anomala de sinais vitais em relacao ao basal do paciente (frequencia_respiratoria, frequencia_cardiaca, pressao_sistolica) |

## Serviços utilizados

| Serviço | Provedor efetivo |
| --- | --- |
| speech_to_text | `nao_executado` |
| text_analytics | `nao_executado` |

> Protótipo acadêmico. Nenhum alerta deste relatório substitui avaliação clínica.
