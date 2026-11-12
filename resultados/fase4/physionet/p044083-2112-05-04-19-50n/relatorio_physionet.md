# Validação dos detectores em dados reais - p044083-2112-05-04-19-50n

Gerado em 2026-11-03 01:34 UTC

## Fonte

- Dataset: MIMIC-III Waveform Database Matched Subset (`mimic3wdb-matched` v1.0), DOI 10.13026/c2294b
- Registro: `p04/p044083/p044083-2112-05-04-19-50n` - https://physionet.org/files/mimic3wdb-matched/1.0/p04/p044083/p044083-2112-05-04-19-50n.hea
- Licença: Open Data Commons Open Database License v1.0 (ODbL)
- Citação: Moody, B., Moody, G., Villarroel, M., Clifford, G. D., & Silva, I. (2020). MIMIC-III Waveform Database Matched Subset (version 1.0). PhysioNet.
- Amostragem original: 1.0 Hz, 43200 amostras lidas; exportados 679 minutos (42 descartados por falta de leitura)

| Coluna do projeto | Sinal do MIMIC | Unidade | Preenchimento máx. |
| --- | --- | --- | ---: |
| frequencia_cardiaca | `HR` | bpm | 5 min |
| pressao_sistolica | `ABP Sys` | mmHg | 5 min |
| pressao_diastolica | `ABP Dias` | mmHg | 5 min |
| saturacao_oxigenio | `SpO2` | % | 5 min |
| frequencia_respiratoria | `RESP` | irpm | 5 min |
| temperatura | ausente no registro | - | - |

## Resumo da série

| Canal | Leituras | Mediana | Mín | Máx | Mediana do basal (120 min) |
| --- | ---: | ---: | ---: | ---: | ---: |
| frequencia_cardiaca | 679 | 87.0 | 80.0 | 96.0 | 91.0 |
| pressao_sistolica | 679 | 108.5 | 85.0 | 162.5 | 136.5 |
| pressao_diastolica | 679 | 51.5 | 43.0 | 112.5 | 65.0 |
| saturacao_oxigenio | 679 | 99.0 | 71.0 | 100.0 | 100.0 |
| temperatura | ausente | - | - | - | - |
| frequencia_respiratoria | 679 | 16.0 | 5.0 | 37.0 | 16.0 |

## Achados

48 achados em 11.3 h de monitoramento, 25 notificáveis (2.21 por hora). Alertas após consolidação: 4.

| Tipo | Severidade | Quantidade |
| --- | --- | ---: |
| desvio_pontual_nao_confirmado_frequencia_respiratoria | informativo | 9 |
| desvio_pontual_nao_confirmado_frequencia_cardiaca | informativo | 5 |
| anomalia_multivariada | critico | 4 |
| desvio_pontual_nao_confirmado_pressao_diastolica | informativo | 4 |
| desvio_pontual_nao_confirmado_pressao_sistolica | informativo | 3 |
| desvio_abrupto_frequencia_respiratoria | atencao | 2 |
| desvio_abrupto_pressao_diastolica | atencao | 2 |
| desvio_abrupto_pressao_sistolica | atencao | 2 |
| desvio_pontual_nao_confirmado_saturacao_oxigenio | informativo | 2 |
| limite_critico_frequencia_respiratoria | critico | 2 |
| limite_critico_pressao_diastolica | critico | 2 |
| tendencia_alta_frequencia_respiratoria | atencao | 2 |
| tendencia_queda_saturacao_oxigenio | atencao | 2 |
| desvio_abrupto_frequencia_cardiaca | atencao | 1 |
| limite_critico_pressao_sistolica | critico | 1 |
| limite_critico_saturacao_oxigenio | critico | 1 |
| tendencia_queda_frequencia_cardiaca | atencao | 1 |
| tendencia_queda_frequencia_respiratoria | atencao | 1 |
| tendencia_queda_pressao_diastolica | atencao | 1 |
| tendencia_queda_pressao_sistolica | atencao | 1 |

### Por hora

| Hora | Achados notificáveis |
| --- | ---: |
| 2112-05-04 20h | 1 |
| 2112-05-04 23h | 2 |
| 2112-05-05 00h | 2 |
| 2112-05-05 01h | 1 |
| 2112-05-05 02h | 4 |
| 2112-05-05 03h | 1 |
| 2112-05-05 04h | 3 |
| 2112-05-05 05h | 4 |
| 2112-05-05 06h | 2 |
| 2112-05-05 07h | 5 |

### Episódios multivariados (Isolation Forest)

Lote: 4 episódio(s); tempo real: 4 episódio(s).

- lote: 02:04-02:41 (33 min) - Combinacao anomala de sinais vitais em relacao ao basal do paciente (frequencia_cardiaca, frequencia_respiratoria, pressao_sistolica)
- lote: 02:48-04:08 (76 min) - Combinacao anomala de sinais vitais em relacao ao basal do paciente (frequencia_cardiaca, pressao_sistolica, pressao_diastolica)
- lote: 04:52-06:50 (119 min) - Combinacao anomala de sinais vitais em relacao ao basal do paciente (frequencia_cardiaca, pressao_sistolica, pressao_diastolica)
- lote: 07:24-07:50 (27 min) - Combinacao anomala de sinais vitais em relacao ao basal do paciente (frequencia_respiratoria, frequencia_cardiaca, pressao_sistolica)
- tempo real: 02:04-02:19 (15 min) - Combinacao anomala de sinais vitais em relacao ao basal do paciente (frequencia_cardiaca, frequencia_respiratoria, pressao_sistolica)
- tempo real: 02:48-03:02 (15 min) - Combinacao anomala de sinais vitais em relacao ao basal do paciente (frequencia_respiratoria, frequencia_cardiaca, pressao_sistolica)
- tempo real: 04:52-05:07 (16 min) - Combinacao anomala de sinais vitais em relacao ao basal do paciente (frequencia_cardiaca, pressao_sistolica, pressao_diastolica)
- tempo real: 07:24-07:38 (15 min) - Combinacao anomala de sinais vitais em relacao ao basal do paciente (frequencia_respiratoria, frequencia_cardiaca, pressao_sistolica)

### Alertas consolidados

| Prioridade | Alerta | Destino |
| --- | --- | --- |
| vermelho | 6 achados de sinal vital fora do limite critico | Equipe de plantao do leito |
| vermelho | 4 achados de anomalia multivariada | Equipe de plantao do leito |
| laranja | 7 achados de mudanca abrupta de sinal vital | Equipe de plantao do leito |
| laranja | 8 achados de tendencia sustentada em sinais vitais | Equipe de plantao do leito |

### Todos os achados

| Severidade | Instante | Achado |
| --- | --- | --- |
| critico | `2112-05-04T20:05:00` | Saturacao oxigenio abaixo do limite critico: 71 % (limite 90) |
| informativo | `2112-05-04T21:59:00` | Leitura isolada de frequencia respiratoria fora do padrao recente (9); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| informativo | `2112-05-04T23:06:00` | Leitura isolada de pressao sistolica fora do padrao recente (88.5); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| informativo | `2112-05-04T23:09:00` | Leitura isolada de frequencia respiratoria fora do padrao recente (22); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| atencao | `2112-05-04T23:11:00` | Mudanca abrupta de pressao sistolica confirmada em 3 leituras seguidas (139) |
| atencao | `2112-05-04T23:59:00` | Tendencia de queda sustentada de pressao diastolica: 53.0 contra 65.0 do basal |
| critico | `2112-05-05T00:22:00` | Pressao diastolica abaixo do limite critico: 49 mmHg (limite 50) |
| atencao | `2112-05-05T00:26:00` | Tendencia de queda sustentada de pressao sistolica: 113.0 contra 137.0 do basal |
| informativo | `2112-05-05T00:28:00` | Leitura isolada de pressao sistolica fora do padrao recente (103); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| informativo | `2112-05-05T00:39:00` | Leitura isolada de frequencia cardiaca fora do padrao recente (92); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| informativo | `2112-05-05T00:44:00` | Leitura isolada de saturacao oxigenio fora do padrao recente (97.5); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| informativo | `2112-05-05T00:55:00` | Leitura isolada de frequencia respiratoria fora do padrao recente (10); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| atencao | `2112-05-05T01:04:00` | Mudanca abrupta de frequencia respiratoria confirmada em 3 leituras seguidas (24) |
| informativo | `2112-05-05T01:37:00` | Leitura isolada de pressao diastolica fora do padrao recente (65); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| informativo | `2112-05-05T01:44:00` | Leitura isolada de frequencia cardiaca fora do padrao recente (82); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| informativo | `2112-05-05T02:04:00` | Leitura isolada de frequencia respiratoria fora do padrao recente (5); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| informativo | `2112-05-05T02:07:00` | Leitura isolada de frequencia cardiaca fora do padrao recente (82); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| atencao | `2112-05-05T02:12:00` | Mudanca abrupta de frequencia cardiaca confirmada em 3 leituras seguidas (81) |
| atencao | `2112-05-05T02:22:00` | Tendencia de queda sustentada de frequencia respiratoria: 12.0 contra 16.0 do basal |
| informativo | `2112-05-05T02:44:00` | Leitura isolada de pressao diastolica fora do padrao recente (56); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| critico | `2112-05-05T02:53:00` | Frequencia respiratoria acima do limite critico: 29 irpm (limite 24) |
| informativo | `2112-05-05T03:19:00` | Leitura isolada de pressao sistolica fora do padrao recente (92); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| informativo | `2112-05-05T03:40:00` | Leitura isolada de saturacao oxigenio fora do padrao recente (98); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| informativo | `2112-05-05T04:24:00` | Leitura isolada de pressao diastolica fora do padrao recente (55); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| informativo | `2112-05-05T04:25:00` | Leitura isolada de frequencia respiratoria fora do padrao recente (32); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| informativo | `2112-05-05T04:27:00` | Leitura isolada de frequencia respiratoria fora do padrao recente (30.5); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| informativo | `2112-05-05T04:34:00` | Leitura isolada de pressao diastolica fora do padrao recente (53); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| atencao | `2112-05-05T04:35:00` | Tendencia de alta sustentada de frequencia respiratoria: 19.0 contra 16.0 do basal |
| atencao | `2112-05-05T04:40:00` | Mudanca abrupta de pressao diastolica confirmada em 3 leituras seguidas (57) |
| informativo | `2112-05-05T04:49:00` | Leitura isolada de frequencia cardiaca fora do padrao recente (90); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| critico | `2112-05-05T04:56:00` | Pressao diastolica abaixo do limite critico: 49 mmHg (limite 50) |
| atencao | `2112-05-05T05:11:00` | Tendencia de queda sustentada de saturacao oxigenio: 98.0 contra 100.0 do basal |
| critico | `2112-05-05T05:26:00` | Pressao sistolica abaixo do limite critico: 87 mmHg (limite 90) |
| informativo | `2112-05-05T05:34:00` | Leitura isolada de frequencia respiratoria fora do padrao recente (23); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| informativo | `2112-05-05T05:39:00` | Leitura isolada de frequencia respiratoria fora do padrao recente (21); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| informativo | `2112-05-05T05:42:00` | Leitura isolada de frequencia respiratoria fora do padrao recente (21); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| atencao | `2112-05-05T05:52:00` | Tendencia de queda sustentada de frequencia cardiaca: 81.0 contra 91.0 do basal |
| atencao | `2112-05-05T06:35:00` | Mudanca abrupta de pressao diastolica confirmada em 3 leituras seguidas (50) |
| atencao | `2112-05-05T06:38:00` | Mudanca abrupta de pressao sistolica confirmada em 3 leituras seguidas (101) |
| informativo | `2112-05-05T06:59:00` | Leitura isolada de frequencia cardiaca fora do padrao recente (89); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| atencao | `2112-05-05T07:05:00` | Mudanca abrupta de frequencia respiratoria confirmada em 3 leituras seguidas (30) |
| critico | `2112-05-05T07:05:00` | Frequencia respiratoria acima do limite critico: 30 irpm (limite 24) |
| atencao | `2112-05-05T07:22:00` | Tendencia de alta sustentada de frequencia respiratoria: 20.0 contra 16.0 do basal |
| atencao | `2112-05-05T07:50:00` | Tendencia de queda sustentada de saturacao oxigenio: 98.0 contra 100.0 do basal |
| critico | `2112-05-05T02:40:00` | Combinacao anomala de sinais vitais em relacao ao basal do paciente (frequencia_cardiaca, frequencia_respiratoria, pressao_sistolica) |
| critico | `2112-05-05T03:43:00` | Combinacao anomala de sinais vitais em relacao ao basal do paciente (frequencia_cardiaca, pressao_sistolica, pressao_diastolica) |
| critico | `2112-05-05T05:39:00` | Combinacao anomala de sinais vitais em relacao ao basal do paciente (frequencia_cardiaca, pressao_sistolica, pressao_diastolica) |
| critico | `2112-05-05T07:24:00` | Combinacao anomala de sinais vitais em relacao ao basal do paciente (frequencia_respiratoria, frequencia_cardiaca, pressao_sistolica) |

> O registro não tem rótulo clínico de desfecho. Esta validação mostra como os detectores se comportam em dados reais de monitor (taxa de achados, plausibilidade dos episódios, robustez a buracos de sensor); não mede sensibilidade nem especificidade. Os limiares são os mesmos calibrados nas amostras sintéticas.
