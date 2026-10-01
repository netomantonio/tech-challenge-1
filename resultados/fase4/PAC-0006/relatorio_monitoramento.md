# Relatório de monitoramento multimodal - PAC-0006

Gerado em 2026-10-01 16:00 UTC

**Risco calculado:** 100.0/100 (nível **alto**)

Modalidades processadas: video, audio, sinais_vitais, prescricoes, movimentacao  
Modalidades ausentes: nenhuma

## Achados por modalidade

| Modalidade | Achados |
| --- | ---: |
| audio | 9 |
| movimentacao | 2 |
| prescricao | 3 |
| sinais_vitais | 18 |
| video | 9 |

| Severidade | Achados |
| --- | ---: |
| informativo | 10 |
| atencao | 19 |
| critico | 12 |

## Quadros identificados pela fusão

Uma mesma família de achado pode sustentar mais de um quadro: a piora dos sinais vitais, por exemplo, entra tanto na suspeita de infecção quanto no risco medicamentoso. Os achados individuais estão listados na última seção.

### deterioracao respiratoria

Queixa respiratoria na consulta junto com piora objetiva de oxigenacao ou ventilacao.

- Severidade: **critico**
- Modalidades envolvidas: audio, sinais_vitais
- Famílias que sustentam (16 achados): `tendencia_sinais_vitais` (6), `queixas_criticas` (5), `alteracoes_vocais` (3), `limite_critico_sinais_vitais` (2)
- Conduta sugerida: Avaliar o paciente presencialmente, repetir oximetria e considerar suporte de oxigenio conforme protocolo

### suspeita infeccao

Relato de febre ou alteracao da ferida junto com febre medida e repercussao hemodinamica.

- Severidade: **critico**
- Modalidades envolvidas: audio, sinais_vitais
- Famílias que sustentam (14 achados): `tendencia_sinais_vitais` (6), `queixas_criticas` (5), `limite_critico_sinais_vitais` (2), `anomalia_multivariada` (1)
- Conduta sugerida: Coletar culturas, reavaliar a ferida operatoria e checar o esquema antimicrobiano antes da proxima dose

### risco medicamentoso

Alteracao inesperada de prescricao em paciente que ja apresenta piora clinica ou queixa de dor.

- Severidade: **critico**
- Modalidades envolvidas: prescricao, audio, sinais_vitais
- Famílias que sustentam (17 achados): `tendencia_sinais_vitais` (6), `queixas_criticas` (5), `alteracao_prescricao` (3), `limite_critico_sinais_vitais` (2), `anomalia_multivariada` (1)
- Conduta sugerida: Revisar a prescricao com a farmacia clinica antes da proxima administracao e registrar a justificativa no prontuario

### reabilitacao insegura

Execucao do exercicio fora do protocolo em paciente que relata dor ou cansaco na mesma janela.

- Severidade: **critico**
- Modalidades envolvidas: video, audio
- Famílias que sustentam (16 achados): `execucao_exercicio` (7), `queixas_criticas` (5), `alteracoes_vocais` (3), `invasao_area_critica` (1)
- Conduta sugerida: Rever o plano de reabilitacao com o fisioterapeuta antes da proxima sessao e reavaliar a liberacao de amplitude

### risco de imobilidade

Imobilidade prolongada no leito em paciente com piora de sinais vitais ou agitacao noturna.

- Severidade: **critico**
- Modalidades envolvidas: sinais_vitais, movimentacao
- Famílias que sustentam (11 achados): `tendencia_sinais_vitais` (6), `limite_critico_sinais_vitais` (2), `padrao_movimentacao` (2), `anomalia_multivariada` (1)
- Conduta sugerida: Reavaliar risco de lesao por pressao e de trombose e reforcar mobilizacao assistida

## Alertas gerados para a equipe

| Prioridade | Alerta | Destino | Instante |
| --- | --- | --- | --- |
| vermelho | Queixa respiratoria na consulta junto com piora objetiva de oxigenacao ou ventilacao | Medico de plantao e enfermagem do leito; Medico assistente da consulta; Equipe de plantao do leito | `2025-06-10T16:51:00` |
| vermelho | Alteracao inesperada de prescricao em paciente que ja apresenta piora clinica ou queixa de dor | Medico de plantao e enfermagem do leito; Farmacia clinica e medico prescritor; Medico assistente da consulta; Equipe de plantao do leito | `2025-06-10T14:05:00` |
| vermelho | Execucao do exercicio fora do protocolo em paciente que relata dor ou cansaco na mesma janela | Medico de plantao e enfermagem do leito; Equipe de reabilitacao e cirurgia responsavel; Medico assistente da consulta | `00:13.20` |
| vermelho | Relato de febre ou alteracao da ferida junto com febre medida e repercussao hemodinamica | Medico de plantao e enfermagem do leito; Medico assistente da consulta; Equipe de plantao do leito | `2025-06-10T16:51:00` |
| vermelho | Imobilidade prolongada no leito em paciente com piora de sinais vitais ou agitacao noturna | Medico de plantao e enfermagem do leito; Equipe de plantao do leito; Equipe de enfermagem do leito | `2025-06-10T02:00:00` |
| laranja | Paciente executou o exercicio sem profissional em cena por 4.0 s | Equipe de reabilitacao e cirurgia responsavel | `00:10.00` |

## Todos os achados

| Modalidade | Severidade | Instante | Achado |
| --- | --- | --- | --- |
| video | atencao | `00:11.40` | Compensacao de tronco acima do tolerado durante o exercicio (19.4 graus) |
| video | atencao | `00:15.07` | Compensacao de tronco acima do tolerado durante o exercicio (17.8 graus) |
| video | critico | `00:11.60` | Amplitude de abducao do ombro esquerdo acima da liberada no pos-operatorio (139.6 graus) |
| video | critico | `00:14.87` | Amplitude de abducao do ombro esquerdo acima da liberada no pos-operatorio (133.8 graus) |
| video | atencao | `00:11.77` | Assimetria entre os ombros na mesma repeticao (49.2 graus) |
| video | atencao | `00:13.37` | Assimetria entre os ombros na mesma repeticao (25.9 graus) |
| video | atencao | `00:14.90` | Assimetria entre os ombros na mesma repeticao (40.1 graus) |
| video | critico | `00:13.20` | Mao do paciente dentro da area critica do dreno durante a sessao |
| video | atencao | `00:10.00` | Paciente executou o exercicio sem profissional em cena por 4.0 s |
| audio | atencao | `00:00.60` | Fala entrecortada com frases curtas e pausas longas, compativel com esforco respiratorio |
| audio | atencao | `00:16.05` | Sinais vocais de cansaco: queda de energia ao longo da consulta e taxa de fala reduzida |
| audio | atencao | `-` | Frequencia fundamental instavel, compativel com voz tremula ou articulacao prejudicada |
| audio | critico | `00:03.37` | Queixa de categoria 'respiratorio' relatada pelo paciente: sem ar |
| audio | critico | `00:05.75` | Queixa de categoria 'dor_toracica' relatada pelo paciente: aperto no peito |
| audio | critico | `00:08.65` | Queixa de categoria 'infeccao' relatada pelo paciente: febre |
| audio | atencao | `00:00.60` | Queixa de categoria 'fadiga' relatada pelo paciente: muito cansada, nao aguento |
| audio | atencao | `00:11.44` | Queixa de categoria 'dor' relatada pelo paciente: dor aumentou |
| audio | informativo | `-` | Sentimento predominante negativo na fala do paciente |
| sinais_vitais | informativo | `2025-06-10T09:49:00` | Leitura isolada de pressao sistolica fora do padrao recente (117.8); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2025-06-10T10:06:00` | Leitura isolada de pressao sistolica fora do padrao recente (135.6); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2025-06-10T10:30:00` | Leitura isolada de frequencia cardiaca fora do padrao recente (185); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2025-06-10T12:07:00` | Leitura isolada de frequencia respiratoria fora do padrao recente (18.4); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2025-06-10T13:29:00` | Leitura isolada de pressao diastolica fora do padrao recente (87.2); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2025-06-10T15:56:00` | Leitura isolada de frequencia respiratoria fora do padrao recente (14.7); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2025-06-10T16:10:00` | Leitura isolada de frequencia respiratoria fora do padrao recente (18.7); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2025-06-10T16:23:00` | Leitura isolada de frequencia respiratoria fora do padrao recente (19.1); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | informativo | `2025-06-10T16:26:00` | Leitura isolada de frequencia cardiaca fora do padrao recente (96.4); nao se repetiu nas leituras seguintes e foi tratada como artefato de sensor |
| sinais_vitais | atencao | `2025-06-10T16:51:00` | Tendencia de alta sustentada de frequencia cardiaca: 93.5 contra 82.7 do basal |
| sinais_vitais | atencao | `2025-06-10T17:01:00` | Tendencia de alta sustentada de temperatura: 37.3 contra 36.9 do basal |
| sinais_vitais | atencao | `2025-06-10T17:07:00` | Tendencia de alta sustentada de frequencia respiratoria: 19.2 contra 16.2 do basal |
| sinais_vitais | atencao | `2025-06-10T17:20:00` | Tendencia de queda sustentada de saturacao oxigenio: 95.0 contra 97.1 do basal |
| sinais_vitais | atencao | `2025-06-10T17:43:00` | Tendencia de queda sustentada de pressao sistolica: 112.8 contra 125.1 do basal |
| sinais_vitais | atencao | `2025-06-10T17:44:00` | Tendencia de queda sustentada de pressao diastolica: 70.1 contra 78.6 do basal |
| sinais_vitais | critico | `2025-06-10T18:03:00` | Temperatura acima do limite critico: 38.08 C (limite 38) |
| sinais_vitais | critico | `2025-06-10T18:17:00` | Frequencia respiratoria acima do limite critico: 24.2 irpm (limite 24) |
| sinais_vitais | critico | `2025-06-10T17:08:00` | Combinacao anomala de sinais vitais em relacao ao basal do paciente (frequencia_cardiaca, temperatura, frequencia_respiratoria) |
| prescricao | critico | `2025-06-10T14:05:00` | Aumento de 400% na dose de morfina (2 para 10 mg) |
| prescricao | critico | `2025-06-10T15:40:00` | ceftriaxona suspenso no dia 2 de um esquema previsto para 7 dias |
| prescricao | critico | `2025-06-10T16:20:00` | Dois medicamentos da classe anti_inflamatorio_nao_esteroidal ativos ao mesmo tempo: cetoprofeno, ibuprofeno |
| movimentacao | atencao | `2025-06-10T09:00:00` | Paciente praticamente imovel por 6 horas seguidas durante o dia |
| movimentacao | atencao | `2025-06-10T02:00:00` | Atividade elevada em 3 horas da madrugada, padrao compativel com agitacao ou delirium |
| multimodal | critico | `2025-06-10T16:51:00` | Queixa respiratoria na consulta junto com piora objetiva de oxigenacao ou ventilacao |
| multimodal | critico | `2025-06-10T16:51:00` | Relato de febre ou alteracao da ferida junto com febre medida e repercussao hemodinamica |
| multimodal | critico | `2025-06-10T14:05:00` | Alteracao inesperada de prescricao em paciente que ja apresenta piora clinica ou queixa de dor |
| multimodal | critico | `00:13.20` | Execucao do exercicio fora do protocolo em paciente que relata dor ou cansaco na mesma janela |
| multimodal | critico | `2025-06-10T02:00:00` | Imobilidade prolongada no leito em paciente com piora de sinais vitais ou agitacao noturna |

## Serviços utilizados

| Serviço | Provedor efetivo |
| --- | --- |
| speech_to_text | `offline` |
| text_analytics | `offline` |

> Protótipo acadêmico. Nenhum alerta deste relatório substitui avaliação clínica.
