# Relatório de análise de áudio - PAC-0006

Arquivo: `consulta.wav`  
Duração: 16.1 s  
Transcrição: provedor `offline`, confiança média 0.88  
Análise de texto: provedor `offline`

## Atributos acústicos

| Atributo | Valor |
| --- | ---: |
| Tempo efetivo de fala | 7.8 s |
| Proporção de pausa | 0.52 |
| Frases detectadas | 6 |
| Duração média da frase | 1.30 s |
| Pausa máxima | 1.42 s |
| F0 média | 173.4 Hz |
| Jitter da F0 | 0.0038 |
| Taxa de fala | 1.31 sílabas/s |
| Queda de energia | 0.42 |

## Transcrição

- `00:00.60` Doutora, desde ontem eu estou muito cansada.
- `00:03.37` Quando eu levanto da cama eu fico sem ar.
- `00:05.75` Hoje de manha senti um aperto no peito.
- `00:08.65` A ferida esta mais quente e tive febre de noite.
- `00:11.44` A dor aumentou e eu nao aguento fazer os exercicios.
- `00:14.08` Preciso parar um pouco para respirar.

## Sentimento e termos críticos

Sentimento: **negativo**
Frases-chave: desde, cansada, levanto, cama, fico, manha, senti, aperto

| Termo | Categoria | Peso |
| --- | --- | ---: |
| sem ar | respiratorio | 0.90 |
| aperto no peito | dor_toracica | 0.90 |
| febre | infeccao | 0.80 |
| muito cansada | fadiga | 0.60 |
| nao aguento | fadiga | 0.60 |
| dor aumentou | dor | 0.60 |

## Achados

- **atencao** (`00:00.60`) Fala entrecortada com frases curtas e pausas longas, compativel com esforco respiratorio (confiança 0.74)
- **atencao** (`00:16.05`) Sinais vocais de cansaco: queda de energia ao longo da consulta e taxa de fala reduzida (confiança 0.95)
- **atencao** Frequencia fundamental instavel, compativel com voz tremula ou articulacao prejudicada (confiança 0.63)
- **critico** (`00:03.37`) Queixa de categoria 'respiratorio' relatada pelo paciente: sem ar (confiança 0.90)
- **critico** (`00:05.75`) Queixa de categoria 'dor_toracica' relatada pelo paciente: aperto no peito (confiança 0.90)
- **critico** (`00:08.65`) Queixa de categoria 'infeccao' relatada pelo paciente: febre (confiança 0.80)
- **atencao** (`00:00.60`) Queixa de categoria 'fadiga' relatada pelo paciente: muito cansada, nao aguento (confiança 0.60)
- **atencao** (`00:11.44`) Queixa de categoria 'dor' relatada pelo paciente: dor aumentou (confiança 0.60)
- **informativo** Sentimento predominante negativo na fala do paciente (confiança 0.67)

> Relatório gerado automaticamente por um protótipo acadêmico. Não substitui avaliação clínica.
