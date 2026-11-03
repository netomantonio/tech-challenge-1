# Relatório de análise de áudio - PAC-0006

Arquivo: `consulta_fala.wav`  
Duração: 24.9 s  
Transcrição: provedor `azure`, confiança média 0.90, taxa de erro de palavras contra a referência 4.0%  
Análise de texto: provedor `azure`

## Atributos acústicos

| Atributo | Valor |
| --- | ---: |
| Tempo efetivo de fala | 7.5 s |
| Proporção de pausa | 0.70 |
| Frases detectadas | 22 |
| Duração média da frase | 0.34 s |
| Pausa máxima | 2.84 s |
| F0 média | 186.7 Hz |
| Jitter da F0 | 0.0219 |
| Taxa de fala | 2.81 sílabas/s |
| Queda de energia | 0.08 |

## Transcrição

- `00:00.51` Doutora, desde ontem eu estou muito cansada.
- `00:05.03` Quando eu levanto da cama, eu fico sem ar.
- `00:08.87` Hoje de manhã senti um aperto no peito.
- `00:12.71` Aferida esta mais quente e tive febre de noite.
- `00:17.21` A dor aumentou e eu não aguento fazer os exercícios.
- `00:21.71` Preciso parar um pouco para respirar.

## Sentimento e termos críticos

Sentimento: **negativo**
Frases-chave: exercícios, Doutora, ontem, cama, ar, manhã, aperto, peito, Aferida, febre, noite, dor

| Termo | Categoria | Peso |
| --- | --- | ---: |
| sem ar | respiratorio | 0.90 |
| aperto no peito | dor_toracica | 0.90 |
| chiado no peito | respiratorio | 0.90 |
| dor no peito | dor_toracica | 0.90 |
| peso no peito | dor_toracica | 0.90 |
| pontada no peito | dor_toracica | 0.90 |
| febre | infeccao | 0.80 |
| muito cansada | fadiga | 0.60 |
| nao aguento | fadiga | 0.60 |
| dor aumentou | dor | 0.60 |
| muito cansado | fadiga | 0.60 |
| doendo muito | dor | 0.60 |
| dor forte | dor | 0.60 |
| dor demais | dor | 0.60 |
| dor insuportavel | dor | 0.60 |
| parar um pouco para respirar | sintoma_relatado_azure | 0.45 |
| muito ansiosa | sono_humor | 0.40 |
| muito ansioso | sono_humor | 0.40 |
| nao consigo dormir | sono_humor | 0.40 |

Entidades clínicas reconhecidas pela Azure (1 documento(s) enviado(s)):

| Entidade | Categoria | Confiança |
| --- | --- | ---: |
| ontem | Time | 0.99 |
| muito | ConditionQualifier | 0.71 |
| cansada | SymptomOrSign | 0.99 |
| eu fico sem ar | SymptomOrSign | 0.80 |
| Hoje de manhã | Time | 0.95 |
| aperto no | SymptomOrSign | 0.86 |
| peito | BodyStructure | 0.57 |
| febre | SymptomOrSign | 0.98 |
| noite | Time | 0.90 |
| dor | SymptomOrSign | 0.99 |
| aumentou | Course | 0.99 |
| parar um pouco para respirar | SymptomOrSign | 0.82 |

## Achados

- **atencao** (`00:00.79`) Fala entrecortada com frases curtas e pausas longas, compativel com esforco respiratorio (confiança 0.95)
- **atencao** Frequencia fundamental instavel, compativel com voz tremula ou articulacao imprecisa (indicador acustico de disartria) (confiança 0.90)
- **critico** (`00:05.03`) Queixa de categoria 'respiratorio' relatada pelo paciente: chiado no peito, sem ar (confiança 0.90)
- **critico** (`00:08.87`) Queixa de categoria 'dor_toracica' relatada pelo paciente: aperto no peito, dor no peito, peso no peito, pontada no peito (confiança 0.90)
- **critico** (`00:12.71`) Queixa de categoria 'infeccao' relatada pelo paciente: febre (confiança 0.80)
- **atencao** (`00:00.51`) Queixa de categoria 'fadiga' relatada pelo paciente: muito cansada, muito cansado, nao aguento (confiança 0.60)
- **atencao** (`00:17.21`) Queixa de categoria 'dor' relatada pelo paciente: doendo muito, dor aumentou, dor demais, dor forte, dor insuportavel (confiança 0.60)
- **informativo** (`00:21.71`) Queixa de categoria 'sintoma_relatado_azure' relatada pelo paciente: parar um pouco para respirar (confiança 0.45)
- **informativo** Queixa de categoria 'sono_humor' relatada pelo paciente: muito ansiosa, muito ansioso, nao consigo dormir (confiança 0.40)
- **informativo** Sentimento predominante negativo na fala do paciente (confiança 0.98)

> Relatório gerado automaticamente por um protótipo acadêmico. Não substitui avaliação clínica.
