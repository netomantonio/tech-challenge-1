# Relatório de análise de áudio - PAC-0006

Arquivo: `consulta_fala.wav`  
Duração: 28.8 s  
Transcrição: provedor `azure`, confiança média 0.89, taxa de erro de palavras contra a referência 0.0%  
Análise de texto: provedor `azure`

## Atributos acústicos

| Atributo | Valor |
| --- | ---: |
| Tempo efetivo de fala | 11.2 s |
| Proporção de pausa | 0.61 |
| Frases detectadas | 30 |
| Duração média da frase | 0.37 s |
| Pausa máxima | 3.19 s |
| F0 média | 186.3 Hz |
| Jitter da F0 | 0.0239 |
| Taxa de fala | 2.81 sílabas/s |
| Queda de energia | 0.07 |

## Transcrição

- `00:00.51` Doutora, desde ontem eu estou muito cansada.
- `00:05.75` Quando eu levanto da cama, eu fico sem ar.
- `00:10.23` Hoje de manhã senti um aperto no peito.
- `00:14.63` A ferida está mais quente e tive febre de noite.
- `00:19.87` A dor aumentou e eu não aguento fazer os exercícios.
- `00:25.10` Preciso parar um pouco para respirar.

## Sentimento e termos críticos

Sentimento: **negativo**
Frases-chave: exercícios, Doutora, ontem, cama, ar, aperto, peito, ferida, febre, noite, dor

| Termo | Categoria | Peso |
| --- | --- | ---: |
| sem ar | respiratorio | 0.90 |
| aperto no peito | dor_toracica | 0.90 |
| febre | infeccao | 0.80 |
| muito cansada | fadiga | 0.60 |
| nao aguento | fadiga | 0.60 |
| dor aumentou | dor | 0.60 |
| ferida | sintoma_relatado_azure | 0.45 |
| parar um pouco para respirar | sintoma_relatado_azure | 0.45 |

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
| ferida | SymptomOrSign | 0.96 |
| mais quente | SymptomOrSign | 0.65 |
| febre | SymptomOrSign | 0.99 |
| noite | Time | 0.91 |
| dor | SymptomOrSign | 0.99 |
| aumentou | Course | 0.99 |
| parar um pouco para respirar | SymptomOrSign | 0.82 |

## Achados

- **critico** (`00:05.75`) Queixa de categoria 'respiratorio' relatada pelo paciente: sem ar (confiança 0.90)
- **critico** (`00:10.23`) Queixa de categoria 'dor_toracica' relatada pelo paciente: aperto no peito (confiança 0.90)
- **critico** (`00:14.63`) Queixa de categoria 'infeccao' relatada pelo paciente: febre (confiança 0.80)
- **atencao** (`00:00.51`) Queixa de categoria 'fadiga' relatada pelo paciente: muito cansada, nao aguento (confiança 0.60)
- **atencao** (`00:19.87`) Queixa de categoria 'dor' relatada pelo paciente: dor aumentou (confiança 0.60)
- **informativo** (`00:14.63`) Queixa de categoria 'sintoma_relatado_azure' relatada pelo paciente: ferida, parar um pouco para respirar (confiança 0.45)
- **informativo** Sentimento predominante negativo na fala do paciente (confiança 0.99)

> Relatório gerado automaticamente por um protótipo acadêmico. Não substitui avaliação clínica.


> Audio de fala sintetizada (TTS): atributos acusticos apenas descritivos; a avaliacao de alteracoes vocais clinicas nao foi aplicada.