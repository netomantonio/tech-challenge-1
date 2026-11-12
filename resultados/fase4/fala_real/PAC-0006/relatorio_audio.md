# Relatório de análise de áudio - PAC-0006

Arquivo: `consulta_fala.wav`  
Duração: 24.9 s  
Transcrição: provedor `local`, confiança média 0.90, taxa de erro de palavras contra a referência 0.0%  
Análise de texto: provedor `offline`

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

- `00:00.14` Doutora, desde ontem eu estou muito cansada.
- `00:04.14` Quando eu levanto da cama eu fico sem ar.
- `00:08.14` Hoje de manha senti um aperto no peito.
- `00:12.14` A ferida está mais quente e tive febre de noite.
- `00:16.14` A dor aumentou e eu não aguento fazer os exercícios.
- `00:20.14` Preciso parar um pouco para respirar.

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

- **atencao** (`00:00.79`) Fala entrecortada com frases curtas e pausas longas, compativel com esforco respiratorio (confiança 0.95)
- **atencao** Frequencia fundamental instavel, compativel com voz tremula ou articulacao imprecisa (indicador acustico de disartria) (confiança 0.90)
- **critico** (`00:04.14`) Queixa de categoria 'respiratorio' relatada pelo paciente: sem ar (confiança 0.90)
- **critico** (`00:08.14`) Queixa de categoria 'dor_toracica' relatada pelo paciente: aperto no peito (confiança 0.90)
- **critico** (`00:12.14`) Queixa de categoria 'infeccao' relatada pelo paciente: febre (confiança 0.80)
- **atencao** (`00:00.14`) Queixa de categoria 'fadiga' relatada pelo paciente: muito cansada, nao aguento (confiança 0.60)
- **atencao** (`00:16.14`) Queixa de categoria 'dor' relatada pelo paciente: dor aumentou (confiança 0.60)
- **informativo** Sentimento predominante negativo na fala do paciente (confiança 0.67)

> Relatório gerado automaticamente por um protótipo acadêmico. Não substitui avaliação clínica.
