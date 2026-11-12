# Relatório de análise de áudio - PAC-0002

Arquivo: `consulta_fala.wav`  
Duração: 17.1 s  
Transcrição: provedor `azure`, confiança média 0.84, taxa de erro de palavras contra a referência 5.7%  
Análise de texto: provedor `azure`

## Atributos acústicos

| Atributo | Valor |
| --- | ---: |
| Tempo efetivo de fala | 11.0 s |
| Proporção de pausa | 0.36 |
| Frases detectadas | 19 |
| Duração média da frase | 0.58 s |
| Pausa máxima | 1.39 s |
| F0 média | 183.4 Hz |
| Jitter da F0 | 0.0270 |
| Taxa de fala | 3.87 sílabas/s |
| Queda de energia | 0.06 |

## Transcrição

- `00:00.51` Bom dia, doutora. Eu estou bem melhor esta semana. Consegui fazer a caminhada todos os dias sem dor. Tomei o remédio nos horários inau, tive enjoo. Estou mais animada e dormindo tranquila à noite.

## Sentimento e termos críticos

Sentimento: **misto**
Frases-chave: horários inau, Bom dia, remédio, doutora, caminhada, dor, enjoo, noite

| Termo | Categoria | Peso |
| --- | --- | ---: |
| enjoo | sintoma_relatado_azure | 0.45 |

Entidades clínicas reconhecidas pela Azure (1 documento(s) enviado(s)):

| Entidade | Categoria | Confiança |
| --- | --- | ---: |
| doutora | HealthcareProfession | 0.83 |
| estou bem melhor | SymptomOrSign | 0.90 |
| esta semana | Time | 0.94 |
| todos os dias | Frequency | 0.79 |
| dor | SymptomOrSign | 0.99 |
| remédio | MedicationClass | 0.68 |
| enjoo | SymptomOrSign | 0.98 |
| animada | SymptomOrSign | 0.96 |
| dormindo tranquila | SymptomOrSign | 0.96 |
| à noite | Time | 0.76 |

## Achados

- **informativo** (`00:00.51`) Queixa de categoria 'sintoma_relatado_azure' relatada pelo paciente: enjoo (confiança 0.45)

> Relatório gerado automaticamente por um protótipo acadêmico. Não substitui avaliação clínica.


> Audio de fala sintetizada (TTS): atributos acusticos apenas descritivos; a avaliacao de alteracoes vocais clinicas nao foi aplicada.