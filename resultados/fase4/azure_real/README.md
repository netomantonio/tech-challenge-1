# Validação integrada com Azure

Em 3 de novembro de 2026, processamos os dois cenários com Azure Speech to
Text e Azure AI Language. Utilizamos `consulta_fala.wav`, gerado por TTS,
e os backends anotados para o vídeo sintético. As chamadas aos serviços
foram reais; os dados não correspondem a consultas de pacientes reais.

| Cenário | WER contra a referência | Alertas finais |
| --- | ---: | ---: |
| PAC-0006 | 0% | 6 |
| PAC-0002 | 5,7% | 0 |

A WER descreve estes áudios de teste e não estima desempenho em consultas
naturais. No PAC-0002, o reconhecimento alterou a frase sobre ausência de
enjoo. O texto resultante produziu um achado informativo, sem alerta.

Após a revisão, “sem dor” e expressões positivas não são promovidas a queixas.
As entidades brutas continuam disponíveis para inspeção. A origem `tts_*`
impede a geração de achados clínicos baseados nas características vocais,
mantendo as medidas acústicas como descrição do sinal.

Para reproduzir, configure as credenciais no `.env` e execute:

```powershell
python -m fase4.cli_demo --todos --provedor-azure azure --audio consulta_fala.wav --dir-saida resultados/fase4/azure_real
```

Os relatórios usados pelo front também foram atualizados pela API, nas pastas
principais `resultados/fase4/PAC-0006/` e `resultados/fase4/PAC-0002/`.
As execuções podem variar conforme o reconhecimento do serviço.
