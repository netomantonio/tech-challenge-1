# Relatório de análise de vídeo - PAC-0006

Arquivo: `fisioterapia.mp4`  
Duração: 20.0 s (600 quadros a 30 fps)  
Estimador de pose: `anotado`  
Detector de objetos: `anotado`

## Amplitudes observadas

| Medida | Máximo | Limite |
| --- | ---: | ---: |
| Abdução do ombro esquerdo | 139.6° | 120° |
| Abdução do ombro direito | 97.2° | 120° |
| Inclinação lateral do tronco | 19.4° | 15° |
| Assimetria entre ombros | 49.2° | 20° |

## Desvios e eventos detectados

| Instante | Severidade | Evento | Evidência |
| --- | --- | --- | --- |
| `00:11.40` | atencao | Compensacao de tronco acima do tolerado durante o exercicio (19.4 graus) | medida=inclinacao_tronco; limite_graus=15.0; valor_maximo_graus=19.4; janela_s=10.97-12.33; quadros=42 |
| `00:15.07` | atencao | Compensacao de tronco acima do tolerado durante o exercicio (17.8 graus) | medida=inclinacao_tronco; limite_graus=15.0; valor_maximo_graus=17.8; janela_s=14.57-15.50; quadros=29 |
| `00:11.60` | critico | Amplitude de abducao do ombro esquerdo acima da liberada no pos-operatorio (139.6 graus) | medida=abducao_ombro_esquerdo; limite_graus=120.0; valor_maximo_graus=139.6; janela_s=11.13-12.23; quadros=34 |
| `00:14.87` | critico | Amplitude de abducao do ombro esquerdo acima da liberada no pos-operatorio (133.8 graus) | medida=abducao_ombro_esquerdo; limite_graus=120.0; valor_maximo_graus=133.8; janela_s=14.53-15.43; quadros=28 |
| `00:11.77` | atencao | Assimetria entre os ombros na mesma repeticao (49.2 graus) | medida=assimetria_ombros; limite_graus=20.0; valor_maximo_graus=49.2; janela_s=10.53-12.83; quadros=70 |
| `00:13.37` | atencao | Assimetria entre os ombros na mesma repeticao (25.9 graus) | medida=assimetria_ombros; limite_graus=20.0; valor_maximo_graus=25.9; janela_s=13.27-13.40; quadros=5 |
| `00:14.90` | atencao | Assimetria entre os ombros na mesma repeticao (40.1 graus) | medida=assimetria_ombros; limite_graus=20.0; valor_maximo_graus=40.1; janela_s=14.23-16.03; quadros=55 |
| `00:13.20` | critico | Mao do paciente dentro da area critica do dreno durante a sessao | sobreposicao_maxima=1.0; duracao_s=0.93; janela_s=13.07-13.97 |
| `00:10.00` | atencao | Paciente executou o exercicio sem profissional em cena por 4.0 s | duracao_s=4.0; limite_s=2.0; janela_s=10.00-13.97 |

> Relatório gerado automaticamente a partir de vídeo sintético. Não substitui a avaliação do fisioterapeuta responsável.
