# Relatório de análise de vídeo - PAC-0006

Arquivo: `fisioterapia.mp4`  
Duração: 20.0 s (600 quadros a 30 fps; pose detectada em 0)  
Estimador de pose: `mediapipe`  
Detector de objetos: `yolov8`  
Área crítica: `manual` (148, 78, 306, 358); mão do paciente por `nenhuma`

## Amplitudes observadas

| Medida | Máximo | Limite |
| --- | ---: | ---: |
| Abdução do ombro esquerdo | 0.0° | 120° |
| Abdução do ombro direito | 0.0° | 120° |
| Inclinação lateral do tronco | 0.0° | 15° |
| Assimetria entre ombros | 0.0° | 20° |

## Desvios e eventos detectados

| Instante | Severidade | Evento | Evidência |
| --- | --- | --- | --- |
| `00:00.00` | atencao | Paciente executou o exercicio sem profissional em cena por 20.0 s | duracao_s=20.0; limite_s=2.0; janela_s=0.00-19.97; criterio=menos de duas pessoas em cena |

> Relatório gerado automaticamente a partir de vídeo sintético. Não substitui a avaliação do fisioterapeuta responsável.
