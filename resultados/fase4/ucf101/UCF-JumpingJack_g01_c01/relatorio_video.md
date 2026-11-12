# Relatório de análise de vídeo - UCF-JumpingJack_g01_c01

Arquivo: `fisioterapia.mp4`  
Duração: 3.1 s (93 quadros a 30 fps; pose detectada em 93)  
Estimador de pose: `mediapipe`  
Detector de objetos: `yolov8`  
Área crítica: `nenhuma`; mão do paciente por `nenhuma`

> Avaliacao exploratoria de acoes humanas: medidas descritivas, sem aplicar limites de pos-operatorio ou inferir anomalias clinicas.
> YOLOv8 identifica pessoas, mas nao distingue paciente, acompanhante ou profissional. Supervisao nao avaliavel por este detector.

## Amplitudes observadas

| Medida | Máximo | Limite |
| --- | ---: | ---: |
| Abdução do ombro esquerdo | 167.5° | Não aplicado |
| Abdução do ombro direito | 159.2° | Não aplicado |
| Inclinação lateral do tronco | 4.3° | Não aplicado |
| Assimetria entre ombros | 19.1° | Não aplicado |

## Desvios e eventos detectados

| Instante | Severidade | Evento | Evidência |
| --- | --- | --- | --- |
| `00:00.00` | informativo | Uma pessoa detectada; acompanhamento profissional nao confirmado | duracao_s=3.1; pessoas_detectadas=1; criterio=contagem de pessoas sem identificacao de papeis |

> Relatório gerado por um protótipo acadêmico. Não substitui a avaliação do fisioterapeuta responsável.
