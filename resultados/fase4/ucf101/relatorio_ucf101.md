# Avaliação exploratória — UCF101

Utilizamos três clipes para verificar a extração de postura em pessoas filmadas. Os modelos foram usados com pesos pré-treinados, sem ajuste ao recorte. Os rótulos de ação não indicam anomalias clínicas; por isso, não aplicamos os limites de pós-operatório.

| Clipe | Quadros | Com pose | Cobertura |
| --- | ---: | ---: | ---: |
| v_BodyWeightSquats_g01_c01.avi | 267 | 267 | 100.0% |
| v_JumpingJack_g01_c01.avi | 93 | 93 | 100.0% |
| v_Lunges_g01_c01.avi | 262 | 202 | 77.1% |

A cobertura mede disponibilidade da pose, não acurácia dos pontos. Não avaliamos classificação das 101 ações, qualidade clínica do exercício ou identidade profissional. A revisão do espelho, os hashes e as versões estão em `avaliacao.json`.
