# Fase 4 - Monitoramento multimodal de pacientes

Pipeline acadêmico que monitora um paciente internado a partir de três tipos
de dado — vídeo de sessão de fisioterapia, áudio de consulta e séries
temporais de sinais vitais, prescrições e movimentação — funde os achados e
emite alertas para a equipe médica.

Todos os dados são sintéticos e gerados pelo próprio repositório. Nenhum
resultado aqui é validado para uso assistencial.

## Início rápido

Na raiz do repositório, com o ambiente da Fase 1/2 já instalado
(`pip install -r requirements.txt`), instale também o OpenCV para gerar e
anotar o vídeo:

```powershell
pip install "opencv-contrib-python>=4.10,<4.12"
```

```powershell
python -m fase4.data.gerar_dados_sinteticos
python -m fase4.cli_demo --paciente-id PAC-0006
```

A primeira linha cria as amostras em `fase4/data/amostras/`; a segunda roda o
pipeline completo e grava os relatórios em `resultados/fase4/PAC-0006/`.

Para processar os dois pacientes e gravar o resumo consolidado:

```powershell
python -m fase4.cli_demo --todos
```

Para ver os alertas de sinais vitais surgindo minuto a minuto, como no leito:

```powershell
python -m fase4.cli_demo --paciente-id PAC-0006 --tempo-real
```

## Os dois casos

| Paciente | Situação | Modalidades |
| --- | --- | --- |
| `PAC-0006` | Pós-operatório de mastectomia com deterioração progressiva no turno | vídeo, áudio, sinais vitais, prescrições, movimentação |
| `PAC-0002` | Caso estável, usado como controle | áudio, sinais vitais, prescrições, movimentação |

O `PAC-0002` existe para verificar especificidade: um pipeline que alerta o
paciente estável não serve. Ele também mostra que a ausência de uma
modalidade (não tem vídeo) não interrompe a execução.

## O que cada modalidade faz

### Vídeo (`video_analysis.py`)

Calcula, quadro a quadro, a abdução de cada ombro, a inclinação lateral do
tronco e a assimetria entre os ombros, e aplica cinco regras:

| Regra | Limite | Severidade |
| --- | ---: | --- |
| Compensação de tronco | 15° | atenção |
| Amplitude de abdução do ombro | 120° | crítico |
| Assimetria entre ombros | 20° | atenção |
| Mão do paciente na área crítica (dreno) | qualquer sobreposição | crítico |
| Profissional ausente de cena | 2 s | atenção |

Um desvio só vira achado depois de persistir por alguns quadros seguidos —
é o que separa desvio real de ruído do estimador de pose.

A percepção é separada da análise por backends:

| Backend | Opção | O que faz |
| --- | --- | --- |
| Pose anotada | `--backend-pose anotado` (padrão) | lê os keypoints do JSON gerado com a amostra |
| MediaPipe Pose | `--backend-pose mediapipe` | extrai os keypoints de um vídeo real |
| Objetos anotados | `--backend-detector anotado` (padrão) | lê as caixas do JSON gerado com a amostra |
| YOLOv8 | `--backend-detector yolov8` | roda o Ultralytics no vídeo real |

O enunciado sugere OpenPose para postura. Usamos MediaPipe Pose porque
entrega o mesmo tipo de saída (keypoints 2D do corpo) e instala por `pip`,
enquanto o OpenPose exige compilar Caffe com CUDA. A análise a jusante é
idêntica: ela consome apenas coordenadas de articulações.

### Áudio (`audio_analysis.py`)

Extrai da forma de onda: proporção de pausa, duração média de frase, pausa
máxima, F0 média, variação da F0 entre quadros, taxa de fala, energia e queda
de energia ao longo da consulta. Três regras usam esses atributos
(fala entrecortada, fadiga vocal, instabilidade vocal).

A transcrição e a análise de linguagem usam a Azure (`azure_services.py`):

| Serviço | Provedor `azure` | Provedor `offline` |
| --- | --- | --- |
| Speech to Text | reconhecimento contínuo do WAV | transcrição de referência gravada com o áudio |
| Text Analytics | sentimento e frases-chave | sentimento por léxico |

O que decide alerta de conteúdo é o léxico clínico de
`data/termos_criticos_audio.json`, nos dois provedores: as frases-chave da
Azure dizem o que é saliente no texto, não o que é clinicamente crítico.

Para usar a Azure de verdade:

```powershell
$env:AZURE_SPEECH_KEY="..."
$env:AZURE_SPEECH_REGION="brazilsouth"
$env:AZURE_LANGUAGE_KEY="..."
$env:AZURE_LANGUAGE_ENDPOINT="https://<recurso>.cognitiveservices.azure.com/"
pip install -r requirements-fase4.txt
python -m fase4.cli_demo --paciente-id PAC-0006 --provedor-azure azure
```

Sem credencial, `--provedor-azure azure` falha com mensagem explícita, em vez
de cair silenciosamente para o modo local.

O WAV sintético contém um sinal acústico de teste, sem palavras inteligíveis.
A transcrição de referência só é usada no modo offline. Para demonstrar a
Azure, use uma gravação de consulta autorizada em WAV PCM mono de 16 bits.

### Processar gravações próprias

Organize os arquivos em `dados/PAC-REAL/`: `consulta.wav`,
`fisioterapia.mp4` e, quando disponíveis, `sinais_vitais.csv`,
`prescricoes.json` e `movimentacao.csv` no formato das amostras.
Com as credenciais definidas no ambiente:

```powershell
python -m fase4.cli_demo --paciente-id PAC-REAL --raiz-amostras dados --dir-saida resultados/fase4_real --provedor-azure azure --backend-pose mediapipe --backend-detector yolov8
```

O YOLOv8 padrão reconhece objetos genéricos. Não reconhece drenos nem
distingue paciente e profissional; as regras dessas classes são demonstradas
com as anotações sintéticas. A análise postural usa os pontos do MediaPipe.
Uma sessão deve enquadrar apenas o paciente para evitar troca de pessoa.
Os alertas são apresentados no terminal e nos relatórios; não há envio a um
sistema hospitalar externo.

A entrega ainda exige executar os serviços Azure com áudio inteligível,
validar a análise com vídeo autorizado e gravar/publicar a demonstração de
até 15 minutos no YouTube ou Vimeo. O modo offline não substitui essas etapas.

### Anomalias (`anomaly_detection.py`)

Sinais vitais, com quatro detectores complementares:

| Detector | Pega | Não pega |
| --- | --- | --- |
| z-score robusto (mediana/MAD móveis) | mudança abrupta | piora lenta, que entra na própria janela de referência |
| Desvio de tendência (mediana recente contra o basal) | piora gradual | mudança brusca dentro da tolerância |
| Limite clínico | valor fora da faixa crítica | piora que ainda não cruzou o limite |
| Isolation Forest treinado no basal | combinação anômala entre canais | desvio de um canal só, dentro da faixa |

Os três primeiros exigem leituras consecutivas antes de confirmar e têm
histerese, para não repetir o mesmo aviso a cada minuto. Um pico isolado de
frequência cardíaca gera apenas um registro informativo, não um alerta.

Prescrições: salto de dose acima de 50%, suspensão de antimicrobiano antes de
80% do esquema previsto e duplicidade terapêutica na mesma classe.

Movimentação: imobilidade diurna prolongada e agitação noturna.

### Fusão (`multimodal_fusion.py`)

Cinco regras combinam achados de modalidades diferentes em quadros clínicos
(`deterioracao_respiratoria`, `suspeita_infeccao`, `risco_medicamentoso`,
`reabilitacao_insegura`, `risco_de_imobilidade`) e calculam um risco 0-100.
Um quadro exige pelo menos duas modalidades: sem isso não é fusão, é repetir
o que uma modalidade sozinha já disse.

### Alertas (`alertas.py`)

Três mecanismos evitam fadiga de alarme:

- **consolidação**: achados da mesma família viram um alerta só;
- **absorção**: quando a fusão identifica um quadro, as famílias que o
  sustentam não geram alerta próprio;
- **supressão temporal**: o mesmo grupo não é reenviado dentro de 15 minutos.

No `PAC-0006` isso leva 36 achados notificáveis a 6 alertas, cada um roteado
para as equipes das modalidades envolvidas.

## Estrutura

| Caminho | Responsabilidade |
| --- | --- |
| `config.py` | caminhos, variáveis de ambiente e todos os limiares |
| `data/gerar_dados_sinteticos.py` | geração reproduzível das amostras |
| `data/termos_criticos_audio.json` | léxico clínico de termos críticos |
| `video_analysis.py` | backends de pose/detector, ângulos, regras e relatório |
| `audio_analysis.py` | atributos acústicos, regras vocais e relatório |
| `azure_services.py` | Speech to Text e Text Analytics, com substituto offline |
| `anomaly_detection.py` | detectores de série, prescrição e movimentação |
| `multimodal_fusion.py` | quadros clínicos e risco do paciente |
| `alertas.py` | achados, alertas, prioridade, roteamento e consolidação |
| `pipeline.py` | orquestração por paciente e gravação dos relatórios |
| `cli_demo.py` | demonstração em linha de comando |
| `logging_utils.py` | auditoria em `.jsonl` |

## Saídas

Em `resultados/fase4/<paciente>/`:

| Arquivo | Conteúdo |
| --- | --- |
| `relatorio_monitoramento.md` | relatório consolidado: achados, quadros e alertas |
| `relatorio_video.md` | relatório automático da sessão gravada |
| `relatorio_audio.md` | atributos acústicos, transcrição e termos críticos |
| `achados.json` / `achados.csv` | todos os achados com instante e evidências |
| `alertas.json` | alertas emitidos, com destino e conduta sugerida |
| `sinais_vitais_scores.csv` | série original com o score do Isolation Forest |
| `fisioterapia_anotado.mp4` | vídeo com ângulos e desvios sobrepostos |

E em `resultados/fase4/`: `resumo_execucao.json` e
`auditoria_monitoramento.jsonl`.

## Testes

```powershell
python -m unittest tests.test_fase4 tests.test_fase4_azure -v
```

São 84 testes, sem rede e sem chave da Azure. O pipeline usa o provedor
`offline`; os testes do SDK Speech simulam sucesso, cancelamento, ausência
de fala e tempo limite. As amostras são geradas em diretório temporário.
