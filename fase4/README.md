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

Cada alerta é entregue numa fila `resultados/fase4/fila_alertas.jsonl` e,
se `FASE4_WEBHOOK_URL` estiver definida, enviado por POST. Para ver o fluxo
final do alerta chegando num "painel da equipe", em outro terminal:

```powershell
python -m fase4.receptor_webhook_demo --porta 8765
# no terminal do pipeline:
$env:FASE4_WEBHOOK_URL = "http://127.0.0.1:8765/alertas"
python -m fase4.cli_demo --paciente-id PAC-0006 --tempo-real --atraso 0.01
```

## Demonstrações sem credencial Azure

Três execuções extras mostram os mesmos módulos em dados que não foram
gerados pelo projeto. Elas exigem os pacotes de `requirements-fase4.txt`.

```powershell
# 1. Fala inteligível em pt-BR (voz do Windows) e transcrição local (faster-whisper)
python -m fase4.data.gerar_fala_tts
python -m fase4.cli_demo --paciente-id PAC-0006 --audio consulta_fala.wav --provedor-azure local --dir-saida resultados/fase4/fala_real

# 2. Detectores de sinais vitais num registro real do PhysioNet (MIMIC-III, acesso aberto)
python -m fase4.data.baixar_physionet
python -m fase4.validacao_physionet

# 3. Estimador de pose e detector reais (MediaPipe + YOLOv8) com área crítica manual
python -m fase4.cli_demo --paciente-id PAC-0006 --backend-pose mediapipe --backend-detector yolov8 --area-critica 250,178,296,220
```

Os resultados de 1 e 2 estão versionados em `resultados/fase4/fala_real/` e
`resultados/fase4/physionet/`. A execução 3 sobre o vídeo sintético prova que
o caminho real roda de ponta a ponta, mas não detecta pose: o esqueleto
desenhado não é uma pessoa.

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
idêntica: ela consome apenas coordenadas de articulações. O backend aceita
tanto a API legada (`mp.solutions.pose`, série 0.10) quanto a API Tasks
(`PoseLandmarker`, 0.10.30+ e 1.x); o modelo `pose_landmarker_lite.task` é
baixado uma vez para `fase4/data/modelos/`.

O YOLOv8 genérico (classes COCO) não conhece `dreno`, `mao_paciente` nem
`profissional`. Para as regras de evento funcionarem em vídeo real:

- `--area-critica x1,y1,x2,y2` define a região do dreno manualmente (em
  pixels); a mão do paciente passa a ser o keypoint do punho, com raio de
  11 px (`raio_mao_px`);
- "profissional presente" vira "duas ou mais pessoas em cena" (`person`).

O relatório de vídeo registra de onde veio a área crítica (`anotada`,
`manual` ou `nenhuma`) e como a mão foi localizada (`caixa_detector`,
`keypoint_punho` ou `nenhuma`).

### Áudio (`audio_analysis.py`)

Extrai da forma de onda: proporção de pausa, duração média de frase, pausa
máxima, F0 média, variação da F0 entre quadros, taxa de fala, energia e queda
de energia ao longo da consulta. Quatro regras usam esses atributos: fala
entrecortada (esforço respiratório), fadiga vocal, instabilidade vocal e
padrão disartrico (fala lenta somada a F0 instável ou voz monótona — os
marcadores acústicos clássicos de disartria, usados como triagem, não como
diagnóstico).

A transcrição e a análise de linguagem usam a Azure (`azure_services.py`):

| Serviço | Provedor `azure` | Provedor `offline` | Provedor `local` |
| --- | --- | --- | --- |
| Speech to Text | reconhecimento contínuo do WAV | transcrição de referência gravada com o áudio | faster-whisper na própria máquina |
| Text Analytics | sentimento, frases-chave e entidades clínicas (Text Analytics for health) | sentimento por léxico | igual ao `offline` |

O que decide alerta de conteúdo é o léxico clínico de
`data/termos_criticos_audio.json`, em todos os provedores: as frases-chave da
Azure dizem o que é saliente no texto, não o que é clinicamente crítico. Com
a Azure, as entidades clínicas reconhecidas são cruzadas com o léxico (uma
entidade que casa herda a categoria e o peso) e sintomas fora do léxico com
confiança alta entram como termos informativos. Textos longos são divididos
em documentos de até 5.000 caracteres, o limite do serviço.

O provedor `local` existe para demonstrar Speech to Text em fala real sem
credencial; `auto` nunca o escolhe. Quando a transcrição vem de um
reconhecedor real (`azure` ou `local`) e há transcrição de referência ao lado
do WAV, o relatório mostra a taxa de erro de palavras (WER).

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
Azure, use uma gravação de consulta autorizada em WAV PCM mono de 16 bits —
ou o `consulta_fala.wav` gerado por `python -m fase4.data.gerar_fala_tts`,
que lê a transcrição de referência com a voz pt-BR do Windows (escolha com
`--audio consulta_fala.wav`). A voz sintética é estável e bem articulada:
serve à camada de linguagem, não às regras acústicas de fadiga.

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

No modo `--tempo-real` o Isolation Forest também roda leitura a leitura
(`DetectorMultivariadoOnline`): treina nas duas primeiras horas e emite o
achado no minuto em que o episódio completa 15 minutos. Canal ausente e
leitura faltante (NaN) não derrubam nenhum detector.

### Validação com dados reais (PhysioNet)

`python -m fase4.data.baixar_physionet` traz um registro numérico de acesso
aberto do MIMIC-III Waveform Database Matched Subset (1 Hz; FC, pressão
arterial, SpO2 e frequência respiratória; sem temperatura), reamostra para
mediana por minuto e grava no formato de `sinais_vitais.csv`, com
`fonte.json` de procedência. `python -m fase4.validacao_physionet` aplica os
mesmos detectores e grava `resultados/fase4/physionet/<registro>/`. Não há
rótulo de desfecho no registro: a validação mede comportamento e taxa de
achados, não sensibilidade. Licença ODbL; citação em `fonte.json`.

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

No `PAC-0006` isso leva 37 achados notificáveis a 6 alertas, cada um roteado
para as equipes das modalidades envolvidas.

Cada alerta é entregue por canais (`CanalEntrega`): a fila
`fila_alertas.jsonl` na pasta da execução (sempre) e um webhook HTTP quando
`FASE4_WEBHOOK_URL` está definida. Falha de canal vira evento
`entrega_falhou` na auditoria e nunca interrompe o monitoramento; o
relatório e o JSON do alerta registram por onde ele passou
(`canais_entrega`).

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
| `alertas.py` | achados, alertas, prioridade, roteamento, consolidação e canais de entrega |
| `pipeline.py` | orquestração por paciente e gravação dos relatórios |
| `cli_demo.py` | demonstração em linha de comando |
| `receptor_webhook_demo.py` | receptor HTTP local que imprime os alertas recebidos |
| `validacao_physionet.py` | detectores de sinais vitais sobre um registro real do PhysioNet |
| `data/gerar_fala_tts.py` | consulta com fala inteligível pt-BR (TTS do Windows) |
| `data/baixar_physionet.py` | download e conversão do registro do PhysioNet |
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

E em `resultados/fase4/`: `resumo_execucao.json`, `fila_alertas.jsonl` e
`auditoria_monitoramento.jsonl`; `fala_real/` (transcrição local da fala
sintetizada) e `physionet/` (validação em dados reais).

## Testes

```powershell
python -m unittest discover -s tests -p "test_fase4*.py" -v
```

São 163 testes em sete arquivos, sem rede, sem chave da Azure e sem os
pacotes opcionais: `test_fase4` (pipeline e regras), `test_fase4_azure`
(SDKs Speech e Text Analytics simulados), `test_fase4_backends` (MediaPipe e
YOLOv8 falsos lendo o vídeo real), `test_fase4_audio_local` (disartria, WER,
whisper falso, TTS), `test_fase4_tempo_real` (Isolation Forest online, NaN),
`test_fase4_entrega` (fila e webhook com servidor local) e
`test_fase4_physionet` (conversão do registro). As amostras são geradas em
diretório temporário.
