# Fase 4 - Monitoramento multimodal de pacientes

Pipeline acadêmico que monitora um paciente internado a partir de três tipos
de dado — vídeo de sessão de fisioterapia, áudio de consulta e séries
temporais de sinais vitais, prescrições e movimentação — funde os achados e
emite alertas para a equipe médica.

Os cenários principais são sintéticos. Há avaliações complementares com
PhysioNet, fala TTS e vídeos UCF101. Nenhum
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

Três execuções complementares avaliam transcrição de fala sintetizada,
sinais reais e vídeos de ações humanas. Elas exigem os pacotes de
`requirements-fase4.txt`.

```powershell
# 1. Fala inteligível em pt-BR (voz do Windows) e transcrição local (faster-whisper)
python -m fase4.data.gerar_fala_tts
python -m fase4.cli_demo --paciente-id PAC-0006 --audio consulta_fala.wav --provedor-azure local --dir-saida resultados/fase4/fala_real

# 2. Detectores de sinais vitais num registro real do PhysioNet (MIMIC-III, acesso aberto)
python -m fase4.data.baixar_physionet
python -m fase4.validacao_physionet

# 3. Estimador de pose e detector reais em clipes UCF101
python -m fase4.data.baixar_ucf101
python -m fase4.validacao_ucf101
```

Os resultados estão em `resultados/fase4/fala_real/`, `physionet/` e
`ucf101/`. O vídeo sintético do PAC-0006 deve usar os backends anotados;
o esqueleto desenhado não é uma pessoa para os modelos de percepção.

## Interface web

Tudo acima também pode ser feito pelo navegador, sem build nem dependência
nova (FastAPI + HTML/JS puro, mesmo padrão da Fase 3):

```powershell
python -m fase4.web_app            # abre http://127.0.0.1:8020
```

| Aba | O que faz |
| --- | --- |
| **Painel** | escolhe paciente (sintético, enviado ou PhysioNet), provedor, backends, áudio e área crítica; processa; mostra risco, quadros, alertas (com canal de entrega), achados filtráveis, relatórios renderizados, áudio, transcrição com termos destacados e o vídeo anotado (quadros servidos pelo servidor, porque o MP4 do OpenCV é FMP4 e o navegador não o reproduz; com `ffmpeg` no PATH a versão H.264 é gerada sob demanda). Backends reais e whisper rodam em **job** com progresso e log. |
| **Monitor ao vivo** | a série de sinais vitais avança minuto a minuto por Server-Sent Events; gráfico de 6 canais com bandas normal/crítica, marcadores de achados, linha do Isolation Forest e os alertas surgindo na linha do tempo. |
| **Equipe** | painel da equipe médica: recebe ao vivo os alertas do Painel, do Monitor e de qualquer processo que aponte `FASE4_WEBHOOK_URL=http://127.0.0.1:8020/api/webhook/alertas` (a própria CLI com `--tempo-real`, por exemplo). |
| **Dados** | cria pacientes enviados e recebe os arquivos (`consulta.wav`, `fisioterapia.mp4`, `sinais_vitais.csv`…) com validação de conteúdo; desenha a área crítica sobre um quadro do vídeo e envia a ROI ao Painel; baixa e processa um registro do PhysioNet em segundo plano. |

A API está em `fase4/web_app.py` (rotas) e `fase4/web_service.py` (regras);
a arquitetura e o contrato das respostas estão em
[docs/arquitetura_fase4_web.md](../docs/arquitetura_fase4_web.md). A
interface só exibe o que a API devolve: nenhuma regra clínica vive no JS.
Operações destrutivas (upload, exclusão, webhook) só são aceitas de
`127.0.0.1`; não há autenticação, então não exponha a porta na rede.

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
- o detector genérico informa contagem, sem atribuir papéis. Zero pessoas
  indica detecção insuficiente; uma pessoa indica acompanhamento não
  confirmado. Esses registros não geram alerta nem aumentam o risco.
  O modo anotado mantém a regra de ausência profissional.
- medidas ausentes aparecem como “Não avaliado” no relatório e N/A no vídeo,
  com `null` na API. Os avisos de cobertura também aparecem no painel.

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

Para usar a Azure de verdade, copie `.env.example` para `.env` na raiz do
projeto (fora do git) e preencha com a chave e a região do recurso de Speech
e com a chave e o endpoint do recurso de Language:

```ini
AZURE_SPEECH_KEY=...
AZURE_SPEECH_REGION=brazilsouth
AZURE_LANGUAGE_KEY=...
AZURE_LANGUAGE_ENDPOINT=https://<recurso>.cognitiveservices.azure.com/
```

`fase4/config.py` lê o `.env` sozinho; variáveis já definidas no ambiente
(`$env:AZURE_SPEECH_KEY="..."` no PowerShell) têm prioridade. Depois:

```powershell
pip install -r requirements-fase4.txt
python -m fase4.verificar_azure        # uma chamada mínima em cada serviço
python -m fase4.cli_demo --paciente-id PAC-0006 --audio consulta_fala.wav --provedor-azure azure
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
| `verificar_azure.py` | confere as credenciais com uma chamada mínima em cada serviço |
| `anomaly_detection.py` | detectores de série, prescrição e movimentação |
| `multimodal_fusion.py` | quadros clínicos e risco do paciente |
| `alertas.py` | achados, alertas, prioridade, roteamento, consolidação e canais de entrega |
| `pipeline.py` | orquestração por paciente e gravação dos relatórios |
| `cli_demo.py` | demonstração em linha de comando |
| `receptor_webhook_demo.py` | receptor HTTP local que imprime os alertas recebidos |
| `web_app.py` / `web_service.py` / `web_jobs.py` | interface web local: API FastAPI, serviços, jobs em segundo plano |
| `markdown_simples.py` | conversor dos relatórios Markdown para HTML |
| `web/` | página em HTML/JS/CSS puro (painel, monitor ao vivo, equipe, dados) |
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

São 200 testes em onze arquivos, sem rede, sem chave da Azure e sem os
pacotes opcionais: `test_fase4` (pipeline e regras), `test_fase4_azure`
(SDKs Speech e Text Analytics simulados), `test_fase4_backends` (MediaPipe e
YOLOv8 falsos lendo o vídeo real), `test_fase4_audio_local` (disartria, WER,
whisper falso, TTS), `test_fase4_tempo_real` (Isolation Forest online, NaN),
`test_fase4_entrega` (fila e webhook com servidor local),
`test_fase4_physionet` (conversão do registro) e `test_fase4_web` (API da
interface: processamento, SSE, jobs, uploads, ROI, webhook). As amostras são
geradas em diretório temporário.

### Avaliação em vídeos reais — UCF101

Selecionamos o UCF101 a partir do catálogo Awesome-Video-Datasets e do
relatório CRCV-TR-12-01, de Soomro, Zamir e Shah (2012). A base contém
13.320 vídeos distribuídos em 101 classes de ações humanas. Nosso recorte
exploratório reúne um clipe de polichinelo, um de agachamento e um de avanço.
Não utilizamos esses vídeos para treinar ou ajustar os modelos.

Os arquivos são obtidos do espelho `sumith2425/UCF101`, com revisão fixa e
SHA-256 em `fase4/data/ucf101_manifesto.json`. Os endereços antigos da UCF
e o arquivo de distribuição indicado no tutorial TensorFlow estavam
indisponíveis no momento da coleta. Mantivemos a origem e a revisão do
espelho nos resultados. A conversão AVI–MP4 preserva os quadros, a resolução
e a taxa de amostragem; o áudio não é utilizado neste experimento.

| Clipe | Quadros | Quadros com pose | Cobertura |
| --- | ---: | ---: | ---: |
| `v_BodyWeightSquats_g01_c01.avi` | 267 | 267 | 100,0% |
| `v_JumpingJack_g01_c01.avi` | 93 | 93 | 100,0% |
| `v_Lunges_g01_c01.avi` | 262 | 202 | 77,1% |

Empregamos MediaPipe 1.2.0 e YOLOv8n, com Ultralytics 8.3.0. A cobertura
expressa a proporção de quadros com pose disponível, não a precisão dos
pontos articulares. O recorte não oferece rótulos de supervisão profissional
ou de anomalias clínicas. Por isso, o contexto `exploratorio` mantém as
medidas descritivas e não aplica os limites de pós-operatório nem produz
alertas clínicos a partir desses exercícios.

```powershell
python -m fase4.data.baixar_ucf101
python -m fase4.validacao_ucf101
python -m fase4.web_app
```

Na interface, os clipes aparecem no grupo **UCF101 · ações humanas**. Ao
selecioná-los, MediaPipe e YOLOv8 são sugeridos; os seletores continuam
disponíveis. Os pacientes sintéticos continuam usando as anotações por
padrão. O modo anotado requer arquivos de anotações correspondentes, que
não acompanham o recorte UCF101. As opções de áudio `offline`, `local`,
`auto` e `azure` permanecem disponíveis para casos com áudio.

Os relatórios individuais e a síntese estão em `resultados/fase4/ucf101/`.
Essa execução demonstra percepção em pessoas filmadas, mas não substitui
uma avaliação em sessões reais de fisioterapia.

### Revisão da análise de linguagem

A validação atual utiliza Speech e Language reais nos dois pacientes: WER
de 0% no PAC-0006 e 5,7% no PAC-0002. O cruzamento com o
léxico foi corrigido: somente sintomas/diagnósticos presentes no texto podem
enriquecer os achados, e expressões não pronunciadas não são inferidas a
partir de fragmentos. O registro anterior preserva a evidência de uso da
nuvem. As saídas corrigidas estão em `resultados/fase4/azure_real/`.

O relatório original está em `resultados/fase4/azure_registro/`, identificado
como histórico. Os relatórios principais e os resultados do front foram
regenerados pela API com Azure e vídeo anotado. Nesta revisão, a suíte completa teve 273 aprovações e um
teste ignorado pela ausência de `wfdb`; o teste dos seletores do frontend
também passou (`node tests/test_fase4_web_ui.mjs`).
### Validação de queixas e áudios de demonstração

Na integração com o Azure Language, verificamos a presença textual dos termos,
as negações próximas e a indicação de certeza retornada para entidades clínicas.
Expressões de melhora presentes no léxico positivo não são acrescentadas como
queixas. Esses filtros são regras de escopo limitado, não uma interpretação
clínica completa. Erros de transcrição ainda podem alterar o sentido de uma frase.

Áudios cuja referência declara origem `tts_*` são utilizados para validar a
transcrição e a análise textual. Seus atributos acústicos permanecem no relatório,
mas não geram achados de alteração vocal: uma voz sintetizada não fornece evidência
de disartria, fadiga ou dificuldade respiratória de um paciente. Essa restrição
depende dos metadados de origem, não de um detector automático de voz sintética.
## Pendências de entrega e alcance das evidências

A implementação inclui análise das modalidades, fusão e entrega de alertas.
Para fortalecer a demonstração dos requisitos clínicos, falta avaliar uma
sessão clínica filmada com desvios de referência e fala natural com alterações
vocais anotadas. Os três clipes UCF101 demonstram execução dos modelos em
pessoas filmadas, sem comprovar detecção de falhas clínicas. A reprodução
incremental dos sinais não corresponde à aquisição contínua de um monitor.

A entrega ainda exige consolidar e publicar as alterações no repositório
remoto e gravar o vídeo demonstrativo de até 15 minutos no YouTube ou Vimeo.
O roteiro está em `docs/script_video_demonstracao_fase4.txt`; a publicação
não foi comprovada nesta revisão. Treinar um modelo próprio não é uma
exigência explícita do enunciado.
