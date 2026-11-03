# Relatório técnico - Tech Challenge Fase 4

## 1. Escopo

Nesta fase, implementamos o monitoramento contínuo de pacientes a partir de
dados multimodais. O sistema analisa vídeo de sessões de fisioterapia, áudio
de consultas médicas e séries temporais de sinais vitais, prescrições e
movimentação no leito; combina os achados das três modalidades; e emite
alertas automáticos para a equipe médica.

Mantivemos o módulo isolado em `fase4/`, como fizemos com a Fase 3, para não
alterar o funcionamento das fases anteriores. As 74 verificações automatizadas
que já existiam continuam passando, e acrescentamos 163 testes específicos
desta fase.

Os dados principais são sintéticos e gerados pelo próprio repositório. Duas
execuções complementares usam dados que não geramos: um registro real de
monitor de leito do PhysioNet (§5.10) e uma consulta com fala inteligível
sintetizada pela voz do sistema operacional, transcrita por um reconhecedor
real (§5.11). Nenhum resultado foi validado para uso assistencial: qualquer
conduta exige avaliação do profissional responsável.

## 2. Dados

### 2.1 Por que dados sintéticos

O enunciado sugere o PhysioNet e o AudioSet. Neste protótipo usamos dados
sintéticos para reproduzir os mesmos casos durante os testes, sem depender
de arquivos externos. Não avaliamos o desempenho nas bases sugeridas.

A alternativa foi gerar amostras sintéticas no formato do projeto,
em `fase4/data/gerar_dados_sinteticos.py`. Isso trouxe uma vantagem
metodológica que compensou a perda de realismo: sabemos exatamente o que foi
injetado em cada amostra, e por isso podemos medir o que o pipeline detectou e
o que deixou passar. O arquivo `eventos_plantados.json` gravado junto de cada
paciente funciona como gabarito.

A contrapartida é clara e precisa ser dita: os limiares deste trabalho foram
calibrados contra dados que nós mesmos geramos. Eles não têm validade clínica
e serviriam apenas de ponto de partida num estudo com dados reais.

### 2.2 Os dois pacientes

Reaproveitamos os pacientes sintéticos da Fase 3 para manter continuidade:

| Paciente | Situação | Modalidades disponíveis |
| --- | --- | --- |
| `PAC-0006` | Pós-operatório de mastectomia (dia 2), com deterioração progressiva ao longo do turno | vídeo, áudio, sinais vitais, prescrições, movimentação |
| `PAC-0002` | Caso clinicamente estável | áudio, sinais vitais, prescrições, movimentação |

O `PAC-0002` foi incluído de propósito como controle. Um sistema de
monitoramento que acerta a deterioração mas também alerta o paciente estável
não tem utilidade: ele treina a equipe a ignorar o painel. O `PAC-0002`
também não tem vídeo, o que exercita o comportamento do pipeline quando falta
uma modalidade.

### 2.3 O que cada amostra contém

| Arquivo | Conteúdo | Eventos injetados |
| --- | --- | --- |
| `sinais_vitais.csv` | 720 leituras (12 h, 1 por minuto) de 6 canais | piora progressiva a partir das 16:00; pico isolado de 185 bpm às 10:30 (artefato de sensor) |
| `consulta.wav` + `.transcricao.json` | 16 s de fala sintetizada, mono 16 kHz | voz cansada com pausas longas e queixas críticas |
| `fisioterapia.mp4` + `.keypoints.json` + `.objetos.json` | 600 quadros (20 s a 30 fps) de uma sessão de abdução de ombro | 2 repetições com compensação de tronco e amplitude excessiva; mão na área do dreno; 4 s sem o profissional em cena |
| `prescricoes.json` | 7 eventos de prescrição do turno | salto de dose de morfina; suspensão precoce de antimicrobiano; duplicidade de anti-inflamatório |
| `movimentacao.csv` | índice horário de atividade (24 h) | imobilidade diurna de 6 h; agitação das 02:00 às 04:00 |

A geração é determinística: a mesma `seed` produz exatamente as mesmas
amostras, o que é verificado por teste automatizado.

O pico de 185 bpm foi colocado na série de propósito. Ele é o teste adversarial
do detector: cruza o limite crítico de frequência cardíaca, mas é uma leitura
única e não deve gerar alerta.

### 2.4 O vídeo sintético

O vídeo é um esqueleto articulado desenhado com OpenCV, não uma gravação de
pessoa. Junto dele gravamos dois arquivos JSON no mesmo formato que um
estimador de pose e um detector de objetos produziriam: posição e confiança de
13 articulações por quadro, e caixas de `paciente`, `profissional`, `maca`,
`dreno` e `mao_paciente`.

Essa separação foi a decisão de projeto mais importante da modalidade vídeo.
A análise — ângulos articulares, regras de desvio, relatório — consome apenas
coordenadas, então é a mesma independentemente de as coordenadas virem das
anotações sintéticas ou de um modelo rodando em vídeo real. Trocar a fonte é
trocar um argumento de linha de comando.

## 3. Fluxo multimodal

```text
                      ┌──────────────────────────────────────┐
  fisioterapia.mp4 ──▶│ video_analysis                       │
                      │  pose (anotado | MediaPipe Pose)     │──┐
                      │  objetos (anotado | YOLOv8)          │  │
                      │  ângulos → 5 regras de desvio        │  │
                      └──────────────────────────────────────┘  │
                      ┌──────────────────────────────────────┐  │
     consulta.wav ───▶│ audio_analysis                       │  │
                      │  atributos acústicos (numpy)         │  │
                      │  Azure Speech to Text  ──┐           │──┤
                      │  Azure Text Analytics  ──┴▶ léxico   │  │
                      │  3 regras vocais + termos críticos   │  │
                      └──────────────────────────────────────┘  │
                      ┌──────────────────────────────────────┐  │  achados
 sinais_vitais.csv ──▶│ anomaly_detection                    │  │
   prescricoes.json ─▶│  z-score robusto / tendência /       │──┤
   movimentacao.csv ─▶│  limite clínico / Isolation Forest   │  │
                      │  3 regras de prescrição + 2 de       │  │
                      │  movimentação                        │  │
                      └──────────────────────────────────────┘  │
                                                                ▼
                      ┌──────────────────────────────────────────────┐
                      │ multimodal_fusion                            │
                      │  5 regras de quadro clínico (≥2 modalidades) │
                      │  risco do paciente 0-100                     │
                      └──────────────────────────────────────────────┘
                                                                ▼
                      ┌──────────────────────────────────────────────┐
                      │ alertas                                      │
                      │  consolidação por família                    │
                      │  absorção pelos quadros identificados        │
                      │  supressão temporal (15 min)                 │
                      │  prioridade + roteamento por equipe          │
                      └──────────────────────────────────────────────┘
                                                                ▼
                            relatórios, alertas e auditoria em
                                  resultados/fase4/
```

A estrutura comum que liga tudo é o `Achado`: modalidade, tipo, descrição,
severidade (`informativo`, `atencao`, `critico`), score, instante e evidências.
Toda modalidade produz `Achado`; a fusão e o motor de alertas consomem
`Achado`. Foi essa uniformização que permitiu escrever a fusão e o motor de
alertas uma única vez, em vez de três integrações aos pares.

## 4. Modelos e técnicas por tipo de dado

### 4.1 Vídeo

Dois estágios de percepção, cada um com backend trocável:

| Estágio | Modelo real | Substituto local |
| --- | --- | --- |
| Postura | MediaPipe Pose (`--backend-pose mediapipe`) | keypoints anotados |
| Objetos e áreas críticas | YOLOv8 / Ultralytics (`--backend-detector yolov8`) | caixas anotadas |

O enunciado sugere OpenPose para a análise postural. Optamos pelo MediaPipe
Pose: ele entrega a mesma saída que a análise consome (keypoints 2D do corpo,
com confiança por ponto) e instala por `pip`, enquanto o OpenPose exige
compilar Caffe com CUDA — inviável de reproduzir nas máquinas da equipe e no
CI. O mapeamento dos 33 pontos do MediaPipe para as 13 articulações que usamos
está em `_INDICES_MEDIAPIPE`. O backend aceita as duas APIs do MediaPipe: a
legada (`mp.solutions.pose`, até a série 0.10) e a Tasks (`PoseLandmarker`,
0.10.30+ e 1.x), porque a versão fixada originalmente (0.10.21) não existe
para o Python 3.13 usado pela equipe; o modelo `pose_landmarker_lite.task` é
baixado uma vez.

O YOLOv8 genérico (classes COCO) não conhece `dreno`, `mao_paciente` nem
`profissional`, e sem isso as regras de evento nunca disparariam em vídeo
real. Três adaptações resolvem isso sem treinar um detector:

- a área crítica pode ser informada manualmente (`--area-critica x1,y1,x2,y2`,
  em pixels), pelo fisioterapeuta que conhece a posição do dreno naquela
  sessão; o relatório registra a origem (`anotada`, `manual` ou `nenhuma`);
- quando o detector não fornece a caixa da mão, a mão do paciente passa a ser
  o keypoint do punho do estimador de pose, com um raio de 11 px (metade da
  caixa de mão das amostras); um teste verifica que, com a ROI manual e os
  punhos anotados, o achado de invasão reproduz o da caixa anotada;
- com o detector genérico, "profissional presente" passa a ser "duas ou mais
  pessoas em cena".

Os dois backends reais foram exercitados em testes com módulos `mediapipe` e
`ultralytics` falsos — que leem o `fisioterapia.mp4` sintético de verdade com
o OpenCV e devolvem landmarks e caixas controlados — e numa execução real
sobre o mesmo vídeo (MediaPipe 1.2, Ultralytics 8.4). A execução real roda de
ponta a ponta e registra zero quadros com pose: o esqueleto desenhado não é
uma pessoa, e isso é o resultado esperado. A validação com uma pessoa
filmada continua pendente (§8).

Sobre os keypoints calculamos três ângulos:

- **abdução de ombro**: ângulo entre o braço e o eixo vertical do corpo, com
  0° no braço ao longo do tronco e 180° acima da cabeça — a convenção clínica.
  O sinal de `dx` distingue abertura lateral de cruzamento do corpo;
- **inclinação lateral do tronco**: ângulo entre a linha que liga o centro dos
  quadris ao centro dos ombros e a vertical;
- **assimetria entre ombros**: diferença absoluta entre as duas abduções no
  mesmo quadro.

As cinco regras e seus limites:

| Regra | Limite | Severidade | Origem do limite |
| --- | ---: | --- | --- |
| Compensação de tronco | 15° | atenção | movimento compensatório durante exercício supervisionado |
| Amplitude de abdução do ombro | 120° | crítico | restrição de amplitude no pós-operatório recente (PROT-004) |
| Assimetria entre ombros | 20° | atenção | diferença aceitável entre os lados na mesma repetição |
| Mão na área crítica do dreno | qualquer sobreposição | crítico | PROT-004 |
| Profissional ausente de cena | 2 s | atenção | supervisão contínua exigida na sessão |

Duas salvaguardas contra ruído de estimação: keypoint com confiança abaixo de
0,35 não entra no cálculo, e um desvio só vira achado após persistir por 5
quadros consecutivos (≈0,17 s a 30 fps).

### 4.2 Áudio

A modalidade tem duas camadas, e a separação entre elas foi deliberada.

A camada acústica não usa nuvem. Os atributos saem direto da forma de onda,
com `wave` da biblioteca padrão e numpy: enquadramento em janelas de 25 ms com
passo de 10 ms, RMS por quadro, detecção de voz por limiar adaptado ao piso de
ruído, agrupamento em frases, F0 por autocorrelação na faixa de 70 a 350 Hz,
contagem de núcleos silábicos por picos do envelope de energia e centroide
espectral por FFT.

O motivo é que o Speech to Text devolve texto, não fisiologia da voz. Quem
indica fadiga ou esforço respiratório é a energia da fala, o tamanho das
pausas, a taxa de fala e a estabilidade da frequência fundamental — e isso se
perde na transcrição.

A camada de linguagem usa os serviços gerenciados pedidos no enunciado:

| Serviço | Uso | Substituto offline |
| --- | --- | --- |
| Azure Speech to Text | reconhecimento contínuo do WAV, com segmentos e confiança por trecho | transcrição de referência gravada com o áudio sintético |
| Azure Text Analytics | sentimento, frases-chave e entidades clínicas (Text Analytics for health, com NER genérico de reserva) | sentimento por léxico com massa neutra fixa |

Em todos os provedores, quem decide alerta de conteúdo é o léxico clínico de
`fase4/data/termos_criticos_audio.json`, com oito categorias e peso por
categoria. As frases-chave da Azure dizem o que é saliente no texto, não o que
é clinicamente crítico — essa é uma decisão institucional, não estatística.
Com a Azure, porém, a identificação dos termos passa a ser feita também pelo
serviço, como pede o enunciado: cada entidade clínica reconhecida é cruzada
com o léxico (a que casa herda categoria e peso) e um sintoma ou diagnóstico
fora do léxico, com confiança ≥ 0,8, entra como termo informativo — a Azure
identificou, o hospital ainda não decidiu a prioridade. Textos longos são
divididos em documentos de até 5.000 caracteres em fim de frase e enviados
em lotes de 10, porque o serviço limita 5.120 caracteres por documento e uma
consulta de 20 minutos transcrita passa disso com folga; o sentimento final é
a média dos scores ponderada pelo tamanho de cada documento.

Há um terceiro provedor só para a transcrição, `local`, que roda o
faster-whisper (modelo `small`, CPU, int8) na própria máquina. Ele existe
para demonstrar reconhecimento de fala real sem credencial da Azure e precisa
ser pedido explicitamente: `auto` continua escolhendo entre `azure` e
`offline`. Quando a transcrição vem de um reconhecedor real e há transcrição
de referência ao lado do WAV, o relatório calcula a taxa de erro de palavras
(WER), normalizando acentos, caixa e pontuação.

As quatro regras vocais:

| Regra | Critério |
| --- | --- |
| Fala entrecortada | proporção de pausa ≥ 0,42 **e** frase média ≤ 1,6 s |
| Fadiga vocal | queda de energia ≥ 30% **ou** taxa de fala ≤ 2,2 sílabas/s |
| Instabilidade vocal | variação relativa da F0 entre quadros ≥ 0,0025 |
| Padrão disartrico | taxa de fala ≤ 2,2 sílabas/s **e** (F0 instável **ou** desvio-padrão da F0 ≤ 8 Hz) |

O enunciado cita disartria entre os sintomas relacionados à fala. A regra
usa os marcadores acústicos clássicos descritos na literatura (Darley,
Aronson e Brown): fala lenta, controle fonatório alterado e voz monótona
(*monopitch*). Ela é um indicador de triagem, não distingue o tipo de
disartria e não substitui avaliação fonoaudiológica; a descrição da
instabilidade vocal também passou a nomear a articulação imprecisa como
indicador de disartria. O limiar de monotonia (8 Hz) foi calibrado nas
amostras sintéticas — 2,8 Hz na voz estável contra 15,9 Hz na voz com piora
— e precisaria ser recalibrado em fala natural, cuja variação de F0 é maior.

O último limiar merece uma ressalva. Nossa medida é a variação relativa da F0
entre quadros consecutivos de 10 ms, não o jitter período a período que
softwares de acústica clínica medem (cujos cortes patológicos ficam na casa de
1% a 2%). A escala é menor porque cada quadro de 25 ms já faz a média de vários
ciclos glotais. Calibramos o corte nas nossas amostras: 0,0006 na voz estável
contra 0,0038 na voz cansada. É útil para comparar gravações dentro deste
projeto; não é um indicador clínico.

Há ainda uma salvaguarda: se a transcrição vier vazia ou com confiança abaixo
de 0,50, as regras de texto não rodam. Alertar a equipe com base numa palavra
que o reconhecedor pode ter errado é pior que não alertar. Os atributos
acústicos continuam valendo nesse caso, porque não dependem da transcrição.

### 4.3 Anomalias em séries temporais

Começamos com um único detector (z-score robusto) e percebemos, testando na
série com deterioração, que ele não pegava a piora gradual. O motivo é
estrutural: quando a piora é lenta, ela entra na própria janela de referência,
a mediana móvel acompanha o paciente e o desvio desaparece. Chegamos então a
quatro detectores complementares, cada um cobrindo o ponto cego dos outros.

| Detector | Técnica | Pega | Não pega |
| --- | --- | --- | --- |
| z-score robusto | z modificado de Iglewicz e Hoaglin, `0,6745·(x−mediana)/MAD`, janela móvel de 60 min, corte 4,0 | mudança abrupta | piora lenta |
| Desvio de tendência | mediana dos últimos 30 min contra a mediana das 2 primeiras horas (basal) | piora gradual | mudança dentro da tolerância |
| Limite clínico | faixas críticas por canal, inspiradas em escores de deterioração tipo NEWS | valor fora da faixa | piora que ainda não cruzou o limite |
| Isolation Forest | 200 árvores treinadas no basal, corte no percentil 1 dos scores do próprio basal | combinação anômala entre os 6 canais | desvio de um canal só, dentro da faixa |

Três decisões de projeto nesta parte:

**Todos os detectores são incrementais.** Os três por canal recebem uma
amostra por vez e decidem com o que já passou. Isso permitiu usar o mesmo
código no processamento em lote (`avaliar_serie`) e no monitoramento em tempo
real (`monitorar_em_tempo_real`), sem duplicar regras — e um teste verifica
que os dois caminhos produzem exatamente os mesmos achados. O Isolation Forest
tem uma versão online (`DetectorMultivariadoOnline`) que acumula as duas
primeiras horas, ajusta o modelo uma única vez e pontua cada leitura seguinte;
o episódio é reconstruído com a mesma regra do lote (união de pontos anômalos
separados por até 5 minutos) e o achado sai no minuto em que o episódio
completa 15 minutos — o preço de não olhar o futuro. Um teste confirma que o
episódio do `PAC-0006` é o mesmo nos dois modos (mesmo início, mesmo corte,
mesmos canais alterados).

**Leitura faltante não é leitura.** Canal ausente no arquivo é simplesmente
ignorado, e uma leitura `NaN` (buraco de sensor) não entra em nenhuma regra:
ela não é normal nem anormal. Sem isso, um único `NaN` nas duas primeiras
horas silenciava para sempre a regra de tendência daquele canal (a mediana do
basal virava `NaN`) e derrubava o Isolation Forest. A correção veio da
validação em dados reais (§5.10), onde buracos são a regra, não a exceção.

**Todos exigem leituras consecutivas e têm histerese.** A confirmação em 3
leituras seguidas é o que separa deterioração de artefato de sensor. A
histerese — a mesma regra só pode disparar de novo depois de 30 minutos com a
condição ausente — é o que impede o detector de emitir um achado por minuto
durante as três horas de piora.

**O Isolation Forest é treinado só no basal, não na série inteira.** Treinar
com a série completa usaria informação do futuro para decidir o presente e
inflaria artificialmente o resultado. No monitoramento real, o que existe
quando o paciente é admitido é o basal dele. O corte de decisão também vem do
basal: um ponto é anômalo quando fica mais isolado que 99% do que já se viu
daquele paciente.

Os limiares de deslocamento mínimo da regra de tendência (10 bpm, 12 mmHg,
2% de saturação, 0,4 °C, 3 irpm) existem porque o deslocamento precisa ser
maior que a variação fisiológica normal do dia. Na primeira versão, a
oscilação circadiana que colocamos no gerador era grande demais e o detector
de tendência acusava o paciente estável — tivemos que corrigir o gerador e
calibrar as tolerâncias.

### 4.4 Anomalias em prescrições

Três regras, escolhidas por serem as que mais aparecem em revisão de
prescrição:

| Regra | Critério | Severidade |
| --- | --- | --- |
| Salto de dose | aumento ≥ 50% sobre a dose vigente | crítico para opioide, atenção nas demais classes |
| Suspensão precoce de antimicrobiano | interrupção antes de 80% do esquema previsto | crítico |
| Duplicidade terapêutica | dois medicamentos da mesma classe ativos ao mesmo tempo | crítico |

A terceira regra usa uma lista explícita de classes em que a duplicidade é
problema (anti-inflamatórios, opioides, anticoagulantes), porque em outras
classes a associação é intencional.

### 4.5 Anomalias de movimentação

Duas regras sobre o índice horário de atividade do sensor do leito:
imobilidade diurna (índice abaixo de 8 por 5 horas seguidas, apenas em horário
diurno — dormir à noite não é imobilidade) e agitação noturna (índice acima de
55 em pelo menos 2 horas entre 22:00 e 06:00).

### 4.6 Fusão multimodal

Cinco regras combinam achados de modalidades diferentes em quadros clínicos:

| Quadro | Combinação exigida |
| --- | --- |
| `deterioracao_respiratoria` | queixa respiratória ou fala entrecortada **+** piora de saturação ou de frequência respiratória |
| `suspeita_infeccao` | relato de febre ou alteração da ferida **+** febre medida **+** repercussão hemodinâmica |
| `risco_medicamentoso` | alteração inesperada de prescrição **+** piora clínica ou queixa de dor |
| `reabilitacao_insegura` | execução do exercício fora do protocolo **+** relato de dor ou cansaço |
| `risco_de_imobilidade` | imobilidade prolongada **+** agitação noturna ou piora de sinais vitais |

Um quadro só é formado se os achados vierem de pelo menos duas modalidades
diferentes; caso contrário não é fusão, é repetir o que uma modalidade sozinha
já disse. Achados com instante absoluto precisam cair na mesma janela de 12
horas, que é a unidade de tempo do plantão. Achados de vídeo e áudio marcam
posição dentro da gravação (`mm:ss`), que não é comparável com o timestamp do
monitor; nesses casos a correlação temporal não é exigida, porque a gravação
pertence ao período monitorado por construção do pipeline.

Optamos por regras escritas, e não por um modelo de fusão treinado. Não temos
dados rotulados de desfecho clínico para treinar fusão supervisionada, e um
escore aprendido a partir de dados que nós mesmos geramos daria uma falsa
impressão de validade. Regras explícitas podem ser conferidas e corrigidas por
um médico, o que é o que faz sentido num trabalho acadêmico.

O risco do paciente (0-100) soma os achados com peso por severidade, adiciona
bônus por quadro identificado e por número de modalidades envolvidas, e satura
numa função de rendimento decrescente — vinte achados de atenção não devem
valer mais que um achado crítico confirmado por duas modalidades.

### 4.7 Alertas

Para reduzir avisos repetidos sobre o mesmo paciente, usamos três mecanismos:

| Mecanismo | O que faz |
| --- | --- |
| Consolidação | achados da mesma família viram um alerta só, que lista todos |
| Absorção | quando a fusão identifica um quadro, as famílias que o sustentam não geram alerta próprio — quem avisa é o alerta do quadro, mais informativo e de prioridade maior |
| Supressão temporal | o mesmo grupo não é reenviado dentro de 15 minutos |

Com os três, os 47 achados do `PAC-0006` produzem 6 alertas. Duas garantias
acompanham a absorção: o alerta do quadro nunca é menos grave que o pior
achado que ele absorveu, e ele é roteado para as equipes de todas as
modalidades envolvidas — foi a soma delas que levantou o quadro, e cada uma
tem uma parte da conduta.

O roteamento por modalidade existe porque, no hospital do projeto, quem
acompanha a sessão de fisioterapia não é quem responde por uma alteração de
prescrição:

| Modalidade | Destinatário |
| --- | --- |
| vídeo | equipe de reabilitação e cirurgia responsável |
| áudio | médico assistente da consulta |
| sinais vitais | equipe de plantão do leito |
| prescrição | farmácia clínica e médico prescritor |
| movimentação | equipe de enfermagem do leito |
| quadro multimodal | médico de plantão e enfermagem, mais as equipes das modalidades envolvidas |

Achados informativos entram no relatório mas nunca geram notificação.

**Entrega.** O alerta não termina no terminal. O motor recebe uma lista de
canais (`CanalEntrega`) e, ao final de cada ciclo, entrega cada alerta a cada
canal: uma fila *append-only* em JSON Lines (`fila_alertas.jsonl` na pasta da
execução, sempre) e um webhook HTTP (`POST` do alerta em JSON para
`FASE4_WEBHOOK_URL`, quando definida). A fila é o contrato mais simples que um
sistema hospitalar consegue consumir; o webhook é o caminho para um painel,
um pager ou um canal de mensagens. Falha de canal vira evento
`entrega_falhou` na auditoria e nunca interrompe o monitoramento; o alerta
registra por onde passou (`canais_entrega`), e isso aparece no relatório.
`fase4/receptor_webhook_demo.py` sobe um receptor local que imprime os
alertas recebidos, para a demonstração do fluxo final. Um teste levanta esse
receptor numa porta livre e verifica que os três alertas de um ciclo chegam
por POST; outro aponta o webhook para uma porta fechada e verifica que o
pipeline segue, com as falhas contadas e a fila intacta.

## 5. Resultados

### 5.1 Consolidado

| Métrica | `PAC-0006` (deterioração) | `PAC-0002` (controle) |
| --- | ---: | ---: |
| Modalidades processadas | 5 | 4 |
| Achados das modalidades | 42 | 5 |
| — críticos | 12 | 0 |
| — de atenção | 20 | 0 |
| — informativos | 10 | 5 |
| Quadros multimodais | 5 | 0 |
| Achados totais | 47 | 5 |
| Alertas emitidos | **6** | **0** |
| Risco calculado | **100,0 (alto)** | **6,5 (baixo)** |

Os 42 achados das modalidades vêm de vídeo (9), áudio (10), sinais vitais (18),
prescrições (3) e movimentação (2); os 5 quadros multimodais são produzidos
pela fusão a partir deles.

Os 5 achados do controle são todos informativos — leituras isoladas não
confirmadas, registradas como prováveis artefatos de sensor. Nenhum alerta foi
emitido para ele, que é o resultado desejado.

### 5.2 Anomalias detectadas nos sinais vitais

A piora foi injetada a partir das 16:00. A tabela mostra quando cada detector
percebeu:

| Instante | Atraso | Detector | Achado |
| --- | ---: | --- | --- |
| 16:51 | +51 min | tendência | frequência cardíaca em alta sustentada (93,5 contra 82,7 do basal) |
| 17:01 | +61 min | tendência | temperatura em alta sustentada (37,3 contra 36,9) |
| 17:07 | +67 min | tendência | frequência respiratória em alta (19,2 contra 16,2) |
| **17:08** | **+68 min** | **Isolation Forest** | **combinação anômala de frequência cardíaca, temperatura e frequência respiratória** |
| 17:20 | +80 min | tendência | saturação em queda (95,0 contra 97,1) |
| 17:43 | +103 min | tendência | pressão sistólica em queda (112,8 contra 125,1) |
| 17:44 | +104 min | tendência | pressão diastólica em queda (70,1 contra 78,6) |
| 18:03 | +123 min | limite clínico | temperatura acima de 38 °C (38,08 °C) |
| 18:17 | +137 min | limite clínico | frequência respiratória acima de 24 irpm (24,2 irpm) |

O resultado que justifica ter quatro detectores em vez de um: o primeiro
alerta sai às 16:51, e o primeiro limite clínico só é cruzado às 18:03. A
detecção por tendência antecipa em **1 h 12 min** o momento em que a
abordagem por limiar fixo — a mais comum em monitores de leito — teria
disparado. O Isolation Forest confirma o quadro de forma independente 55
minutos antes do limite, e é o único que nomeia a *combinação* de canais, que
é o que caracteriza a deterioração sistêmica.

### 5.3 O artefato de sensor

O pico isolado de 185 bpm às 10:30 ultrapassa o limite crítico de frequência
cardíaca (130 bpm). O sistema o tratou como esperado:

```text
10:30  (nota) Leitura isolada de frequencia cardiaca fora do padrao recente
              (185); nao se repetiu nas leituras seguintes e foi tratada como
              artefato de sensor
```

Nenhum alerta foi emitido às 10:30. A regra de confirmação em 3 leituras
consecutivas, aplicada também à regra de limite clínico, é o que produziu esse
comportamento. Um teste automatizado verifica justamente isso: que o pico de
185 bpm gera exatamente um achado informativo e nenhum achado crítico naquele
instante.

Ao longo das 4.320 leituras do turno (720 minutos × 6 canais), o detector
registrou 9 leituras não confirmadas no `PAC-0006` e 5 no `PAC-0002` — cerca
de 0,2% e 0,1%, compatível com o corte de 4 desvios adotado.

### 5.4 Anomalias detectadas no vídeo

| Instante | Severidade | Evento | Medida |
| --- | --- | --- | ---: |
| 00:10,00 | atenção | profissional ausente de cena por 4,0 s | limite 2 s |
| 00:11,40 | atenção | compensação de tronco | 19,4° (limite 15°) |
| 00:11,60 | crítico | amplitude de abdução do ombro esquerdo | 139,6° (limite 120°) |
| 00:11,77 | atenção | assimetria entre ombros | 49,2° (limite 20°) |
| 00:13,20 | crítico | mão do paciente na área do dreno | sobreposição total |
| 00:13,37 | atenção | assimetria entre ombros | 25,9° |
| 00:14,87 | crítico | amplitude de abdução do ombro esquerdo | 133,8° |
| 00:14,90 | atenção | assimetria entre ombros | 40,1° |
| 00:15,07 | atenção | compensação de tronco | 17,8° |

Os cinco tipos de evento injetados foram todos detectados, nas repetições
corretas (a 4ª e a 5ª). As três primeiras repetições e a última, executadas
dentro do protocolo, não geraram nenhum achado — e o ombro direito, que nunca
passou de 97,2°, também não.

### 5.5 Anomalias detectadas no áudio

| Instante | Severidade | Achado |
| --- | --- | --- |
| 00:00,60 | atenção | fala entrecortada (pausa 0,52; frase média 1,30 s) |
| 00:00,60 | atenção | queixa de fadiga: "muito cansada", "nao aguento" |
| 00:03,37 | crítico | queixa respiratória: "sem ar" |
| 00:05,75 | crítico | queixa de dor torácica: "aperto no peito" |
| 00:08,65 | crítico | queixa de infecção: "febre" |
| 00:11,44 | atenção | queixa de dor: "dor aumentou" |
| 00:16,05 | atenção | fadiga vocal (queda de energia 0,42; taxa de fala 1,31 sílabas/s) |
| — | atenção | instabilidade vocal (variação da F0 0,0038) |
| — | atenção | padrão disartrico (taxa de fala reduzida, F0 instável) |

A comparação entre os dois pacientes mostra que os atributos acústicos
separam bem os casos:

| Atributo | `PAC-0006` (cansada) | `PAC-0002` (estável) |
| --- | ---: | ---: |
| Proporção de pausa | 0,52 | 0,17 |
| Duração média da frase | 1,30 s | 3,35 s |
| Taxa de fala | 1,31 sílabas/s | 3,62 sílabas/s |
| Queda de energia | 0,42 | 0,05 |
| Variação da F0 | 0,0038 | 0,0006 |
| F0 média | 173,4 Hz | 201,9 Hz |

Nenhuma das quatro regras vocais disparou no `PAC-0002`, e nenhum termo
crítico foi encontrado na transcrição dele. A voz estável tem desvio de F0 de
2,8 Hz — abaixo do limiar de monotonia — mas a taxa de fala normal impede o
padrão disartrico, que exige as duas coisas.

### 5.6 Anomalias detectadas nas prescrições e na movimentação

| Instante | Severidade | Achado |
| --- | --- | --- |
| 02:00 | atenção | atividade elevada em 3 horas da madrugada (agitação ou delirium) |
| 09:00 | atenção | paciente praticamente imóvel por 6 horas seguidas durante o dia |
| 14:05 | crítico | aumento de 400% na dose de morfina (2 mg para 10 mg) |
| 15:40 | crítico | ceftriaxona suspenso no dia 2 de um esquema previsto para 7 dias |
| 16:20 | crítico | cetoprofeno e ibuprofeno ativos ao mesmo tempo (mesma classe) |

As três alterações de prescrição injetadas foram detectadas, cada uma pela
regra correspondente, e nenhuma foi acusada no `PAC-0002`, cuja evolução de
prescrição é coerente (inclusive um ajuste de frequência de dipirona, que não
é salto de dose).

### 5.7 Quadros identificados pela fusão

| Quadro | Modalidades | Achados absorvidos |
| --- | --- | ---: |
| `deterioracao_respiratoria` | áudio + sinais vitais | 16 |
| `suspeita_infeccao` | áudio + sinais vitais | 14 |
| `risco_medicamentoso` | prescrição + áudio + sinais vitais | 17 |
| `reabilitacao_insegura` | vídeo + áudio | 16 |
| `risco_de_imobilidade` | sinais vitais + movimentação | 11 |

O caso do `deterioracao_respiratoria` ilustra o ganho da fusão. Isoladamente,
a queixa "fico sem ar" na consulta é uma informação subjetiva, e a queda de
saturação de 97% para 95% está dentro de uma faixa que muitos monitores nem
sinalizam. Juntas, na mesma janela de plantão, formam um quadro de
deterioração respiratória com conduta definida — repetir a oximetria e
considerar suporte de oxigênio.

### 5.8 Alertas finais

| Prioridade | Alerta | Destino |
| --- | --- | --- |
| vermelho | queixa respiratória com piora objetiva de oxigenação ou ventilação | plantão + médico da consulta + equipe do leito |
| vermelho | alteração inesperada de prescrição em paciente com piora clínica | plantão + farmácia clínica + médico da consulta + equipe do leito |
| vermelho | execução do exercício fora do protocolo com relato de dor ou cansaço | plantão + reabilitação + médico da consulta |
| vermelho | relato de febre com febre medida e repercussão hemodinâmica | plantão + médico da consulta + equipe do leito |
| vermelho | imobilidade prolongada com piora de sinais vitais | plantão + equipe do leito + enfermagem |
| laranja | paciente executou o exercício sem profissional em cena por 4,0 s | reabilitação |

Cinco dos seis alertas são quadros multimodais; o sexto é o único achado de
vídeo que nenhum quadro explicava. Dos 47 achados, 37 eram notificáveis (os 10
informativos nunca geram alerta) e resultaram em 6 notificações — uma redução
de 84% em relação aos achados notificáveis.

Nenhuma informação se perdeu no caminho: cada alerta carrega os achados que o
sustentam, e o relatório consolidado lista todos os 47 com instante e
evidência. Os seis alertas foram entregues na fila
`resultados/fase4/fila_alertas.jsonl` (seis linhas, uma por alerta), e cada
um registra `canais_entrega = ["fila_jsonl"]`.

### 5.9 Integração com a Azure

A integração com Azure Speech to Text e Azure Text Analytics está
implementada em `fase4/azure_services.py`. Os provedores usam as estruturas
`ResultadoTranscricao` e `ResultadoAnaliseTexto`. Os testes executam o modo
offline e simulam as respostas dos dois SDKs: no Speech, sucesso,
cancelamento, ausência de fala e tempo limite; no Text Analytics, a divisão
de um texto de 61 mil caracteres em 13 documentos e 2 lotes, a média
ponderada do sentimento, a herança de categoria e peso por uma entidade que
casa com o léxico, o sintoma fora do léxico que vira termo informativo, a
queda para o NER genérico quando o recurso de saúde não existe e a falha
explícita quando um documento retorna erro. Isso não comprova a execução dos
serviços em nuvem.

Os resultados numéricos deste relatório foram produzidos com o provedor
`offline`, sem credenciais Azure disponíveis nesta execução. O relatório
gerado registra isso explicitamente, em vez de omitir:

```text
| Serviço          | Provedor efetivo |
| speech_to_text   | offline          |
| text_analytics   | offline          |
```

Com as variáveis `AZURE_SPEECH_KEY`/`AZURE_SPEECH_REGION` e
`AZURE_LANGUAGE_KEY`/`AZURE_LANGUAGE_ENDPOINT` definidas e os SDKs de
`requirements-fase4.txt` instalados, o provedor `auto` passa a usar a nuvem
sem nenhuma outra mudança. Se `--provedor-azure azure` for pedido sem
credencial, o pipeline falha com mensagem explícita — preferimos isso a cair
silenciosamente para o modo local, porque uma execução que diz ter usado a
Azure precisa ter usado a Azure.

### 5.10 Validação dos detectores em dados reais (PhysioNet)

O enunciado sugere o PhysioNet. Baixamos um registro numérico de acesso
aberto do **MIMIC-III Waveform Database Matched Subset** (v1.0, licença
ODbL, DOI 10.13026/c2294b): `p04/p044083/p044083-2112-05-04-19-50n`, 45,8 h
de monitor de leito a 1 Hz com frequência cardíaca, pressão arterial
invasiva, SpO2 e frequência respiratória. `fase4/data/baixar_physionet.py`
reamostra para mediana por minuto, arrasta buracos de até 5 minutos,
descarta minutos sem leitura completa e grava no esquema de
`sinais_vitais.csv`, com `fonte.json` registrando procedência, mapeamento de
sinais e preenchimentos. Não há temperatura nos numéricos do MIMIC; a coluna
fica ausente e os detectores seguem com cinco canais.

Das primeiras 12 horas (721 minutos), 679 foram exportados e 42 descartados
por falta de leitura. `fase4/validacao_physionet.py` aplicou os mesmos
detectores, com os mesmos limiares calibrados nas amostras sintéticas:

| Métrica | Valor |
| --- | ---: |
| Minutos analisados | 679 (11,3 h) |
| Achados | 48 |
| — informativos (leituras isoladas não confirmadas) | 23 |
| — de atenção (tendência e mudança abrupta) | 15 |
| — críticos (limite clínico e multivariado) | 10 |
| Achados notificáveis por hora | 2,2 |
| Episódios do Isolation Forest em lote | 4 |
| Episódios do Isolation Forest em tempo real | 4 |
| Alertas após consolidação | 4 |

O registro é de um paciente de UTI com pressão arterial caindo de 136 mmHg
(mediana do basal) para 108 mmHg e saturação chegando a 71%, e os detectores
reagiram a isso: tendências de queda de pressão sistólica e diastólica,
limites críticos de pressão diastólica, saturação e frequência respiratória,
e quatro episódios multivariados concentrados na madrugada — idênticos no
lote e no modo online. Também apareceu o que esperávamos de dados reais: 23
leituras isoladas não confirmadas em 11 horas, tratadas como artefato e
nunca como alerta, e 42 minutos descartados por buraco de sensor, que foi o
que motivou as guardas contra `NaN` (§4.3).

O que isso valida e o que não valida precisa ficar claro. Não há rótulo de
desfecho no registro, então não medimos sensibilidade nem especificidade.
Medimos que os detectores rodam em dados reais sem erro, que a taxa de
alertas após consolidação (4 em 11 horas) é compatível com um painel de
plantão, e que o detector online reproduz o lote. O relatório completo está
em `resultados/fase4/physionet/p044083-2112-05-04-19-50n/relatorio_physionet.md`.

### 5.11 Fala real com transcrição local

O `consulta.wav` sintético não tem palavras. Para exercitar a camada de
linguagem em fala real, `fase4/data/gerar_fala_tts.py` lê a transcrição de
referência de cada paciente com a voz pt-BR do Windows (Microsoft Maria),
preserva as pausas da referência e grava `consulta_fala.wav` (24,9 s para o
`PAC-0006`, 14,3 s para o `PAC-0002`). O pipeline foi executado com
`--audio consulta_fala.wav --provedor-azure local` e os resultados estão em
`resultados/fase4/fala_real/`:

| | `PAC-0006` | `PAC-0002` |
| --- | ---: | ---: |
| Confiança média da transcrição | 0,90 | 0,77 |
| Taxa de erro de palavras (WER) | **0,0%** | 11,4% |
| Termos críticos encontrados | sem ar, aperto no peito, febre, muito cansada, não aguento, dor aumentou | nenhum |
| Sentimento (léxico) | negativo | positivo |

A transcrição real do `PAC-0006` reproduziu palavra a palavra a referência e
o léxico encontrou todos os seis termos críticos, com instante por segmento;
no `PAC-0002`, os erros do reconhecedor não criaram nenhum falso termo
crítico. É o comportamento que a Azure deve ter quando for executada com a
mesma gravação.

Dois resultados secundários são informativos. Primeiro, as regras acústicas
calibradas no sinal sintético não se aplicam à voz do TTS: a variação da F0
medida fica em 0,022 nas duas gravações (contra o corte de 0,0025) e a
instabilidade vocal dispara para os dois pacientes, enquanto a fadiga vocal e
o padrão disartrico não disparam para nenhum. A voz sintetizada é articulada
e estável por construção, e o nosso *jitter* não é o da acústica clínica
(§4.2): o resultado confirma a limitação já declarada, não a contradiz.
Segundo, a fala entrecortada disparou no `PAC-0006` porque as pausas da
referência foram preservadas — a regra responde ao padrão temporal da fala,
e não à voz.

## 6. Monitoramento em tempo real

O modo `--tempo-real` percorre a série minuto a minuto, como se as leituras
estivessem chegando do monitor do leito:

```powershell
python -m fase4.cli_demo --paciente-id PAC-0006 --tempo-real
```

Ele existe para deixar verificável que as regras de sinais vitais não
dependem de ver a série inteira: a cada minuto o detector decide com o que já
passou. Um teste automatizado compara a saída do modo incremental com a do
processamento em lote e exige que os achados sejam idênticos, na mesma ordem.

O Isolation Forest entra nesse laço pelo `DetectorMultivariadoOnline`
(§4.3): treina nas duas primeiras horas e pontua cada leitura seguinte, e o
episódio é confirmado no minuto em que completa 15 minutos.

Os *achados* são os mesmos nos dois modos, mas os *alertas* não: no modo em
tempo real a execução emite 5 alertas em vez dos 6 do lote. A diferença é
esperada e vem da supressão temporal. Das seis tendências detectadas, três são
agrupadas porque caem na janela de 15 minutos de uma anterior (a alta de
temperatura das 17:01 dentro da janela aberta às 16:51, a queda de saturação
das 17:20 dentro da aberta às 17:07, e a queda de pressão diastólica das 17:44
dentro da aberta às 17:43); o mesmo vale para o limite crítico de frequência
respiratória das 18:17, suprimido pelo de temperatura das 18:03. É o
comportamento desejado: a equipe de plantão recebe cinco avisos ao longo de
duas horas, e não um por minuto. No modo em lote, em que o turno inteiro é
avaliado de uma vez, a consolidação por família dá o mesmo efeito.

A saída do turno completo do `PAC-0006`, com cada alerta entregue na fila:

```text
10:30  (nota) Leitura isolada de frequencia cardiaca fora do padrao recente (185)
16:23  (isolation forest) episodio multivariado desde 16:09
16:23  [VERMELHO] Combinacao anomala de sinais vitais (frequencia cardiaca, temperatura, frequencia respiratoria) [fila_jsonl]
16:51  [LARANJA] Tendencia de alta sustentada de frequencia cardiaca: 93.5 contra 82.7 [fila_jsonl]
17:07  [LARANJA] Tendencia de alta sustentada de frequencia respiratoria: 19.2 contra 16.2 [fila_jsonl]
17:43  [LARANJA] Tendencia de queda sustentada de pressao sistolica: 112.8 contra 125.1 [fila_jsonl]
18:03  [VERMELHO] Temperatura acima do limite critico: 38.08 C (limite 38) [fila_jsonl]
```

O primeiro alerta do turno passa a ser o do Isolation Forest online, às
16:23 — 14 minutos depois do início do episódio (16:09), 28 minutos antes da
primeira tendência confirmada e **1 h 40 min** antes do primeiro limite
clínico. No lote, o mesmo episódio aparecia com o instante do pior score
(17:08), que é quando ele fica mais evidente, não quando começa; o modo online
revela que a combinação anômala já era detectável às 16:09, nove minutos
depois da piora injetada. É o resultado mais relevante desta fase para o
objetivo de "detecção em tempo real para monitoramento preventivo".

## 7. Testes

```powershell
python -m unittest discover -s tests -v
```

São 237 testes no total: os 74 das fases anteriores, que continuam passando, e
163 desta fase, em sete arquivos. Nenhum depende de rede, de chave da Azure
nem dos pacotes opcionais: o pipeline usa o modo offline, os SDKs da Azure, o
MediaPipe, o YOLOv8 e o faster-whisper são substituídos por módulos falsos
injetados em `sys.modules`, e o webhook é testado contra um servidor HTTP
local. As amostras são geradas em diretório temporário.

Os testes da Fase 4 cobrem, entre outros:

| Grupo | Verificações |
| --- | --- |
| Geração de amostras | reprodutibilidade por `seed`, estrutura das séries, modalidade ausente |
| Ângulos articulares | casos de referência conhecidos (0°, 45°, 90°, 140°), braço cruzando o corpo |
| Filtro de ruído de pose | pico isolado não vira desvio; keypoint de baixa confiança é ignorado |
| Backends reais de vídeo | MediaPipe (API legada e Tasks) e YOLOv8 falsos lendo o vídeo real; ROI manual reproduz a invasão anotada; "duas pessoas em cena"; sessão completa com os dois backends |
| Atributos acústicos | voz cansada contra voz estável em cada atributo; padrão disartrico dispara só com fala lenta |
| WER e transcrição local | distância de edição por palavra; whisper falso com confiança por `avg_logprob`; `auto` nunca escolhe `local` |
| Serviços Azure | Speech: sucesso, cancelamento, sem fala, tempo limite; Text Analytics: chunking, lotes de 10, sentimento ponderado, entidades, NER de reserva, erro por documento |
| Máquina de estados das regras | confirmação, não repetição, artefato, histerese |
| Detectores de série | controle sem alerta, artefato não confirmado, tendência antes do limite, equivalência lote/tempo real, Isolation Forest online igual ao lote, canal ausente e `NaN` |
| Prescrições | as três regras, e os casos em que não devem disparar |
| Fusão | exigência de duas modalidades, severidade nunca rebaixada, absorção de famílias, risco limitado a 100 |
| Alertas e entrega | consolidação, absorção, prioridade, supressão temporal, roteamento multimodal; fila com uma linha por alerta; webhook recebido por servidor local; falha de canal não derruba o motor |
| PhysioNet | reamostragem e renomeação, fallback para pressão não invasiva, buracos longos descartados, valores impossíveis, runner sobre CSV sintético, round-trip WFDB |
| Pipeline | caso completo, controle sem alerta, modalidade ausente, redução de notificações, fila gravada na raiz da execução |

## 8. Limitações

São limitações do trabalho, não detalhes a ajustar:

1. **Os dados principais são sintéticos.** Os limiares foram calibrados
   contra amostras que nós mesmos geramos. Eles não têm validade clínica e
   serviriam apenas de ponto de partida num estudo com dados reais. A
   validação no registro do PhysioNet (§5.10) mostra comportamento e taxa de
   alertas em dados reais, mas sem rótulo de desfecho não mede acurácia.
2. **O vídeo não é uma gravação de pessoa.** É um esqueleto articulado
   desenhado programaticamente. Os keypoints vêm de anotação, não de inferência
   sobre pixels. Os backends MediaPipe e YOLOv8 rodam de ponta a ponta sobre
   esse vídeo (e são testados com módulos falsos), mas não foram exercitados
   com uma pessoa filmada, porque não temos vídeo clínico autorizado. A área
   crítica em vídeo real depende de ROI informada manualmente.
3. **O áudio principal não é fala humana.** É um sinal vozeado sintetizado
   com F0, jitter, taxa silábica e ruído respiratório controlados. A fala
   gerada por TTS (§5.11) é inteligível e foi transcrita por um reconhecedor
   real, mas é uma voz sintética estável: valida a camada de linguagem, não
   as regras acústicas de fadiga e disartria, cujos limiares não se transferem
   para voz natural sem recalibração.
4. **A Azure não foi executada de fato.** A integração está implementada e
   testada quanto ao contrato (incluindo entidades clínicas e divisão em
   documentos), mas os números deste relatório vieram dos provedores
   `offline` e `local`. O `consulta_fala.wav` deixa a execução com a Azure
   pronta para quando houver credencial.
5. **A fusão é baseada em regras.** Sem dados rotulados de desfecho não há
   como treinar nem avaliar uma fusão supervisionada. As cinco regras cobrem
   os quadros que os dados sintéticos contêm, e não um conjunto exaustivo de
   apresentações clínicas.
6. **O jitter medido não é o jitter da literatura.** É a variação relativa da
   F0 entre quadros de 10 ms, numa escala própria, calibrada nas nossas
   amostras.
7. **Não há avaliação de sensibilidade e especificidade em população.** Com
   dois pacientes é possível dizer que o sistema detectou tudo o que foi
   injetado em um e não alertou o outro; não é possível estimar taxas.

## 9. Conclusão

O protótipo implementa os módulos de vídeo, áudio, detecção de anomalias,
fusão multimodal e entrega de alertas. A execução principal deste relatório
usa dados sintéticos, poses anotadas e transcrições de referência; as
execuções complementares usam um registro real de monitor de leito do
PhysioNet e uma consulta com fala inteligível transcrita por reconhecedor
real. As integrações com a Azure e os modelos de percepção em pessoa filmada
ainda precisam ser executados com gravações autorizadas para completar a
demonstração exigida no enunciado.

Os alertas são entregues numa fila JSON Lines e, quando configurado, por
webhook HTTP a um receptor; não há integração com um sistema hospitalar real.
O modo incremental reproduz uma série gravada; não está conectado a um
monitor. Também está pendente a gravação e publicação do vídeo de até 15
minutos no YouTube ou Vimeo.

O YOLOv8 padrão reconhece classes genéricas e não identifica drenos nem
distingue paciente e profissional; em vídeo real a área crítica vem de ROI
manual e a mão do keypoint do punho. O áudio sintético principal não contém
fala inteligível; a fala por TTS cobre a camada de linguagem.

Três resultados nos parecem os mais relevantes. O primeiro é a antecipação:
no modo em tempo real, o Isolation Forest online sinaliza a combinação
anômala às 16:23, 1 h 40 min antes de qualquer limiar fixo ser cruzado, e a
detecção por tendência confirma às 16:51 — exatamente o tipo de margem que
importa num monitoramento preventivo. O segundo é a redução de ruído: a
fusão multimodal transformou 37 achados notificáveis em 6 alertas
contextualizados, roteados e entregues, sem descartar informação — sem isso,
o sistema teria o defeito que mais inviabiliza painéis clínicos na prática,
que é alertar tanto que ninguém olha. O terceiro é que os detectores se
comportaram em dados reais como se comportaram nos sintéticos: 11 horas de
um paciente de UTI do MIMIC-III produziram 4 alertas consolidados, quatro
episódios multivariados idênticos em lote e online, e nenhum erro diante dos
buracos de sensor que dados reais sempre têm.

---

**Autores:** Antonio Miranda, Elaine, Marcos Mol, Lucas da Costa, Ricardo
Loureiro — AI for Devs (8IADT)
