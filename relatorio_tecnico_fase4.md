# Relatório técnico - Tech Challenge Fase 4

## 1. Escopo

Nesta fase, implementamos o monitoramento contínuo de pacientes a partir de
dados multimodais. O sistema analisa vídeo de sessões de fisioterapia, áudio
de consultas médicas e séries temporais de sinais vitais, prescrições e
movimentação no leito; combina os achados das três modalidades; e emite
alertas automáticos para a equipe médica.

Mantivemos o módulo isolado em `fase4/`, como fizemos com a Fase 3, para não
alterar o funcionamento das fases anteriores. As 74 verificações automatizadas
que já existiam continuam passando, e acrescentamos 84 testes específicos
desta fase.

Todos os dados são sintéticos e gerados pelo próprio repositório. Nenhum
resultado foi validado para uso assistencial: qualquer conduta exige avaliação
do profissional responsável.

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
está em `_INDICES_MEDIAPIPE`.

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
| Azure Text Analytics | sentimento e frases-chave | sentimento por léxico com massa neutra fixa |

Nos dois provedores, quem decide alerta de conteúdo é o léxico clínico de
`fase4/data/termos_criticos_audio.json`, com oito categorias e peso por
categoria. As frases-chave da Azure dizem o que é saliente no texto, não o que
é clinicamente crítico — essa é uma decisão institucional, não estatística.

As três regras vocais:

| Regra | Critério |
| --- | --- |
| Fala entrecortada | proporção de pausa ≥ 0,42 **e** frase média ≤ 1,6 s |
| Fadiga vocal | queda de energia ≥ 30% **ou** taxa de fala ≤ 2,2 sílabas/s |
| Instabilidade vocal | variação relativa da F0 entre quadros ≥ 0,0025 |

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

**Os três primeiros detectores são incrementais.** Recebem uma amostra por vez
e decidem com o que já passou. Isso permitiu usar o mesmo código no
processamento em lote (`avaliar_serie`) e no monitoramento em tempo real
(`monitorar_em_tempo_real`), sem duplicar regras — e um teste verifica que os
dois caminhos produzem exatamente os mesmos achados.

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

Com os três, os 46 achados do `PAC-0006` produzem 6 alertas. Duas garantias
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

## 5. Resultados

### 5.1 Consolidado

| Métrica | `PAC-0006` (deterioração) | `PAC-0002` (controle) |
| --- | ---: | ---: |
| Modalidades processadas | 5 | 4 |
| Achados das modalidades | 41 | 5 |
| — críticos | 12 | 0 |
| — de atenção | 19 | 0 |
| — informativos | 10 | 5 |
| Quadros multimodais | 5 | 0 |
| Achados totais | 46 | 5 |
| Alertas emitidos | **6** | **0** |
| Risco calculado | **100,0 (alto)** | **6,5 (baixo)** |

Os 41 achados das modalidades vêm de vídeo (9), áudio (9), sinais vitais (18),
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

Nenhuma das três regras vocais disparou no `PAC-0002`, e nenhum termo crítico
foi encontrado na transcrição dele.

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
vídeo que nenhum quadro explicava. Dos 46 achados, 36 eram notificáveis (os 10
informativos nunca geram alerta) e resultaram em 6 notificações — uma redução
de 83% em relação aos achados notificáveis.

Nenhuma informação se perdeu no caminho: cada alerta carrega os achados que o
sustentam, e o relatório consolidado lista todos os 46 com instante e
evidência.

### 5.9 Integração com a Azure

A integração com Azure Speech to Text e Azure Text Analytics está
implementada em `fase4/azure_services.py`. Os provedores usam as estruturas
`ResultadoTranscricao` e `ResultadoAnaliseTexto`. Os testes executam o modo
offline e simulam as respostas do SDK Speech, incluindo falhas. Isso não
comprova a execução dos serviços em nuvem.

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

O Isolation Forest fica fora desse laço por depender de um modelo treinado; na
prática, ele seria reavaliado em janelas periódicas, e não a cada leitura.

Os *achados* são os mesmos nos dois modos, mas os *alertas* não: no modo em
tempo real a execução emite 4 alertas em vez dos 6 do lote. A diferença é
esperada e vem da supressão temporal. Das seis tendências detectadas, três são
agrupadas porque caem na janela de 15 minutos de uma anterior (a alta de
temperatura das 17:01 dentro da janela aberta às 16:51, a queda de saturação
das 17:20 dentro da aberta às 17:07, e a queda de pressão diastólica das 17:44
dentro da aberta às 17:43); o mesmo vale para o limite crítico de frequência
respiratória das 18:17, suprimido pelo de temperatura das 18:03. É o
comportamento desejado: a equipe de plantão recebe quatro avisos ao longo de
duas horas, e não um por minuto. No modo em lote, em que o turno inteiro é
avaliado de uma vez, a consolidação por família dá o mesmo efeito.

A saída do turno completo do `PAC-0006`:

```text
10:30  (nota) Leitura isolada de frequencia cardiaca fora do padrao recente (185)
16:51  [LARANJA] Tendencia de alta sustentada de frequencia cardiaca: 93.5 contra 82.7
17:07  [LARANJA] Tendencia de alta sustentada de frequencia respiratoria: 19.2 contra 16.2
17:43  [LARANJA] Tendencia de queda sustentada de pressao sistolica: 112.8 contra 125.1
18:03  [VERMELHO] Temperatura acima do limite critico: 38.08 C (limite 38)
```

## 7. Testes

```powershell
python -m unittest discover -s tests -v
```

São 158 testes no total: os 74 das fases anteriores, que continuam passando, e
84 desta fase. Nenhum depende de rede ou de chave da Azure: o pipeline usa
o modo offline e os testes do SDK usam respostas simuladas. As amostras são
geradas em diretório temporário.

Os testes da Fase 4 cobrem, entre outros:

| Grupo | Verificações |
| --- | --- |
| Geração de amostras | reprodutibilidade por `seed`, estrutura das séries, modalidade ausente |
| Ângulos articulares | casos de referência conhecidos (0°, 45°, 90°, 140°), braço cruzando o corpo |
| Filtro de ruído de pose | pico isolado não vira desvio; keypoint de baixa confiança é ignorado |
| Atributos acústicos | voz cansada contra voz estável em cada atributo |
| Serviços Azure | contrato dos dois provedores, falha explícita sem credencial, busca sem acento |
| Máquina de estados das regras | confirmação, não repetição, artefato, histerese |
| Detectores de série | controle sem alerta, artefato não confirmado, tendência antes do limite, equivalência lote/tempo real |
| Prescrições | as três regras, e os casos em que não devem disparar |
| Fusão | exigência de duas modalidades, severidade nunca rebaixada, absorção de famílias, risco limitado a 100 |
| Alertas | consolidação, absorção, prioridade, supressão temporal, roteamento multimodal |
| Pipeline | caso completo, controle sem alerta, modalidade ausente, redução de notificações |

## 8. Limitações

São limitações do trabalho, não detalhes a ajustar:

1. **Os dados são sintéticos.** Os limiares foram calibrados contra amostras
   que nós mesmos geramos. Eles não têm validade clínica e serviriam apenas de
   ponto de partida num estudo com dados reais.
2. **O vídeo não é uma gravação de pessoa.** É um esqueleto articulado
   desenhado programaticamente. Os keypoints vêm de anotação, não de inferência
   sobre pixels. O backend MediaPipe está implementado para vídeo real, mas não
   foi exercitado com um, porque não temos vídeo clínico autorizado.
3. **O áudio não é fala humana.** É um sinal vozeado sintetizado com F0,
   jitter, taxa silábica e ruído respiratório controlados. Ele valida que o
   extrator de atributos reage ao que deveria; não substitui validação com voz
   de pacientes.
4. **A Azure não foi executada de fato.** A integração está implementada e
   testada quanto ao contrato, mas os números deste relatório vieram do
   provedor offline.
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

O protótipo implementa os módulos de vídeo, áudio, detecção de anomalias e
fusão multimodal. A execução validada neste relatório usa dados sintéticos,
poses anotadas e transcrições de referência. As integrações Azure e os modelos
de percepção ainda precisam ser executados com gravações autorizadas para
completar a demonstração exigida no enunciado.

Os alertas são gerados no terminal e em arquivos, sem envio a uma equipe real.
O modo incremental reproduz uma série gravada; não está conectado a um monitor
hospitalar. Também está pendente a gravação e publicação do vídeo de até
15 minutos no YouTube ou Vimeo.

O YOLOv8 padrão reconhece classes genéricas e não identifica drenos nem
distingue paciente e profissional. Os testes dessas regras usam anotações.
O áudio sintético não contém fala inteligível: seu texto de referência serve
ao teste offline, não à validação de reconhecimento de voz.

Dois resultados nos parecem os mais relevantes. O primeiro é a antecipação:
combinando detecção de tendência e análise multivariada, a deterioração foi
sinalizada 1 h 12 min antes de qualquer limiar fixo ser cruzado, que é
exatamente o tipo de margem que importa num monitoramento preventivo. O
segundo é a redução de ruído: a fusão multimodal transformou 36 achados
notificáveis em 6 alertas contextualizados e roteados, sem descartar informação —
sem isso, o sistema teria o defeito que mais inviabiliza painéis clínicos na
prática, que é alertar tanto que ninguém olha.

---

**Autores:** Antonio Miranda, Elaine, Marcos Mol, Lucas da Costa, Ricardo
Loureiro — AI for Devs (8IADT)
