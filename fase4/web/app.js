/* Interface da Fase 4: estado, roteador por hash e as quatro abas. */
(function (F4) {
  "use strict";

  const { el, limpar, chip, vazio, erro, rotulo, fmtInstante, fmtHora, fmtNum } = F4.render;
  const api = F4.api;
  const $app = document.getElementById("app");
  const $statusTexto = document.getElementById("runtime-status-text");
  const $status = document.getElementById("runtime-status");

  const estado = {
    status: null,
    pacientes: [],
    rota: { view: "painel", param: null },
    painel: {
      pacienteId: null, detalhe: null, resultado: null, infoVideo: null, erro: null, job: null, poll: null,
      processando: false, areaCritica: null,
      opcoes: { provedor_azure: "offline", backend_pose: "anotado", backend_detector: "anotado", arquivo_audio: "consulta.wav", gravar_video_anotado: true },
    },
    monitor: { pacienteId: null, conexao: null, grafico: null, msPorMinuto: 20, multivariado: true, totais: null },
    equipe: { conexao: null, alertas: new Map(), seqMax: 0 },
    dados: { enviadoSelecionado: null, roi: null, roiPaciente: null, roiCanvas: null, roiInfo: null, jobPoll: null, job: null, erro: null, confirmarExclusao: null },
  };

  /* --- utilidades ------------------------------------------------------- */

  function pacientePorId(id) {
    return estado.pacientes.find((p) => p.paciente_id === id) || null;
  }

  function selectPacientes(filtro, selecionado, aoMudar) {
    const grupos = { sinteticos: [], enviados: [], physionet: [], ucf101: [] };
    estado.pacientes.filter(filtro || (() => true)).forEach((p) => grupos[p.origem].push(p));
    const select = el("select", { onchange: () => aoMudar(select.value) });
    select.appendChild(el("option", { value: "", text: "Selecione um paciente" }));
    Object.keys(grupos).forEach((origem) => {
      if (!grupos[origem].length) return;
      const grupo = el("optgroup", { label: rotulo("origem", origem) });
      grupos[origem].forEach((p) => grupo.appendChild(el("option", {
        value: p.paciente_id, text: `${p.paciente_id}${p.resultado_disponivel ? " · processado" : ""}`,
      })));
      select.appendChild(grupo);
    });
    if (selecionado) select.value = selecionado;
    return select;
  }

  function listarEventosPlantados(plantados) {
    const eventos = plantados && plantados.sinais_vitais;
    if (!eventos) return [];
    const descrever = (ev) => (typeof ev === "string" ? ev : Object.entries(ev).map(([k, v]) => `${k}: ${Array.isArray(v) ? v.join(", ") : typeof v === "object" && v !== null ? JSON.stringify(v) : v}`).join(" · "));
    if (Array.isArray(eventos)) return eventos.map(descrever);
    return Object.entries(eventos).map(([nome, ev]) => `${nome} — ${descrever(ev)}`);
  }

  function campo(rotuloTexto, controle, ajuda) {
    return el("label", { class: "campo" }, [el("span", { text: rotuloTexto }), controle, ajuda ? el("small", { text: ajuda }) : null]);
  }

  function pararPolling(area) {
    if (area.poll) { clearInterval(area.poll); area.poll = null; }
    if (area.jobPoll) { clearInterval(area.jobPoll); area.jobPoll = null; }
  }

  async function carregarStatus() {
    try {
      estado.status = await api.get("/api/status");
      $status.classList.remove("error");
      $status.classList.add("ready");
      const b = estado.status.backends;
      $statusTexto.textContent = `Serviço ativo · ${b.mediapipe ? "MediaPipe" : "sem MediaPipe"} · ${b.yolov8 ? "YOLOv8" : "sem YOLOv8"} · ${b.whisper ? "whisper" : "sem whisper"}`;
    } catch (e) {
      $status.classList.add("error");
      $statusTexto.textContent = "Serviço indisponível";
    }
  }

  async function carregarPacientes() {
    estado.pacientes = await api.get("/api/pacientes");
  }

  /* --- roteador ---------------------------------------------------------- */

  function lerRota() {
    const m = location.hash.match(/^#\/(painel|monitor|equipe|dados)(?:\/([^/?#]+))?/);
    return m ? { view: m[1], param: m[2] ? decodeURIComponent(m[2]) : null } : { view: "painel", param: null };
  }

  function destruirView(view) {
    if (view === "painel") pararPolling(estado.painel);
    if (view === "monitor") pararMonitor();
    if (view === "equipe" && estado.equipe.conexao) { estado.equipe.conexao.fechar(); estado.equipe.conexao = null; }
    if (view === "dados") pararPolling(estado.dados);
  }

  async function aplicarRota() {
    const anterior = estado.rota.view;
    estado.rota = lerRota();
    if (anterior !== estado.rota.view) destruirView(anterior);
    document.querySelectorAll(".view-tab[data-view]").forEach((b) => b.classList.toggle("active", b.dataset.view === estado.rota.view));
    $app.dataset.view = estado.rota.view;
    limpar($app);
    try {
      await views[estado.rota.view]();
    } catch (e) {
      $app.appendChild(erro(`Falha ao montar a tela: ${e.message}`));
    }
  }

  document.querySelectorAll(".view-tab[data-view]").forEach((botao) => {
    botao.addEventListener("click", () => { location.hash = `#/${botao.dataset.view}`; });
  });
  window.addEventListener("hashchange", aplicarRota);

  /* --- Painel ------------------------------------------------------------ */

  async function selecionarPaciente(id) {
    const p = estado.painel;
    pararPolling(p);
    p.pacienteId = id || null;
    p.detalhe = null; p.resultado = null; p.infoVideo = null; p.erro = null; p.job = null;
    if (id) history.replaceState(null, "", `#/painel/${encodeURIComponent(id)}`);
    if (!id) { renderPainel(); return; }
    try {
      p.detalhe = await api.get(`/api/pacientes/${encodeURIComponent(id)}`);
      if (p.detalhe.origem === "ucf101") {
        p.opcoes.backend_pose = "mediapipe";
        p.opcoes.backend_detector = "yolov8";
      } else if (p.detalhe.origem === "sinteticos") {
        p.opcoes.backend_pose = "anotado";
        p.opcoes.backend_detector = "anotado";
      }
      if (p.detalhe.audios && p.detalhe.audios.length && !p.detalhe.audios.includes(p.opcoes.arquivo_audio)) p.opcoes.arquivo_audio = p.detalhe.audios[0];
      if (p.detalhe.resultado_disponivel) {
        p.resultado = await api.get(`/api/pacientes/${encodeURIComponent(id)}/resultado`);
        await carregarInfoVideo();
      }
    } catch (e) {
      p.erro = e.message;
    }
    renderPainel();
  }

  async function carregarInfoVideo() {
    const p = estado.painel;
    p.infoVideo = null;
    if (!p.resultado || !p.resultado.midias) return;
    const fonte = p.resultado.midias.video_anotado ? "anotado" : p.resultado.midias.video_original ? "original" : null;
    if (!fonte) return;
    try {
      const transcodificar = estado.status && estado.status.backends.ffmpeg ? 1 : 0;
      p.infoVideo = await api.get(`/api/pacientes/${encodeURIComponent(p.pacienteId)}/video/info?fonte=${fonte}&transcodificar=${transcodificar}`);
    } catch (_e) { p.infoVideo = null; }
  }

  async function processar() {
    const p = estado.painel;
    if (!p.pacienteId || p.processando) return;
    p.erro = null; p.processando = true; p.job = null;
    renderPainel();
    const corpo = Object.assign({}, p.opcoes, { area_critica: p.areaCritica || null, modo: "auto" });
    try {
      const resposta = await api.post(`/api/pacientes/${encodeURIComponent(p.pacienteId)}/processar`, corpo);
      if (resposta.modo === "job") {
        p.job = resposta.job;
        iniciarPollingJob();
      } else {
        p.resultado = resposta.resultado;
        p.processando = false;
        await carregarInfoVideo();
        await carregarPacientes();
      }
    } catch (e) {
      p.erro = e.message;
      p.processando = false;
    }
    renderPainel();
  }

  function iniciarPollingJob() {
    const p = estado.painel;
    pararPolling(p);
    p.poll = setInterval(async () => {
      try {
        const { job } = await api.get("/api/jobs/atual");
        p.job = job;
        if (!job || !["aguardando", "executando", "cancelando"].includes(job.estado)) {
          pararPolling(p);
          p.processando = false;
          if (job && job.estado === "concluido" && job.resultado && job.resultado.paciente_id === p.pacienteId) {
            p.resultado = job.resultado;
            await carregarInfoVideo();
            await carregarPacientes();
          } else if (job && job.erro) {
            p.erro = job.erro;
          }
        }
        renderPainel();
      } catch (e) {
        p.erro = e.message; pararPolling(p); p.processando = false; renderPainel();
      }
    }, 1000);
  }

  async function cancelarJob() {
    try { await api.post("/api/jobs/cancelar"); } catch (e) { estado.painel.erro = e.message; renderPainel(); }
  }

  function formularioPainel() {
    const p = estado.painel;
    const status = estado.status || { servicos: {}, backends: {} };
    const backends = status.backends || {};
    const servicos = status.servicos || {};
    const azureOk = servicos.speech_to_text && servicos.speech_to_text.provedor_efetivo === "azure";
    const provedor = el("select", { onchange: (e) => { p.opcoes.provedor_azure = e.target.value; renderPainel(); } }, [
      el("option", { value: "offline", text: "offline — substituto local determinístico" }),
      el("option", { value: "auto", text: `auto — Azure se houver credencial (${azureOk ? "disponível" : "sem credencial → offline"})` }),
      el("option", { value: "local", text: `local — faster-whisper nesta máquina${backends.whisper ? "" : " (não instalado)"}`, disabled: !backends.whisper }),
      el("option", { value: "azure", text: `azure — exige credencial${azureOk ? "" : " (indisponível)"}`, disabled: !azureOk }),
    ]);
    provedor.value = p.opcoes.provedor_azure;
    const pose = el("select", { onchange: (e) => { p.opcoes.backend_pose = e.target.value; renderPainel(); } }, [
      el("option", { value: "anotado", text: "anotado — keypoints da amostra" }),
      el("option", { value: "mediapipe", text: `mediapipe — estimador real${backends.mediapipe ? "" : " (não instalado)"}`, disabled: !backends.mediapipe }),
    ]);
    pose.value = p.opcoes.backend_pose;
    const detector = el("select", { onchange: (e) => { p.opcoes.backend_detector = e.target.value; renderPainel(); } }, [
      el("option", { value: "anotado", text: "anotado — caixas da amostra" }),
      el("option", { value: "yolov8", text: `yolov8 — detector real${backends.yolov8 ? "" : " (não instalado)"}`, disabled: !backends.yolov8 }),
    ]);
    detector.value = p.opcoes.backend_detector;
    const audios = (p.detalhe && p.detalhe.audios) || [];
    const audio = el("select", { onchange: (e) => { p.opcoes.arquivo_audio = e.target.value; }, disabled: !audios.length }, audios.length
      ? audios.map((a) => el("option", { value: a, text: a === "consulta_fala.wav" ? `${a} — fala pt-BR (TTS)` : a }))
      : [el("option", { value: "", text: "sem áudio" })]);
    if (audios.length) audio.value = p.opcoes.arquivo_audio;
    const area = el("input", { type: "text", placeholder: "x1,y1,x2,y2 (pixels)", value: p.areaCritica ? p.areaCritica.join(",") : "" });
    area.addEventListener("change", () => {
      const partes = area.value.split(",").map((v) => Number(v.trim()));
      p.areaCritica = area.value.trim() && partes.length === 4 && partes.every((n) => Number.isFinite(n)) ? partes : null;
      if (area.value.trim() && !p.areaCritica) { p.erro = "Área crítica: informe quatro números x1,y1,x2,y2."; renderPainel(); }
    });
    const gravarVideo = el("input", { type: "checkbox", checked: p.opcoes.gravar_video_anotado, onchange: (e) => { p.opcoes.gravar_video_anotado = e.target.checked; } });
    const temVideo = Boolean(p.detalhe && p.detalhe.arquivos && (p.detalhe.arquivos.video || p.detalhe.arquivos.keypoints));
    const precisaJob = p.opcoes.backend_pose !== "anotado" || p.opcoes.backend_detector !== "anotado" || p.opcoes.provedor_azure === "local" || (p.detalhe && p.detalhe.origem === "physionet");

    return el("form", { class: "formulario", onsubmit: (e) => { e.preventDefault(); processar(); } }, [
      campo("Paciente", selectPacientes(null, p.pacienteId, selecionarPaciente)),
      p.detalhe ? el("p", { class: "nota" }, [
        chip(rotulo("origem", p.detalhe.origem), "neutro"), " ",
        (p.detalhe.modalidades || []).map((m) => rotulo("modalidade", m)).join(" · ") || "sem modalidades",
      ]) : null,
      campo("Transcrição e linguagem", provedor, "Azure Speech/Text Analytics, substituto offline ou whisper local."),
      campo("Áudio da consulta", audio),
      campo("Estimador de pose", pose, temVideo ? null : "Este paciente não tem vídeo."),
      campo("Detector de objetos", detector, p.opcoes.backend_detector === "yolov8" ? "Conta pessoas; não identifica quem é profissional. Sem detecção, o acompanhamento fica não avaliado." : null),
      p.detalhe && p.detalhe.origem === "ucf101" ? el("p", { class: "nota", text: "UCF101: ações humanas para avaliação de postura e detecção. Estes clipes não são sessões clínicas; os limites de pós-operatório não são aplicados." }) : null,
      campo("Área crítica (ROI)", el("div", { class: "linha" }, [
        area,
        el("button", { type: "button", class: "secondary-button", text: "Desenhar", onclick: () => { estado.dados.roiPaciente = p.pacienteId; location.hash = "#/dados"; } }),
      ]), "Necessária com YOLOv8; com o backend anotado substitui a caixa do dreno."),
      el("label", { class: "check-field" }, [gravarVideo, el("span", { text: "Gravar vídeo anotado (mais lento)" })]),
      el("div", { class: "acoes" }, [
        el("button", { type: "submit", class: "primary-button", disabled: !p.pacienteId || p.processando, text: p.processando ? "Processando…" : precisaJob ? "Processar em segundo plano" : "Processar" }),
        p.resultado ? el("button", { type: "button", class: "secondary-button", text: "Ver no monitor", onclick: () => { location.hash = `#/monitor/${encodeURIComponent(p.pacienteId)}`; } }) : null,
      ]),
      p.erro ? erro(p.erro) : null,
    ]);
  }

  function renderPainel() {
    if (estado.rota.view !== "painel") return;
    const p = estado.painel;
    limpar($app);
    const aside = el("aside", { class: "coluna coluna-lateral" }, [
      el("div", { class: "panel-heading" }, [el("p", { class: "overline", text: "Monitoramento multimodal" }), el("h1", { text: "Painel do paciente" })]),
      formularioPainel(),
      p.detalhe && p.detalhe.eventos_plantados ? el("details", { class: "gabarito" }, [
        el("summary", { text: "Eventos plantados na amostra (gabarito)" }),
        el("ul", { class: "lista-compacta" }, listarEventosPlantados(p.detalhe.eventos_plantados).map((ev) => el("li", { text: ev }))),
      ]) : null,
    ]);
    const centro = el("section", { class: "coluna coluna-central" });
    const direita = el("aside", { class: "coluna coluna-direita" });

    if (p.job) centro.appendChild(F4.render.jobStatus(p.job, ["aguardando", "executando", "cancelando"].includes(p.job.estado) ? cancelarJob : null));
    if (!p.pacienteId) {
      centro.appendChild(el("div", { class: "boas-vindas" }, [
        el("h3", { text: "Escolha um paciente para começar" }),
        el("p", { text: "O pipeline analisa vídeo da fisioterapia, áudio da consulta e séries de sinais vitais, prescrições e movimentação; funde os achados em quadros clínicos e emite alertas para a equipe." }),
      ]));
    } else if (!p.resultado) {
      centro.appendChild(el("div", { class: "boas-vindas" }, [
        el("h3", { text: `${p.pacienteId} ainda não foi processado` }),
        el("p", { text: "Ajuste as opções à esquerda e clique em Processar. Com os backends anotados e o provedor offline leva alguns segundos." }),
      ]));
    } else {
      const r = p.resultado;
      centro.appendChild(F4.render.cartaoRisco(r));
      ((r.video && r.video.avisos) || []).forEach((aviso) => centro.appendChild(el("p", { class: "nota", text: aviso })));
      centro.appendChild(F4.render.cartoesQuadros(r));
      centro.appendChild(F4.render.cartoesAlertas(r.alertas || []));
      centro.appendChild(F4.render.tabelaAchados(r.achados || []));
      centro.appendChild(F4.render.painelRelatorio(r.relatorios || [], (nome) => api.get(`/api/pacientes/${encodeURIComponent(p.pacienteId)}/relatorios/${nome}`)));
      direita.appendChild(el("div", { class: "panel-heading" }, [el("p", { class: "overline", text: "Mídia" }), el("h2", { text: "Consulta e sessão" })]));
      direita.appendChild(F4.render.midia(r, p.infoVideo));
      direita.appendChild(el("p", { class: "nota", text: `Executado em ${fmtHora(r.executado_em)} · provedor ${(r.servicos_azure || {}).speech_to_text || "—"}` }));
    }
    $app.appendChild(aside);
    $app.appendChild(centro);
    $app.appendChild(direita);
  }

  /* --- Monitor ao vivo -------------------------------------------------- */

  function pararMonitor() {
    const m = estado.monitor;
    if (m.conexao) { m.conexao.fechar(); m.conexao = null; }
    if (m.grafico) { m.grafico.destruir(); m.grafico = null; }
  }

  function renderMonitor() {
    const m = estado.monitor;
    if (estado.rota.param && !m.pacienteId) m.pacienteId = estado.rota.param;
    limpar($app);
    const canvas = el("canvas", { class: "grafico-canvas" });
    const legenda = el("p", { class: "nota legenda", text: "Passe o mouse sobre o gráfico para ler os valores." });
    const timeline = el("ul", { class: "timeline" });
    const resumo = el("p", { class: "nota", text: "Aguardando início." });
    const botaoIniciar = el("button", { type: "button", class: "primary-button", text: "Iniciar" });
    const botaoParar = el("button", { type: "button", class: "secondary-button", text: "Parar", disabled: true });
    const velocidade = el("select", { onchange: (e) => { m.msPorMinuto = Number(e.target.value); } }, [
      el("option", { value: "500", text: "lento — 0,5 s por minuto" }),
      el("option", { value: "100", text: "normal — 0,1 s por minuto" }),
      el("option", { value: "20", text: "rápido — 20 ms por minuto" }),
      el("option", { value: "0", text: "máximo" }),
    ]);
    velocidade.value = String(m.msPorMinuto);
    const multivariado = el("input", { type: "checkbox", checked: m.multivariado, onchange: (e) => { m.multivariado = e.target.checked; } });

    function adicionarTimeline(classe, hora, texto, extra) {
      const item = el("li", { class: classe }, [el("code", { text: hora }), " ", texto, extra ? el("small", { text: ` ${extra}` }) : null]);
      timeline.insertBefore(item, timeline.firstChild);
      while (timeline.children.length > 80) timeline.removeChild(timeline.lastChild);
    }

    function iniciar() {
      if (!m.pacienteId) return;
      pararMonitor();
      limpar(timeline);
      m.totais = { leituras: 0, achados: 0, alertas: 0 };
      botaoIniciar.disabled = true; botaoParar.disabled = false;
      resumo.textContent = "Conectando…";
      const url = `/api/pacientes/${encodeURIComponent(m.pacienteId)}/tempo-real?ms_por_minuto=${m.msPorMinuto}&multivariado=${m.multivariado ? 1 : 0}`;
      m.conexao = api.sse(url, {
        inicio: (dados) => {
          m.grafico = new F4.GraficoSinais(canvas, { canais: dados.canais, faixas: dados.faixas, totalMinutos: dados.total_minutos, inicio: dados.inicio, legenda });
          resumo.textContent = `${dados.total_minutos} leituras de ${fmtInstante(dados.inicio)} a ${fmtInstante(dados.fim)}${dados.multivariado ? " · Isolation Forest online ativo" : ""}`;
        },
        leitura: (dados) => {
          if (!m.grafico) return;
          m.grafico.adicionar(dados);
          m.totais.leituras += 1;
          (dados.achados || []).forEach((a) => {
            m.totais.achados += 1;
            if (a.severidade === "informativo") adicionarTimeline("nota", fmtInstante(a.instante || dados.instante), `nota: ${a.descricao}`);
            else if (a.tipo === "anomalia_multivariada") adicionarTimeline("multivariado", fmtInstante(dados.instante), `Isolation Forest: ${a.descricao}`, `episódio desde ${fmtInstante(a.evidencias && a.evidencias.inicio)}`);
          });
          (dados.alertas || []).forEach((al) => {
            m.totais.alertas += 1;
            adicionarTimeline(`alerta prio-${al.prioridade}`, fmtInstante(dados.instante), `[${rotulo("prioridade", al.prioridade).toUpperCase()}] ${al.titulo}`, `→ ${al.destino} · via ${(al.canais_entrega || []).map((c) => rotulo("canalEntrega", c)).join(", ")}`);
          });
          if (m.totais.leituras % 30 === 0) resumo.textContent = `${m.totais.leituras} leituras · ${m.totais.achados} achados · ${m.totais.alertas} alertas`;
        },
        fim: (dados) => {
          resumo.textContent = `Turno encerrado: ${dados.minutos} leituras, ${dados.total_alertas} alertas emitidos${dados.entregas_falhas ? `, ${dados.entregas_falhas} entregas falharam` : ""}.`;
          if (m.conexao) { m.conexao.fechar(); m.conexao = null; }
          botaoIniciar.disabled = false; botaoParar.disabled = true;
        },
        erro: (_evento, readyState) => {
          if (readyState === EventSource.CLOSED) return;
          resumo.textContent = "Conexão com o servidor interrompida.";
          botaoIniciar.disabled = false; botaoParar.disabled = true;
        },
      });
    }
    function parar() {
      if (m.conexao) { m.conexao.fechar(); m.conexao = null; }
      botaoIniciar.disabled = false; botaoParar.disabled = true;
      resumo.textContent = `Interrompido com ${m.totais ? m.totais.leituras : 0} leituras.`;
    }
    botaoIniciar.addEventListener("click", iniciar);
    botaoParar.addEventListener("click", parar);

    $app.appendChild(el("section", { class: "pagina-larga" }, [
      el("div", { class: "cabecalho-pagina" }, [
        el("div", {}, [el("p", { class: "overline", text: "Monitor de leito" }), el("h1", { text: "Sinais vitais em tempo real" }),
          el("p", { class: "nota", text: "A série é percorrida minuto a minuto, como se as leituras chegassem do monitor. Os detectores decidem só com o passado; o Isolation Forest confirma o episódio quando ele completa 15 minutos." })]),
      ]),
      el("div", { class: "controles" }, [
        campo("Paciente", selectPacientes((p) => p.arquivos && p.arquivos.sinais_vitais, m.pacienteId, (id) => { m.pacienteId = id || null; history.replaceState(null, "", `#/monitor/${encodeURIComponent(id || "")}`); })),
        campo("Velocidade", velocidade),
        el("label", { class: "check-field" }, [multivariado, el("span", { text: "Incluir Isolation Forest" })]),
        el("div", { class: "acoes" }, [botaoIniciar, botaoParar]),
      ]),
      el("div", { class: "monitor-grid" }, [
        el("div", { class: "bloco grafico-wrap" }, [canvas, legenda, resumo]),
        el("div", { class: "bloco" }, [el("div", { class: "bloco-titulo" }, [el("h3", { text: "Linha do tempo" })]), timeline]),
      ]),
    ]));
  }

  /* --- Equipe ------------------------------------------------------------ */

  function renderEquipe() {
    const q = estado.equipe;
    limpar($app);
    const feed = el("div", { class: "equipe-feed" });
    const contador = el("span", { class: "chip neutro", text: "0" });
    const estadoConexao = el("span", { class: "nota", text: "conectando…" });
    const webhookUrl = `${location.origin}/api/webhook/alertas`;

    function redesenhar() {
      limpar(feed);
      const alertas = [...q.alertas.values()].sort((a, b) => b.seq - a.seq);
      contador.textContent = String(alertas.length);
      if (!alertas.length) { feed.appendChild(vazio("Nenhum alerta recebido ainda. Processe um paciente no Painel ou rode o Monitor ao vivo.")); return; }
      const grupos = new Map();
      alertas.forEach((a) => {
        const chave = (a.destino || "Equipe médica").split(";")[0].trim();
        if (!grupos.has(chave)) grupos.set(chave, []);
        grupos.get(chave).push(a);
      });
      grupos.forEach((lista, destino) => {
        feed.appendChild(el("section", { class: "bloco" }, [
          el("div", { class: "bloco-titulo" }, [el("h3", { text: destino }), chip(String(lista.length), "coral")]),
          el("div", { class: "cards" }, lista.map((a) => F4.render.cartaoAlerta(a, { hora: fmtHora(a.recebido_em), origem: a.origem }))),
        ]));
      });
    }
    function receber(alerta) {
      q.alertas.set(alerta.seq, alerta);
      q.seqMax = Math.max(q.seqMax, alerta.seq);
      while (q.alertas.size > 200) q.alertas.delete(Math.min(...q.alertas.keys()));
      redesenhar();
    }
    async function conectar() {
      try {
        const historico = await api.get(`/api/alertas?desde=${q.seqMax}`);
        historico.alertas.forEach((a) => { q.alertas.set(a.seq, a); q.seqMax = Math.max(q.seqMax, a.seq); });
        redesenhar();
      } catch (e) { feed.appendChild(erro(e.message)); }
      if (q.conexao) q.conexao.fechar();
      q.conexao = api.sse(`/api/alertas/stream?desde=${q.seqMax}`, {
        alerta: (dados) => { estadoConexao.textContent = "ao vivo"; receber(dados); },
        erro: (_e, readyState) => { estadoConexao.textContent = readyState === EventSource.CLOSED ? "desconectado" : "reconectando…"; },
      });
      q.conexao.fonte.onopen = () => { estadoConexao.textContent = "ao vivo"; };
    }

    $app.appendChild(el("section", { class: "pagina-larga" }, [
      el("div", { class: "cabecalho-pagina" }, [
        el("div", {}, [el("p", { class: "overline", text: "Painel da equipe médica" }), el("h1", {}, ["Alertas entregues ", contador]),
          el("p", { class: "nota" }, ["Esta aba é o receptor: recebe os alertas do Painel, do Monitor ao vivo e de qualquer processo que aponte o webhook para ", el("code", { text: webhookUrl }), ". Estado: ", estadoConexao])]),
        el("div", { class: "bloco dica" }, [
          el("p", { class: "overline", text: "Da linha de comando" }),
          el("pre", { class: "codigo", text: `$env:FASE4_WEBHOOK_URL = "${webhookUrl}"\npython -m fase4.cli_demo --paciente-id PAC-0006 --tempo-real --atraso 0.05` }),
        ]),
      ]),
      feed,
    ]));
    conectar();
  }

  /* --- Dados: enviados, ROI, PhysioNet ---------------------------------- */

  async function renderDados() {
    const d = estado.dados;
    limpar($app);
    const status = estado.status || { uploads: { arquivos: [] }, backends: {} };
    const pagina = el("section", { class: "pagina-larga" });
    pagina.appendChild(el("div", { class: "cabecalho-pagina" }, [
      el("div", {}, [el("p", { class: "overline", text: "Dados" }), el("h1", { text: "Gravações próprias, área crítica e PhysioNet" })]),
    ]));
    const grade = el("div", { class: "dados-grid" });
    pagina.appendChild(grade);
    $app.appendChild(pagina);

    /* Enviados */
    const painelEnviados = el("section", { class: "bloco" });
    grade.appendChild(painelEnviados);
    function redesenharEnviados() {
      limpar(painelEnviados);
      painelEnviados.appendChild(el("div", { class: "bloco-titulo" }, [el("h3", { text: "Pacientes enviados" })]));
      const inputId = el("input", { type: "text", placeholder: "PAC-REAL-01", pattern: "^[A-Za-z0-9][A-Za-z0-9_-]{1,39}$" });
      painelEnviados.appendChild(el("form", { class: "linha", onsubmit: async (e) => {
        e.preventDefault();
        try {
          await api.post("/api/pacientes/enviados", { paciente_id: inputId.value.trim() });
          d.enviadoSelecionado = inputId.value.trim();
          await carregarPacientes();
          redesenharEnviados();
        } catch (err) { d.erro = err.message; redesenharEnviados(); }
      } }, [inputId, el("button", { type: "submit", class: "secondary-button", text: "Criar paciente" })]));
      if (d.erro) { painelEnviados.appendChild(erro(d.erro)); d.erro = null; }
      const enviados = estado.pacientes.filter((p) => p.origem === "enviados");
      if (!enviados.length) painelEnviados.appendChild(vazio("Nenhum paciente enviado. Crie um id e envie os arquivos com os nomes padrão do projeto."));
      enviados.forEach((p) => {
        const aberto = d.enviadoSelecionado === p.paciente_id;
        const card = el("article", { class: `enviado-card ${aberto ? "aberto" : ""}` }, [
          el("header", {}, [
            el("button", { type: "button", class: "link-button", text: p.paciente_id, onclick: () => { d.enviadoSelecionado = aberto ? null : p.paciente_id; redesenharEnviados(); } }),
            el("span", { class: "nota", text: (p.modalidades || []).map((m) => rotulo("modalidade", m)).join(" · ") || "sem arquivos" }),
            el("div", { class: "acoes" }, [
              el("button", { type: "button", class: "secondary-button", text: "Abrir no Painel", onclick: () => { location.hash = `#/painel/${encodeURIComponent(p.paciente_id)}`; } }),
              el("button", { type: "button", class: "danger-button", text: d.confirmarExclusao === p.paciente_id ? "Confirmar exclusão" : "Excluir", onclick: async () => {
                if (d.confirmarExclusao !== p.paciente_id) { d.confirmarExclusao = p.paciente_id; redesenharEnviados(); return; }
                try { await api.del(`/api/pacientes/enviados/${encodeURIComponent(p.paciente_id)}`); d.confirmarExclusao = null; await carregarPacientes(); redesenharEnviados(); }
                catch (err) { d.erro = err.message; redesenharEnviados(); }
              } }),
            ]),
          ]),
        ]);
        if (aberto) {
          const lista = el("ul", { class: "upload-lista" });
          (status.uploads.arquivos || []).forEach((nome) => {
            const presente = Object.values(p.arquivos || {}).includes(nome) || (p.audios || []).includes(nome) || Object.keys(p.arquivos || {}).some((k) => (p.arquivos[k] === nome));
            const input = el("input", { type: "file" });
            const barra = el("div", { class: "progress-track" }, [el("span")]);
            const estadoTexto = el("small", { text: presente ? "enviado" : "ausente" });
            const botao = el("button", { type: "button", class: "secondary-button", text: "Enviar", onclick: async () => {
              const arquivo = input.files && input.files[0];
              if (!arquivo) { estadoTexto.textContent = "escolha um arquivo"; return; }
              botao.disabled = true; estadoTexto.textContent = "enviando…";
              try {
                const resposta = await api.putArquivo(`/api/pacientes/enviados/${encodeURIComponent(p.paciente_id)}/arquivos/${nome}`, arquivo, (fracao) => { barra.firstChild.style.width = `${Math.round(fracao * 100)}%`; });
                estadoTexto.textContent = `ok · ${JSON.stringify(resposta.validacao)}`;
                await carregarPacientes();
              } catch (err) { estadoTexto.textContent = `erro: ${err.message}`; }
              botao.disabled = false;
            } });
            lista.appendChild(el("li", {}, [el("code", { text: nome }), input, botao, barra, estadoTexto]));
          });
          card.appendChild(lista);
        }
        painelEnviados.appendChild(card);
      });
    }
    redesenharEnviados();

    /* ROI */
    const painelRoi = el("section", { class: "bloco" });
    grade.appendChild(painelRoi);
    painelRoi.appendChild(el("div", { class: "bloco-titulo" }, [el("h3", { text: "Área crítica sobre o vídeo" })]));
    const comVideo = (p) => p.arquivos && (p.arquivos.video);
    const saidaRoi = el("code", { class: "roi-saida", text: d.roi ? d.roi.join(",") : "arraste sobre o quadro para marcar" });
    const tempo = el("input", { type: "range", min: "0", max: "10", step: "0.1", value: "0" });
    const rotuloTempo = el("span", { class: "nota", text: "0,0 s" });
    const areaCanvas = el("div", { class: "roi-area" });
    let roiCanvas = null;
    async function carregarQuadro() {
      if (!d.roiPaciente) return;
      try {
        if (!d.roiInfo || d.roiInfo.paciente !== d.roiPaciente) {
          d.roiInfo = await api.get(`/api/pacientes/${encodeURIComponent(d.roiPaciente)}/video/info?fonte=original`);
          d.roiInfo.paciente = d.roiPaciente;
          tempo.max = String(Math.max(0, d.roiInfo.duracao_s - 1 / (d.roiInfo.fps || 30)));
        }
        const t = Number(tempo.value);
        rotuloTempo.textContent = `${fmtNum(t, 1)} s · ${d.roiInfo.largura}×${d.roiInfo.altura} px`;
        if (!roiCanvas) {
          limpar(areaCanvas);
          roiCanvas = new F4.RoiCanvas(areaCanvas, (roi) => {
            d.roi = roi;
            if (!roi) { saidaRoi.textContent = "arraste sobre o quadro para marcar"; return; }
            const fracao = ((roi[2] - roi[0]) * (roi[3] - roi[1])) / (d.roiInfo.largura * d.roiInfo.altura);
            saidaRoi.textContent = `${roi.join(",")} (${(fracao * 100).toFixed(0)}% do quadro)`;
            saidaRoi.classList.toggle("alerta-roi", fracao > 0.25);
          });
          if (d.roi) roiCanvas.definir(d.roi);
        }
        roiCanvas.carregar(`/api/pacientes/${encodeURIComponent(d.roiPaciente)}/quadro.jpg?fonte=original&t=${t.toFixed(2)}`, d.roiInfo.largura, d.roiInfo.altura);
      } catch (err) { limpar(areaCanvas).appendChild(erro(err.message)); }
    }
    tempo.addEventListener("input", carregarQuadro);
    painelRoi.appendChild(el("div", { class: "controles" }, [
      campo("Vídeo", selectPacientes(comVideo, d.roiPaciente, (id) => { d.roiPaciente = id || null; d.roiInfo = null; roiCanvas = null; carregarQuadro(); })),
      campo("Instante", el("div", { class: "linha" }, [tempo, rotuloTempo])),
    ]));
    painelRoi.appendChild(areaCanvas);
    painelRoi.appendChild(el("div", { class: "linha" }, [
      el("span", { class: "nota", text: "ROI: " }), saidaRoi,
      el("button", { type: "button", class: "primary-button", text: "Usar no Painel", onclick: () => {
        if (!d.roi) return;
        estado.painel.areaCritica = d.roi.slice();
        estado.painel.pacienteId = d.roiPaciente;
        location.hash = `#/painel/${encodeURIComponent(d.roiPaciente)}`;
      } }),
      el("button", { type: "button", class: "secondary-button", text: "Limpar", onclick: () => { d.roi = null; if (roiCanvas) roiCanvas.definir(null); saidaRoi.textContent = "arraste sobre o quadro para marcar"; } }),
    ]));
    painelRoi.appendChild(el("p", { class: "nota", text: "Marque só a região do dreno: uma ROI grande demais (acima de 25% do quadro) é recusada, porque a mão estaria sempre dentro dela. A ROI substitui a caixa do dreno que o YOLOv8 genérico não reconhece; a mão do paciente passa a ser o keypoint do punho. Dica para a amostra sintética: 250,178,296,220." }));
    if (d.roiPaciente && !comVideo(pacientePorId(d.roiPaciente) || {})) d.roiPaciente = null;
    if (d.roiPaciente) carregarQuadro();

    /* PhysioNet */
    const painelPn = el("section", { class: "bloco" });
    grade.appendChild(painelPn);
    const registro = el("input", { type: "text", value: "p04/p044083/p044083-2112-05-04-19-50n" });
    const minutos = el("input", { type: "number", min: "0", max: "3000", value: "720" });
    const areaJob = el("div");
    function redesenharJob() {
      limpar(areaJob);
      if (d.job) areaJob.appendChild(F4.render.jobStatus(d.job, ["aguardando", "executando", "cancelando"].includes(d.job.estado) ? async () => { try { await api.post("/api/jobs/cancelar"); } catch (_e) { /* ignora */ } } : null));
    }
    function acompanharJob() {
      pararPolling(d);
      d.jobPoll = setInterval(async () => {
        try {
          const { job } = await api.get("/api/jobs/atual");
          d.job = job; redesenharJob();
          if (!job || !["aguardando", "executando", "cancelando"].includes(job.estado)) { pararPolling(d); await carregarPacientes(); redesenharPn(); }
        } catch (_e) { pararPolling(d); }
      }, 1000);
    }
    const listaPn = el("ul", { class: "lista-compacta" });
    function redesenharPn() {
      limpar(listaPn);
      const registros = estado.pacientes.filter((p) => p.origem === "physionet");
      if (!registros.length) listaPn.appendChild(el("li", { class: "vazio", text: "Nenhum registro baixado." }));
      registros.forEach((p) => listaPn.appendChild(el("li", {}, [
        el("code", { text: p.paciente_id }), p.resultado_disponivel ? " · processado " : " ",
        el("button", { type: "button", class: "link-button", text: "abrir no Painel", onclick: () => { location.hash = `#/painel/${encodeURIComponent(p.paciente_id)}`; } }),
      ])));
    }
    painelPn.appendChild(el("div", { class: "bloco-titulo" }, [el("h3", { text: "Registro real do PhysioNet (MIMIC-III)" }), chip(status.backends.wfdb ? "wfdb instalado" : "wfdb ausente", status.backends.wfdb ? "neutro" : "coral")]));
    painelPn.appendChild(el("form", { class: "controles", onsubmit: async (e) => {
      e.preventDefault();
      try {
        const resposta = await api.post("/api/physionet/baixar", { registro: registro.value.trim(), minutos: Number(minutos.value) });
        d.job = resposta.job; redesenharJob(); acompanharJob();
      } catch (err) { areaJob.appendChild(erro(err.message)); }
    } }, [
      campo("Registro (pXX/pXXXXXX/nome)", registro),
      campo("Minutos (0 = tudo)", minutos),
      el("div", { class: "acoes" }, [el("button", { type: "submit", class: "primary-button", text: "Baixar e processar", disabled: !status.backends.wfdb })]),
    ]));
    painelPn.appendChild(areaJob);
    painelPn.appendChild(el("p", { class: "nota", text: "Acesso aberto (ODbL). Baixa o registro, converte para mediana por minuto no esquema do projeto e roda os mesmos detectores. Sem rótulo de desfecho: valida comportamento, não acurácia." }));
    painelPn.appendChild(listaPn);
    redesenharPn();
    if (d.job && ["aguardando", "executando", "cancelando"].includes(d.job.estado)) { redesenharJob(); acompanharJob(); }
  }

  const views = {
    painel: async () => {
      if (estado.rota.param && estado.rota.param !== estado.painel.pacienteId) { await selecionarPaciente(estado.rota.param); return; }
      renderPainel();
      if (estado.painel.job && ["aguardando", "executando", "cancelando"].includes(estado.painel.job.estado)) iniciarPollingJob();
    },
    monitor: async () => renderMonitor(),
    equipe: async () => renderEquipe(),
    dados: async () => renderDados(),
  };

  async function iniciar() {
    await carregarStatus();
    try { await carregarPacientes(); } catch (e) { $app.appendChild(erro(e.message)); }
    await aplicarRota();
    setInterval(carregarStatus, 30000);
  }

  iniciar();
})(window.F4);
