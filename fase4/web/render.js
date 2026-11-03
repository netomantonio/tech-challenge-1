/* Construtores de DOM da interface da Fase 4. Nada de innerHTML, exceto o
   HTML dos relatorios, que vem do servidor ja escapado e convertido. */
window.F4 = window.F4 || {};

(function (F4) {
  "use strict";

  function el(tag, atributos, filhos) {
    const node = document.createElement(tag);
    if (atributos) {
      Object.keys(atributos).forEach((chave) => {
        const valor = atributos[chave];
        if (valor === null || valor === undefined || valor === false) return;
        if (chave === "class") node.className = valor;
        else if (chave === "text") node.textContent = valor;
        else if (chave === "dataset") Object.assign(node.dataset, valor);
        else if (chave.startsWith("on") && typeof valor === "function") node.addEventListener(chave.slice(2), valor);
        else if (chave === "checked" || chave === "disabled" || chave === "selected" || chave === "hidden") node[chave] = Boolean(valor);
        else if (chave === "value") node.value = valor;
        else node.setAttribute(chave, valor);
      });
    }
    (Array.isArray(filhos) ? filhos : filhos !== undefined ? [filhos] : []).forEach((filho) => {
      if (filho === null || filho === undefined || filho === false) return;
      node.appendChild(typeof filho === "string" || typeof filho === "number" ? document.createTextNode(String(filho)) : filho);
    });
    return node;
  }

  function limpar(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
    return node;
  }

  const ROTULOS = {
    modalidade: {
      video: "Vídeo", audio: "Áudio", sinais_vitais: "Sinais vitais", prescricao: "Prescrição",
      prescricoes: "Prescrições", movimentacao: "Movimentação", multimodal: "Multimodal",
    },
    severidade: { informativo: "informativo", atencao: "atenção", critico: "crítico" },
    prioridade: { vermelho: "Vermelho", laranja: "Laranja", amarelo: "Amarelo" },
    canal: {
      frequencia_cardiaca: "FC", pressao_sistolica: "PAS", pressao_diastolica: "PAD",
      saturacao_oxigenio: "SpO₂", temperatura: "Temp", frequencia_respiratoria: "FR",
    },
    canalLongo: {
      frequencia_cardiaca: "Frequência cardíaca", pressao_sistolica: "Pressão sistólica",
      pressao_diastolica: "Pressão diastólica", saturacao_oxigenio: "Saturação de O₂",
      temperatura: "Temperatura", frequencia_respiratoria: "Frequência respiratória",
    },
    origem: { sinteticos: "Sintético", enviados: "Enviado", physionet: "PhysioNet", ucf101: "UCF101 · ações humanas" },
    canalEntrega: { fila_jsonl: "fila JSONL", webhook: "webhook", painel_web: "painel web" },
  };

  function rotulo(grupo, chave) {
    return (ROTULOS[grupo] && ROTULOS[grupo][chave]) || String(chave || "").replace(/_/g, " ");
  }

  function fmtInstante(valor) {
    if (!valor) return "—";
    const texto = String(valor);
    if (/^\d{4}-\d{2}-\d{2}T/.test(texto)) return texto.slice(11, 16);
    return texto;
  }

  /* Janela de um achado de video ("10.00-13.97") ou de serie ("inicio"/"fim" ISO). */
  function fmtJanela(evidencias) {
    if (!evidencias) return null;
    const mmss = (s) => `${Math.floor(s / 60).toString().padStart(2, "0")}:${(s % 60).toFixed(1).padStart(4, "0")}`;
    if (typeof evidencias.janela_s === "string" && evidencias.janela_s.includes("-")) {
      const [a, b] = evidencias.janela_s.split("-").map(Number);
      if (Number.isFinite(a) && Number.isFinite(b)) return el("small", { class: "janela", text: ` ${mmss(a)}–${mmss(b)}` });
    }
    if (evidencias.inicio && evidencias.fim) return el("small", { class: "janela", text: ` ${fmtInstante(evidencias.inicio)}–${fmtInstante(evidencias.fim)}` });
    return null;
  }

  function fmtHora(iso) {
    if (!iso) return "—";
    const data = new Date(iso);
    if (Number.isNaN(data.getTime())) return String(iso).slice(11, 19);
    return data.toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function fmtNum(valor, casas) {
    if (valor === null || valor === undefined || Number.isNaN(Number(valor))) return "—";
    return Number(valor).toLocaleString("pt-BR", { minimumFractionDigits: casas, maximumFractionDigits: casas });
  }

  function chip(texto, classe) {
    return el("span", { class: `chip ${classe || ""}`.trim(), text: texto });
  }

  function vazio(texto) {
    return el("p", { class: "vazio", text: texto });
  }

  function erro(mensagem) {
    return el("div", { class: "error-message", role: "alert", text: mensagem });
  }

  /* --- Painel ------------------------------------------------------------ */

  function cartaoRisco(resultado) {
    if (resultado.video && resultado.video.contexto === "exploratorio") {
      return el("section", { class: "bloco" }, [
        el("h3", { text: "Avaliação de movimento — UCF101" }),
        el("p", { text: `Pose detectada em ${resultado.video.quadros_com_pose} de ${resultado.video.quadros_analisados} quadros.` }),
        el("p", { class: "nota", text: "Ações humanas sem rótulos clínicos. Risco clínico não avaliado; consulte as medidas e a cobertura no relatório de vídeo." }),
      ]);
    }
    const fusao = resultado.fusao || {};
    const risco = Number(fusao.risco || 0);
    const nivel = fusao.nivel_risco || "sem_achados";
    const card = el("section", { class: "risco-card", dataset: { nivel } }, [
      el("p", { class: "overline", text: "Risco do paciente" }),
      el("div", { class: "risco-valor" }, [
        el("strong", { text: fmtNum(risco, 1) }),
        el("span", { text: "/ 100" }),
      ]),
      el("div", { class: "risco-barra" }, [el("span", { class: "risco-barra-preenchimento" })]),
      el("p", { class: "risco-nivel", text: `nível ${rotulo("", nivel)}` }),
      el("dl", { class: "fatos" }, [
        el("div", {}, [el("dt", { text: "Achados" }), el("dd", { text: String(resultado.total_achados ?? 0) })]),
        el("div", {}, [el("dt", { text: "Quadros" }), el("dd", { text: String((fusao.sindromes || []).length) })]),
        el("div", {}, [el("dt", { text: "Alertas" }), el("dd", { text: String(resultado.total_alertas ?? 0) })]),
      ]),
      el("p", { class: "nota" }, [
        "Processadas: ", (resultado.modalidades_processadas || []).map((m) => rotulo("modalidade", m)).join(", ") || "nenhuma",
        el("br"),
        "Ausentes: ", (resultado.modalidades_ausentes || []).map((m) => rotulo("modalidade", m)).join(", ") || "nenhuma",
      ]),
      el("p", { class: "nota" }, [
        "Serviços: ",
        Object.entries(resultado.servicos_azure || {}).map(([k, v]) => `${k} = ${v}`).join(" · ") || "—",
      ]),
    ]);
    card.querySelector(".risco-barra-preenchimento").style.width = `${Math.max(0, Math.min(100, risco))}%`;
    return card;
  }

  function cartoesQuadros(resultado) {
    const quadros = (resultado.fusao && resultado.fusao.sindromes) || [];
    const secao = el("section", { class: "bloco" }, [
      el("div", { class: "bloco-titulo" }, [el("h3", { text: "Quadros clínicos (fusão multimodal)" }), chip(String(quadros.length), "neutro")]),
    ]);
    if (!quadros.length) {
      secao.appendChild(vazio("Nenhuma combinação entre modalidades atingiu os critérios de fusão."));
      return secao;
    }
    secao.appendChild(el("div", { class: "cards" }, quadros.map((q) => el("article", { class: "quadro-card", dataset: { severidade: q.severidade } }, [
      el("header", {}, [el("strong", { text: rotulo("", q.nome) }), chip(rotulo("severidade", q.severidade), `sev-${q.severidade}`)]),
      el("p", { text: q.descricao }),
      el("p", { class: "nota" }, [
        "Modalidades: ", (q.modalidades || []).map((m) => rotulo("modalidade", m)).join(" + "),
        q.achados ? ` · ${q.achados.length} achados sustentam` : "",
      ]),
      el("p", { class: "conduta" }, [el("span", { text: "Conduta: " }), q.acao]),
    ]))));
    return secao;
  }

  function cartaoAlerta(alerta, extras) {
    const canais = (alerta.canais_entrega || []).map((c) => rotulo("canalEntrega", c));
    const cabecalho = [chip(rotulo("prioridade", alerta.prioridade), `prio-${alerta.prioridade}`), el("strong", { text: alerta.titulo })];
    if (extras && extras.hora) cabecalho.push(el("time", { class: "alerta-hora", text: extras.hora }));
    const card = el("article", { class: "alerta-card", dataset: { prioridade: alerta.prioridade } }, [
      el("header", {}, cabecalho),
      el("p", { class: "destino" }, [el("span", { text: "Para: " }), alerta.destino || "—"]),
      alerta.acao_recomendada ? el("p", { class: "conduta" }, [el("span", { text: "Conduta: " }), alerta.acao_recomendada]) : null,
      el("p", { class: "nota" }, [
        alerta.paciente_id ? `Paciente ${alerta.paciente_id} · ` : "",
        (alerta.modalidades || []).map((m) => rotulo("modalidade", m)).join(", "),
        canais.length ? ` · entregue via ${canais.join(", ")}` : "",
        extras && extras.origem ? ` · origem ${extras.origem}` : "",
      ]),
    ]);
    const achados = alerta.achados || [];
    if (achados.length) {
      const detalhes = el("details", {}, [
        el("summary", { text: `${achados.length} achado(s) que sustentam` }),
        el("ul", { class: "lista-compacta" }, achados.map((a) => el("li", {}, [
          el("code", { text: fmtInstante(a.instante) }), " ", a.descricao,
        ]))),
      ]);
      card.appendChild(detalhes);
    }
    return card;
  }

  function cartoesAlertas(alertas, titulo) {
    const secao = el("section", { class: "bloco" }, [
      el("div", { class: "bloco-titulo" }, [el("h3", { text: titulo || "Alertas para a equipe" }), chip(String(alertas.length), alertas.length ? "coral" : "neutro")]),
    ]);
    if (!alertas.length) secao.appendChild(vazio("Nenhum alerta emitido: não houve achado de atenção ou crítico."));
    else secao.appendChild(el("div", { class: "cards" }, alertas.map((a) => cartaoAlerta(a))));
    return secao;
  }

  function tabelaAchados(achados) {
    const estado = { modalidade: "", severidade: "" };
    const corpo = el("tbody");
    const filtroMod = el("select", { onchange: () => { estado.modalidade = filtroMod.value; redesenhar(); } }, [
      el("option", { value: "", text: "Todas as modalidades" }),
      ...[...new Set(achados.map((a) => a.modalidade))].map((m) => el("option", { value: m, text: rotulo("modalidade", m) })),
    ]);
    const filtroSev = el("select", { onchange: () => { estado.severidade = filtroSev.value; redesenhar(); } }, [
      el("option", { value: "", text: "Todas as severidades" }),
      ...["critico", "atencao", "informativo"].map((s) => el("option", { value: s, text: rotulo("severidade", s) })),
    ]);
    const contador = el("span", { class: "chip neutro", text: String(achados.length) });

    function redesenhar() {
      limpar(corpo);
      const filtrados = achados.filter((a) => (!estado.modalidade || a.modalidade === estado.modalidade) && (!estado.severidade || a.severidade === estado.severidade));
      contador.textContent = `${filtrados.length} de ${achados.length}`;
      if (!filtrados.length) {
        corpo.appendChild(el("tr", {}, [el("td", { colspan: "4", class: "vazio", text: "Nenhum achado com esse filtro." })]));
        return;
      }
      filtrados.forEach((a) => corpo.appendChild(el("tr", { dataset: { severidade: a.severidade } }, [
        el("td", { text: rotulo("modalidade", a.modalidade) }),
        el("td", {}, [chip(rotulo("severidade", a.severidade), `sev-${a.severidade}`)]),
        el("td", {}, [el("code", { text: fmtInstante(a.instante) }), fmtJanela(a.evidencias)]),
        el("td", { text: a.descricao }),
      ])));
    }
    redesenhar();
    return el("section", { class: "bloco" }, [
      el("div", { class: "bloco-titulo" }, [el("h3", { text: "Todos os achados" }), contador]),
      el("div", { class: "filtros" }, [filtroMod, filtroSev]),
      el("div", { class: "tabela-wrap" }, [el("table", { class: "tabela-achados" }, [
        el("thead", {}, [el("tr", {}, ["Modalidade", "Severidade", "Instante", "Achado"].map((t) => el("th", { text: t })))]),
        corpo,
      ])]),
    ]);
  }

  function painelRelatorio(nomes, carregar) {
    const abas = el("div", { class: "relatorio-abas" });
    const conteudo = el("div", { class: "relatorio" });
    const titulos = {
      relatorio_monitoramento: "Monitoramento", relatorio_audio: "Áudio", relatorio_video: "Vídeo", relatorio_physionet: "PhysioNet",
    };
    async function abrir(nome, botao) {
      abas.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b === botao));
      limpar(conteudo).appendChild(vazio("Carregando relatório…"));
      try {
        const dados = await carregar(nome);
        limpar(conteudo);
        const container = el("div", { class: "relatorio-html" });
        container.innerHTML = dados.html; // HTML vindo do servidor, ja escapado pelo conversor
        conteudo.appendChild(container);
      } catch (erroApi) {
        limpar(conteudo).appendChild(erro(erroApi.message));
      }
    }
    nomes.forEach((nome, indice) => {
      const botao = el("button", { type: "button", class: "view-tab", text: titulos[nome] || nome });
      botao.addEventListener("click", () => abrir(nome, botao));
      abas.appendChild(botao);
      if (indice === 0) abrir(nome, botao);
    });
    if (!nomes.length) conteudo.appendChild(vazio("Nenhum relatório gerado ainda."));
    return el("section", { class: "bloco" }, [
      el("div", { class: "bloco-titulo" }, [el("h3", { text: "Relatórios" })]),
      el("div", { class: "view-tabs relatorio-tabs" }, [abas]),
      conteudo,
    ]);
  }

  function transcricao(audio) {
    if (!audio || !audio.transcricao) return vazio("Sem transcrição.");
    const termos = ((audio.analise_texto && audio.analise_texto.termos_criticos) || []).map((t) => t.termo.toLowerCase());
    const segmentos = audio.transcricao.segmentos || [];
    function destacar(texto) {
      const partes = [];
      let resto = texto;
      while (resto.length) {
        let melhor = null;
        termos.forEach((termo) => {
          const posicao = resto.toLowerCase().indexOf(termo);
          if (posicao >= 0 && (melhor === null || posicao < melhor.posicao)) melhor = { posicao, termo };
        });
        if (!melhor) { partes.push(resto); break; }
        if (melhor.posicao > 0) partes.push(resto.slice(0, melhor.posicao));
        partes.push(el("mark", { text: resto.slice(melhor.posicao, melhor.posicao + melhor.termo.length) }));
        resto = resto.slice(melhor.posicao + melhor.termo.length);
      }
      return partes;
    }
    const lista = el("ul", { class: "transcricao" }, segmentos.map((s) => el("li", {}, [
      el("code", { text: `${Math.floor(s.inicio_s / 60).toString().padStart(2, "0")}:${(s.inicio_s % 60).toFixed(1).padStart(4, "0")}` }),
      " ", ...destacar(s.texto),
    ])));
    const analise = audio.analise_texto || {};
    return el("div", {}, [
      el("p", { class: "nota" }, [
        `Transcrição por ${audio.transcricao.provedor} (confiança ${fmtNum(audio.transcricao.confianca, 2)})`,
        audio.wer_referencia !== null && audio.wer_referencia !== undefined ? ` · WER ${fmtNum(audio.wer_referencia * 100, 1)}%` : "",
        ` · sentimento ${analise.sentimento || "—"} (${analise.provedor || "—"})`,
      ]),
      lista,
      analise.termos_criticos && analise.termos_criticos.length
        ? el("p", { class: "nota" }, ["Termos críticos: ", analise.termos_criticos.map((t) => `${t.termo} (${t.categoria}, ${t.peso})`).join("; ")])
        : vazio("Nenhum termo crítico do léxico encontrado."),
    ]);
  }

  function scrubber(pacienteId, info, fonte) {
    const imagem = el("img", { class: "quadro-video", alt: "Quadro do vídeo" });
    const faixa = el("input", { type: "range", min: "0", max: String(Math.max(0, (info.duracao_s || 0) - 1 / (info.fps || 30))), step: "0.1", value: "0" });
    const rotuloTempo = el("span", { class: "nota", text: "0,0 s" });
    let agendado = null;
    function atualizar() {
      const t = Number(faixa.value);
      rotuloTempo.textContent = `${fmtNum(t, 1)} s de ${fmtNum(info.duracao_s, 1)} s`;
      if (agendado) cancelAnimationFrame(agendado);
      agendado = requestAnimationFrame(() => {
        imagem.src = `/api/pacientes/${encodeURIComponent(pacienteId)}/quadro.jpg?fonte=${fonte}&t=${t.toFixed(2)}&_=${Date.now()}`;
      });
    }
    faixa.addEventListener("input", atualizar);
    atualizar();
    return el("div", { class: "scrubber" }, [
      imagem,
      el("div", { class: "scrubber-controles" }, [faixa, rotuloTempo]),
      el("p", { class: "nota", text: `O MP4 gerado pelo OpenCV (${info.fourcc || "FMP4"}) não é reproduzido pelo navegador; os quadros são servidos pelo servidor.` }),
    ]);
  }

  function midia(resultado, infoVideo) {
    const midias = resultado.midias || {};
    const partes = [];
    if (midias.audio) {
      partes.push(el("h4", { text: "Áudio da consulta" }));
      partes.push(el("audio", { controls: "controls", preload: "metadata", src: midias.audio }));
    }
    partes.push(el("h4", { text: "Transcrição e termos críticos" }));
    partes.push(transcricao(resultado.audio));
    if (midias.video_anotado || midias.video_original) {
      partes.push(el("h4", { text: midias.video_anotado ? "Vídeo anotado da sessão" : "Vídeo da sessão" }));
      const fonte = midias.video_anotado ? "anotado" : "original";
      if (infoVideo && infoVideo.reproduzivel_no_navegador) {
        partes.push(el("video", { controls: "controls", preload: "metadata", src: `${midias.video_anotado || midias.video_original}?h264=1` }));
      } else if (infoVideo) {
        partes.push(scrubber(resultado.paciente_id, infoVideo, fonte));
      }
      const video = resultado.video || {};
      partes.push(el("p", { class: "nota" }, [
        `${video.quadros_analisados ?? "—"} quadros a ${fmtNum(video.fps, 0)} fps, pose detectada em ${video.quadros_com_pose ?? "—"} · `,
        `área crítica ${video.origem_area_critica || "—"} · mão por ${video.regra_mao || "—"}`,
      ]));
    }
    return el("section", { class: "bloco midia" }, partes);
  }

  function jobStatus(job, aoCancelar) {
    if (!job) return null;
    const ativo = ["aguardando", "executando", "cancelando"].includes(job.estado);
    const classe = ativo ? "running" : job.estado === "concluido" ? "success" : "failed";
    const barra = el("div", { class: "progress-track" }, [el("span")]);
    barra.firstChild.style.width = `${Math.round((job.progresso || 0) * 100)}%`;
    return el("section", { class: "bloco job" }, [
      el("div", { class: "bloco-titulo" }, [
        el("h3", { text: job.titulo || "Execução em segundo plano" }),
        el("span", { class: `job-status ${classe}`, text: job.estado }),
      ]),
      el("p", { class: "nota", text: job.etapa || "" }),
      barra,
      job.erro ? erro(job.erro) : null,
      el("pre", { class: "job-log", text: (job.logs || []).slice(-40).join("\n") }),
      ativo && aoCancelar ? el("button", { type: "button", class: "danger-button", text: "Cancelar", onclick: aoCancelar }) : null,
    ]);
  }

  F4.render = {
    el, limpar, chip, vazio, erro, rotulo, fmtInstante, fmtHora, fmtNum,
    cartaoRisco, cartoesQuadros, cartaoAlerta, cartoesAlertas, tabelaAchados, painelRelatorio, midia, jobStatus, scrubber,
  };
})(window.F4);
