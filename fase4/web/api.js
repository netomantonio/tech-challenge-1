/* Acesso a API da interface da Fase 4: JSON, upload com progresso e SSE. */
window.F4 = window.F4 || {};

(function (F4) {
  "use strict";

  class ErroApi extends Error {
    constructor(mensagem, status, corpo) {
      super(mensagem);
      this.status = status;
      this.corpo = corpo;
    }
  }

  function detalhe(corpo, status) {
    if (corpo && corpo.detail) {
      if (typeof corpo.detail === "string") return corpo.detail;
      try {
        return corpo.detail.map((d) => `${(d.loc || []).slice(-1)[0] || "campo"}: ${d.msg}`).join("; ");
      } catch (_erro) {
        return JSON.stringify(corpo.detail);
      }
    }
    return `HTTP ${status}`;
  }

  async function json(caminho, opcoes) {
    const config = Object.assign({ headers: {} }, opcoes || {});
    if (config.body && typeof config.body !== "string") config.body = JSON.stringify(config.body);
    if (config.body) config.headers["Content-Type"] = "application/json";
    const resposta = await fetch(caminho, config);
    let corpo = null;
    const tipo = resposta.headers.get("content-type") || "";
    if (tipo.includes("application/json")) {
      try { corpo = await resposta.json(); } catch (_erro) { corpo = null; }
    }
    if (!resposta.ok) throw new ErroApi(detalhe(corpo, resposta.status), resposta.status, corpo);
    return corpo;
  }

  function putArquivo(url, arquivo, aoProgredir) {
    return new Promise((resolver, rejeitar) => {
      const xhr = new XMLHttpRequest();
      xhr.open("PUT", url, true);
      xhr.setRequestHeader("Content-Type", "application/octet-stream");
      xhr.upload.onprogress = (evento) => {
        if (aoProgredir && evento.lengthComputable) aoProgredir(evento.loaded / evento.total);
      };
      xhr.onerror = () => rejeitar(new ErroApi("Falha de rede no envio.", 0, null));
      xhr.onload = () => {
        let corpo = null;
        try { corpo = JSON.parse(xhr.responseText); } catch (_erro) { corpo = null; }
        if (xhr.status >= 200 && xhr.status < 300) resolver(corpo);
        else rejeitar(new ErroApi(detalhe(corpo, xhr.status), xhr.status, corpo));
      };
      xhr.send(arquivo);
    });
  }

  /* EventSource com um handler por nome de evento; `fechar()` encerra. */
  function sse(url, handlers) {
    const fonte = new EventSource(url);
    Object.keys(handlers).forEach((nome) => {
      if (nome === "erro") return;
      fonte.addEventListener(nome, (evento) => {
        let dados = null;
        try { dados = JSON.parse(evento.data); } catch (_erro) { dados = evento.data; }
        handlers[nome](dados, evento);
      });
    });
    fonte.onerror = (evento) => {
      if (handlers.erro) handlers.erro(evento, fonte.readyState);
    };
    return { fonte, fechar: () => fonte.close() };
  }

  F4.api = {
    ErroApi,
    get: (caminho) => json(caminho),
    post: (caminho, corpo) => json(caminho, { method: "POST", body: corpo || {} }),
    del: (caminho) => json(caminho, { method: "DELETE" }),
    putArquivo,
    sse,
  };
})(window.F4);
