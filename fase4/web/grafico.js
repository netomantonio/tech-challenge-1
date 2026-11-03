/* Grafico dos sinais vitais em <canvas>, sem biblioteca: uma faixa por canal,
   bandas normal/critica sombreadas, marcadores nos minutos com achado. */
window.F4 = window.F4 || {};

(function (F4) {
  "use strict";

  const CORES = {
    linha: "#0d6a63", normal: "rgba(13, 106, 99, 0.10)", critico: "rgba(168, 68, 58, 0.12)",
    grade: "#d8e1de", texto: "#40534f", agora: "#846317",
    prioridade: { vermelho: "#a8443a", laranja: "#c97a1a", amarelo: "#846317" },
    informativo: "#70807c", multivariado: "#5b3f9e",
  };
  const ROTULOS = {
    frequencia_cardiaca: "FC (bpm)", pressao_sistolica: "PAS (mmHg)", pressao_diastolica: "PAD (mmHg)",
    saturacao_oxigenio: "SpO₂ (%)", temperatura: "Temp (°C)", frequencia_respiratoria: "FR (irpm)",
  };
  const GUTTER = 118;
  const ALTURA_FAIXA = 84;
  const TOPO = 18;
  const RODAPE = 26;

  class GraficoSinais {
    constructor(canvas, opcoes) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.canais = opcoes.canais || [];
      this.faixas = {};
      (opcoes.faixas || []).forEach((f) => { this.faixas[f.nome] = f; });
      this.totalMinutos = Math.max(1, opcoes.totalMinutos || 720);
      this.inicio = opcoes.inicio ? new Date(opcoes.inicio) : null;
      this.leituras = [];
      this.marcadores = [];
      this.cursor = -1;
      this.dominios = {};
      this.canais.forEach((c) => { this.dominios[c] = this._dominioInicial(c); });
      this._agendado = null;
      this._legenda = opcoes.legenda || null;
      this._observador = new ResizeObserver(() => this.agendar());
      this._observador.observe(canvas.parentElement || canvas);
      canvas.addEventListener("mousemove", (e) => this._aoMover(e));
      canvas.addEventListener("mouseleave", () => this._legendar(null));
      this.agendar();
    }

    destruir() {
      this._observador.disconnect();
    }

    _dominioInicial(canal) {
      const f = this.faixas[canal];
      if (!f) return [0, 1];
      const amplitude = Math.max(1e-6, f.maximo - f.minimo);
      const baixo = f.critico_baixo !== null && f.critico_baixo !== undefined ? f.critico_baixo : f.minimo;
      const alto = f.critico_alto !== null && f.critico_alto !== undefined ? f.critico_alto : f.maximo;
      return [baixo - amplitude * 0.35, alto + amplitude * 0.35];
    }

    adicionar(leitura) {
      this.leituras.push(leitura);
      this.cursor = leitura.minuto;
      this.canais.forEach((c) => {
        const v = leitura.valores ? leitura.valores[c] : undefined;
        if (v === undefined || v === null) return;
        const d = this.dominios[c];
        if (v < d[0]) d[0] = v - Math.abs(v) * 0.02 - 0.5;
        if (v > d[1]) d[1] = v + Math.abs(v) * 0.02 + 0.5;
      });
      (leitura.achados || []).forEach((a) => {
        const canal = this.canais.find((c) => a.tipo && a.tipo.endsWith(c));
        this.marcadores.push({
          minuto: leitura.minuto, canal, severidade: a.severidade,
          multivariado: a.tipo === "anomalia_multivariada",
        });
      });
      (leitura.alertas || []).forEach((al) => {
        this.marcadores.push({ minuto: leitura.minuto, alerta: true, prioridade: al.prioridade });
      });
      this.agendar();
    }

    agendar() {
      if (this._agendado) return;
      this._agendado = requestAnimationFrame(() => { this._agendado = null; this.desenhar(); });
    }

    _geometria() {
      const dpr = window.devicePixelRatio || 1;
      const largura = Math.max(320, (this.canvas.parentElement || this.canvas).clientWidth || 640);
      const altura = TOPO + this.canais.length * ALTURA_FAIXA + RODAPE;
      if (this.canvas.width !== Math.round(largura * dpr) || this.canvas.height !== Math.round(altura * dpr)) {
        this.canvas.width = Math.round(largura * dpr);
        this.canvas.height = Math.round(altura * dpr);
        this.canvas.style.height = `${altura}px`;
      }
      return { dpr, largura, altura, x0: GUTTER, x1: largura - 12 };
    }

    _x(minuto, g) {
      return g.x0 + ((g.x1 - g.x0) * minuto) / Math.max(1, this.totalMinutos - 1);
    }

    _y(valor, canal, indice) {
      const [baixo, alto] = this.dominios[canal];
      const topo = TOPO + indice * ALTURA_FAIXA + 8;
      const base = TOPO + (indice + 1) * ALTURA_FAIXA - 8;
      const razao = (valor - baixo) / Math.max(1e-6, alto - baixo);
      return base - razao * (base - topo);
    }

    _rotuloTempo(minuto) {
      if (!this.inicio) return `${minuto} min`;
      const data = new Date(this.inicio.getTime() + minuto * 60000);
      return `${String(data.getHours()).padStart(2, "0")}:${String(data.getMinutes()).padStart(2, "0")}`;
    }

    desenhar() {
      const g = this._geometria();
      const ctx = this.ctx;
      ctx.setTransform(g.dpr, 0, 0, g.dpr, 0, 0);
      ctx.clearRect(0, 0, g.largura, g.altura);
      ctx.font = "11px Inter, system-ui, sans-serif";

      this.canais.forEach((canal, i) => {
        const topo = TOPO + i * ALTURA_FAIXA;
        const base = topo + ALTURA_FAIXA;
        const f = this.faixas[canal];
        // Fundo e bandas.
        ctx.fillStyle = i % 2 ? "#ffffff" : "#f6f8f7";
        ctx.fillRect(g.x0, topo, g.x1 - g.x0, ALTURA_FAIXA);
        if (f) {
          const yMax = this._y(f.maximo, canal, i);
          const yMin = this._y(f.minimo, canal, i);
          ctx.fillStyle = CORES.normal;
          ctx.fillRect(g.x0, yMax, g.x1 - g.x0, yMin - yMax);
          ctx.fillStyle = CORES.critico;
          if (f.critico_alto !== null && f.critico_alto !== undefined) {
            const yCrit = this._y(f.critico_alto, canal, i);
            ctx.fillRect(g.x0, topo + 8, g.x1 - g.x0, Math.max(0, yCrit - topo - 8));
          }
          if (f.critico_baixo !== null && f.critico_baixo !== undefined) {
            const yCrit = this._y(f.critico_baixo, canal, i);
            ctx.fillRect(g.x0, yCrit, g.x1 - g.x0, Math.max(0, base - 8 - yCrit));
          }
        }
        ctx.strokeStyle = CORES.grade;
        ctx.beginPath(); ctx.moveTo(g.x0, base); ctx.lineTo(g.x1, base); ctx.stroke();
        // Rotulos da faixa.
        ctx.fillStyle = CORES.texto;
        ctx.textAlign = "left";
        ctx.fillText(ROTULOS[canal] || canal, 8, topo + 20);
        const d = this.dominios[canal];
        ctx.fillStyle = "#70807c";
        ctx.font = "10px Inter, system-ui, sans-serif";
        ctx.fillText(d[1].toFixed(d[1] < 50 ? 1 : 0), 8, topo + 36);
        ctx.fillText(d[0].toFixed(d[0] < 50 ? 1 : 0), 8, base - 12);
        ctx.font = "11px Inter, system-ui, sans-serif";
        // Serie.
        ctx.strokeStyle = CORES.linha;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        let desenhando = false;
        this.leituras.forEach((l) => {
          const v = l.valores ? l.valores[canal] : undefined;
          if (v === undefined || v === null) { desenhando = false; return; }
          const x = this._x(l.minuto, g);
          const y = this._y(v, canal, i);
          if (!desenhando) { ctx.moveTo(x, y); desenhando = true; } else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.lineWidth = 1;
      });

      // Eixo do tempo.
      ctx.fillStyle = CORES.texto;
      ctx.textAlign = "center";
      const passo = this.totalMinutos > 360 ? 60 : 30;
      for (let m = 0; m < this.totalMinutos; m += passo) {
        const x = this._x(m, g);
        ctx.strokeStyle = CORES.grade;
        ctx.beginPath(); ctx.moveTo(x, TOPO); ctx.lineTo(x, g.altura - RODAPE); ctx.stroke();
        ctx.fillText(this._rotuloTempo(m), x, g.altura - 8);
      }

      // Marcadores.
      this.marcadores.forEach((mk) => {
        const x = this._x(mk.minuto, g);
        if (mk.alerta) {
          ctx.fillStyle = CORES.prioridade[mk.prioridade] || CORES.informativo;
          ctx.beginPath(); ctx.moveTo(x, TOPO - 2); ctx.lineTo(x - 5, TOPO - 12); ctx.lineTo(x + 5, TOPO - 12); ctx.closePath(); ctx.fill();
          return;
        }
        if (mk.multivariado) {
          ctx.strokeStyle = CORES.multivariado;
          ctx.setLineDash([3, 3]);
          ctx.beginPath(); ctx.moveTo(x, TOPO); ctx.lineTo(x, g.altura - RODAPE); ctx.stroke();
          ctx.setLineDash([]);
          return;
        }
        const indice = this.canais.indexOf(mk.canal);
        if (indice < 0) return;
        const leitura = this.leituras.find((l) => l.minuto === mk.minuto);
        const v = leitura && leitura.valores ? leitura.valores[mk.canal] : undefined;
        if (v === undefined) return;
        const y = this._y(v, mk.canal, indice);
        ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2);
        if (mk.severidade === "informativo") { ctx.strokeStyle = CORES.informativo; ctx.stroke(); }
        else { ctx.fillStyle = mk.severidade === "critico" ? CORES.prioridade.vermelho : CORES.prioridade.laranja; ctx.fill(); }
      });

      // Cursor do "agora".
      if (this.cursor >= 0) {
        const x = this._x(this.cursor, g);
        ctx.strokeStyle = CORES.agora;
        ctx.beginPath(); ctx.moveTo(x, TOPO - 2); ctx.lineTo(x, g.altura - RODAPE); ctx.stroke();
      }
    }

    _aoMover(evento) {
      const g = this._geometria();
      const rect = this.canvas.getBoundingClientRect();
      const x = evento.clientX - rect.left;
      if (x < g.x0 || x > g.x1 || !this.leituras.length) { this._legendar(null); return; }
      const minuto = Math.round(((x - g.x0) / Math.max(1, g.x1 - g.x0)) * (this.totalMinutos - 1));
      let melhor = this.leituras[0];
      this.leituras.forEach((l) => { if (Math.abs(l.minuto - minuto) < Math.abs(melhor.minuto - minuto)) melhor = l; });
      this._legendar(melhor);
    }

    _legendar(leitura) {
      if (!this._legenda) return;
      while (this._legenda.firstChild) this._legenda.removeChild(this._legenda.firstChild);
      if (!leitura) { this._legenda.textContent = "Passe o mouse sobre o gráfico para ler os valores."; return; }
      const partes = [`${this._rotuloTempo(leitura.minuto)}`];
      this.canais.forEach((c) => {
        const v = leitura.valores ? leitura.valores[c] : undefined;
        partes.push(`${(ROTULOS[c] || c).split(" ")[0]} ${v === undefined ? "—" : Number(v).toLocaleString("pt-BR", { maximumFractionDigits: 1 })}`);
      });
      this._legenda.textContent = partes.join("  ·  ");
    }
  }

  F4.GraficoSinais = GraficoSinais;
})(window.F4);
