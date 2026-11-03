/* Desenho da area critica (ROI) sobre um quadro do video. As coordenadas
   saem em pixels do video original, independentemente do tamanho exibido. */
window.F4 = window.F4 || {};

(function (F4) {
  "use strict";

  class RoiCanvas {
    constructor(container, aoMudar) {
      this.container = container;
      this.aoMudar = aoMudar;
      this.imagem = document.createElement("img");
      this.imagem.className = "roi-imagem";
      this.imagem.alt = "Quadro do vídeo para marcar a área crítica";
      this.canvas = document.createElement("canvas");
      this.canvas.className = "roi-canvas";
      this.wrap = document.createElement("div");
      this.wrap.className = "roi-wrap";
      this.wrap.appendChild(this.imagem);
      this.wrap.appendChild(this.canvas);
      container.appendChild(this.wrap);
      this.largura = 0;
      this.altura = 0;
      this.roi = null;
      this._arrastando = null;
      this.imagem.addEventListener("load", () => this._ajustar());
      new ResizeObserver(() => this._ajustar()).observe(this.wrap);
      this.canvas.addEventListener("pointerdown", (e) => this._inicio(e));
      this.canvas.addEventListener("pointermove", (e) => this._mover(e));
      this.canvas.addEventListener("pointerup", (e) => this._fim(e));
      this.canvas.addEventListener("pointercancel", (e) => this._fim(e));
    }

    carregar(url, largura, altura) {
      this.largura = largura;
      this.altura = altura;
      this.imagem.src = url;
    }

    definir(roi) {
      this.roi = roi ? roi.map(Number) : null;
      this._desenhar();
    }

    _ajustar() {
      const rect = this.imagem.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      this.canvas.style.width = `${rect.width}px`;
      this.canvas.style.height = `${rect.height}px`;
      this.canvas.width = Math.round(rect.width * dpr);
      this.canvas.height = Math.round(rect.height * dpr);
      this._desenhar();
    }

    _escala() {
      const rect = this.canvas.getBoundingClientRect();
      const largura = this.largura || this.imagem.naturalWidth || rect.width;
      const altura = this.altura || this.imagem.naturalHeight || rect.height;
      return { sx: largura / Math.max(1, rect.width), sy: altura / Math.max(1, rect.height), rect, largura, altura };
    }

    _ponto(evento) {
      const { sx, sy, rect, largura, altura } = this._escala();
      const x = Math.min(largura, Math.max(0, (evento.clientX - rect.left) * sx));
      const y = Math.min(altura, Math.max(0, (evento.clientY - rect.top) * sy));
      return [x, y];
    }

    _inicio(evento) {
      this.canvas.setPointerCapture(evento.pointerId);
      this._arrastando = this._ponto(evento);
      this.roi = null;
      this._desenhar();
    }

    _mover(evento) {
      if (!this._arrastando) return;
      const atual = this._ponto(evento);
      this.roi = this._normalizar(this._arrastando, atual);
      this._desenhar();
    }

    _fim(evento) {
      if (!this._arrastando) return;
      const atual = this._ponto(evento);
      this.roi = this._normalizar(this._arrastando, atual);
      this._arrastando = null;
      if (this.roi && (this.roi[2] - this.roi[0] < 2 || this.roi[3] - this.roi[1] < 2)) this.roi = null;
      this._desenhar();
      if (this.aoMudar) this.aoMudar(this.roi);
    }

    _normalizar(a, b) {
      return [Math.round(Math.min(a[0], b[0])), Math.round(Math.min(a[1], b[1])), Math.round(Math.max(a[0], b[0])), Math.round(Math.max(a[1], b[1]))];
    }

    _desenhar() {
      const ctx = this.canvas.getContext("2d");
      const dpr = window.devicePixelRatio || 1;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const { sx, sy, rect } = this._escala();
      ctx.clearRect(0, 0, rect.width, rect.height);
      if (!this.roi) return;
      const [x1, y1, x2, y2] = this.roi;
      const px = x1 / sx, py = y1 / sy, pw = (x2 - x1) / sx, ph = (y2 - y1) / sy;
      ctx.fillStyle = "rgba(168, 68, 58, 0.18)";
      ctx.fillRect(px, py, pw, ph);
      ctx.strokeStyle = "#a8443a";
      ctx.lineWidth = 2;
      ctx.strokeRect(px, py, pw, ph);
      ctx.fillStyle = "#a8443a";
      ctx.font = "11px Inter, system-ui, sans-serif";
      ctx.fillText(`${x1},${y1} → ${x2},${y2}`, px + 4, Math.max(12, py - 4));
    }
  }

  F4.RoiCanvas = RoiCanvas;
})(window.F4);
