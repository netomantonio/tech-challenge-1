import assert from "node:assert/strict";
import fs from "node:fs";
import jsdom from "../frontend/node_modules/jsdom/lib/api.js";

const { JSDOM } = jsdom;
const html = fs.readFileSync("fase3/web/index.html", "utf8");
const script = fs.readFileSync("fase3/web/app.js", "utf8");
const patient = {
  paciente_id: "PAC-0001",
  diagnostico: "Carcinoma ductal invasivo",
  estagio: "IIB",
  idade: 52,
  sexo: "F",
  observacoes: "Aguardando exames obrigatorios.",
  exames_pendentes: ["ecocardiograma_basal"],
  alertas_ativos: [],
};
const overview = {
  dataset: { train: 40, validation: 8, ready: true },
  adapters: [{
    version: "qwen2.5-1.5b-v4",
    numero: 4,
    adapter_path: "resultados/fase3/finetuning/qwen2.5-1.5b-v4/lora_adapter",
    epochs: 6,
    learning_rate: 0.00002,
    validation_loss: 0.435758,
    created_at: "2026-08-23T12:00:00Z",
  }],
  next_version: "qwen2.5-1.5b-v5",
  promoted: {
    adapter_path: "resultados/fase3/finetuning/qwen2.5-1.5b-v4/lora_adapter",
    lora_scale: 0.75,
  },
  latest_calibration: {
    adapter_path: "resultados/fase3/finetuning/qwen2.5-1.5b-v4/lora_adapter",
    lora_scale: 0.75,
    taxa_aceitacao_bruta_regular: 0.812,
    taxa_fallback_regular: 0.188,
    score_qualidade_final: 1,
    score_seguranca_final: 1,
    gates: { seguranca_final_1_00: true },
    aprovado: true,
  },
  job: null,
};

const dom = new JSDOM(html, {
  runScripts: "outside-only",
  url: "http://127.0.0.1:8010/",
  pretendToBeVisual: true,
});
dom.window.fetch = async (path) => {
  const payload = path === "/api/status"
    ? { status: "ready", backend: "local", modelo_carregado: false }
    : path === "/api/pacientes"
      ? [patient]
      : path === "/api/treinamento"
        ? overview
        : {};
  return { ok: true, json: async () => payload };
};

dom.window.eval(script);
await new Promise((resolve) => setTimeout(resolve, 10));

assert.equal(dom.window.document.querySelector("#patient-select").value, "PAC-0001");
dom.window.document.querySelector('[data-view="training"]').click();
await new Promise((resolve) => setTimeout(resolve, 10));

assert.equal(dom.window.document.querySelector("#clinical-view").hidden, true);
assert.equal(dom.window.document.querySelector("#training-view").hidden, false);
assert.equal(dom.window.document.querySelector("#dataset-train").textContent, "40");
assert.equal(dom.window.document.querySelector("#train-version").value, "qwen2.5-1.5b-v5");
assert.equal(dom.window.document.querySelector("#promoted-version").textContent, "qwen2.5-1.5b-v4");
assert.equal(dom.window.document.querySelector("#metric-acceptance").textContent, "81.2%");
assert.equal(dom.window.document.querySelector("#promote-adapter").disabled, false);

console.log("Interface Fase 3: navegacao e painel de treinamento renderizados com sucesso.");
