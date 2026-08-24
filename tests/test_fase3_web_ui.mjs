import assert from "node:assert/strict";
import fs from "node:fs";
import jsdom from "../frontend/node_modules/jsdom/lib/api.js";

const { JSDOM } = jsdom;
const html = fs.readFileSync("fase3/web/index.html", "utf8");
const script = fs.readFileSync("fase3/web/app.js", "utf8");
const patient = {
  paciente_id: "PAC-0001",
  diagnostico: "Carcinoma ductal invasivo",
  estagio: "a_definir",
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
  next_versions: { "qwen2.5-1.5b": "qwen2.5-1.5b-v5" },
  models: [{ alias: "qwen2.5-1.5b", label: "Qwen2.5 1.5B", source: "Qwen/Qwen2.5-1.5B-Instruct", installed: true, builtin: true }],
  hardware: { device: "cuda", device_name: "RTX Test", memory_bytes: 12884901888, quantization: ["auto", "fp32", "fp16", "nf4"] },
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
const profiles = [
  { id: "backend-a", name: "Backend A", baseUrl: "http://127.0.0.1:8010", networkType: "loopback" },
  { id: "backend-b", name: "Backend B", baseUrl: "http://192.168.1.20:8010", networkType: "private" },
];
dom.window.localStorage.setItem("fase3.backendProfiles.v1", JSON.stringify(profiles));
dom.window.localStorage.setItem("fase3.activeBackend.v1", "backend-a");
dom.window.localStorage.setItem("fase3.completedTours.v1", JSON.stringify({ clinical: true, training: true }));
const requests = [];
dom.window.fetch = async (input) => {
  const path = new URL(String(input), dom.window.location.href).pathname;
  requests.push(path);
  const payload = path === "/api/status"
    ? { status: "ready", backend: "local", modelo_carregado: false }
    : path === "/api/capabilities"
      ? { api_version: "2.0.0", instance_name: "Teste", hardware: overview.hardware, models: overview.models, promoted: { model_alias: "qwen2.5-1.5b" }, features: { remote_training: false } }
    : path === "/api/pacientes"
      ? [patient]
      : path === "/api/treinamento"
        ? overview
        : path === "/api/consultas"
          ? { resposta: "Resposta teste", modo_resposta: "llm", bloqueado: false, fontes: [], alertas: [], etapas_executadas: [] }
        : {};
  return { ok: true, json: async () => payload };
};

dom.window.eval(script);
await new Promise((resolve) => setTimeout(resolve, 10));

assert.equal(dom.window.document.querySelector("#patient-select").value, "PAC-0001");
assert.equal(dom.window.document.querySelector(".stage-badge").textContent, "Estagio em definicao");
assert.equal(dom.window.fase3ClassifyBackendUrl("http://127.0.0.1:8010").type, "loopback");
assert.equal(dom.window.fase3ClassifyBackendUrl("http://192.168.1.20:8010").type, "private");
assert.equal(dom.window.fase3ClassifyBackendUrl("http://example.com:8010").type, "invalid");

const question = dom.window.document.querySelector("#question");
question.value = "Qual a conduta segura?";
question.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
const requestsBeforeShortcut = requests.filter((path) => path === "/api/consultas").length;
const lineBreakEvent = new dom.window.KeyboardEvent("keydown", { key: "Enter", shiftKey: true, bubbles: true, cancelable: true });
question.dispatchEvent(lineBreakEvent);
assert.equal(lineBreakEvent.defaultPrevented, false);
assert.equal(requests.filter((path) => path === "/api/consultas").length, requestsBeforeShortcut);
const submitEvent = new dom.window.KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true });
question.dispatchEvent(submitEvent);
assert.equal(submitEvent.defaultPrevented, true);
await new Promise((resolve) => setTimeout(resolve, 20));
assert.equal(requests.filter((path) => path === "/api/consultas").length, requestsBeforeShortcut + 1);
assert.equal(dom.window.document.querySelectorAll(".message").length, 2);

const profileSelect = dom.window.document.querySelector("#profile-select");
profileSelect.value = "backend-b";
profileSelect.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
await new Promise((resolve) => setTimeout(resolve, 20));
assert.equal(dom.window.document.querySelectorAll(".message").length, 0);
profileSelect.value = "backend-a";
profileSelect.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
await new Promise((resolve) => setTimeout(resolve, 20));
assert.equal(dom.window.document.querySelectorAll(".message").length, 2);

dom.window.document.querySelector('[data-view="training"]').click();
await new Promise((resolve) => setTimeout(resolve, 10));

assert.equal(dom.window.document.querySelector("#clinical-view").hidden, true);
assert.equal(dom.window.document.querySelector("#training-view").hidden, false);
assert.equal(dom.window.document.querySelector("#dataset-train").textContent, "40");
assert.equal(dom.window.document.querySelector("#train-version").value, "qwen2.5-1.5b-v5");
assert.equal(dom.window.document.querySelector("#promoted-version").textContent, "qwen2.5-1.5b-v4");
assert.equal(dom.window.document.querySelector("#metric-acceptance").textContent, "81.2%");
assert.equal(dom.window.document.querySelector("#promote-adapter").disabled, false);
const requestsBeforeTour = requests.length;
dom.window.document.querySelector("#help-button").click();
await new Promise((resolve) => setTimeout(resolve, 5));
assert.equal(requests.length, requestsBeforeTour);
assert.equal(dom.window.document.querySelector("#tour-layer").hidden, false);
dom.window.document.querySelector("#tour-close").click();

const firstRun = new JSDOM(html, { runScripts: "outside-only", url: "https://fase3.pages.dev/", pretendToBeVisual: true });
firstRun.window.HTMLDialogElement.prototype.showModal = function showModal() { this.setAttribute("open", ""); };
firstRun.window.HTMLDialogElement.prototype.close = function close() { this.removeAttribute("open"); };
firstRun.window.fetch = async () => { throw new Error("Nao deveria consultar sem perfil"); };
firstRun.window.eval(script);
await new Promise((resolve) => setTimeout(resolve, 5));
assert.equal(firstRun.window.document.querySelector("#setup-dialog").open, true);
assert.match(firstRun.window.document.querySelector("#setup-title").textContent, /Configurar backend/);
firstRun.window.document.querySelector("#setup-next").click();
const generatedCommand = firstRun.window.document.querySelector("#setup-command");
assert.match(generatedCommand.textContent, /^python -m fase3/);
assert.match(generatedCommand.textContent, /--allowed-origin https:\/\/fase3\.pages\.dev/);
assert.match(generatedCommand.textContent, /--instance-name "Backend local"/);
assert.match(generatedCommand.textContent, /--no-open-browser/);
firstRun.window.document.querySelector("#setup-bind-host").value = "0.0.0.0";
firstRun.window.document.querySelector("#setup-bind-host").dispatchEvent(new firstRun.window.Event("change", { bubbles: true }));
firstRun.window.document.querySelector("#setup-port").value = "9010";
firstRun.window.document.querySelector("#setup-port").dispatchEvent(new firstRun.window.Event("change", { bubbles: true }));
firstRun.window.document.querySelector("#setup-remote-training").click();
assert.match(generatedCommand.textContent, /--host 0\.0\.0\.0/);
assert.match(generatedCommand.textContent, /--port 9010/);
assert.match(generatedCommand.textContent, /--allow-remote-training/);
assert.equal(firstRun.window.document.querySelector("#setup-command-warning").hidden, false);

console.log("Interface Fase 3: perfis, isolamento, wizard, tours e treinamento validados.");
