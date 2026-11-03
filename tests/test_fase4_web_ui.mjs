import assert from "node:assert/strict";
import fs from "node:fs";
import jsdom from "../frontend/node_modules/jsdom/lib/api.js";

const dom = new jsdom.JSDOM(fs.readFileSync("fase4/web/index.html", "utf8"), {
  url: "http://localhost:8020/#/painel", runScripts: "outside-only",
});
const { window } = dom;
const pacientes = [
  { paciente_id: "PAC-0006", origem: "sinteticos", arquivos: { video: "fisioterapia.mp4", keypoints: "fisioterapia.keypoints.json" }, audios: ["consulta.wav"], modalidades: ["video", "audio"] },
  { paciente_id: "UCF-JumpingJack_g01_c01", origem: "ucf101", arquivos: { video: "fisioterapia.mp4" }, audios: [], modalidades: ["video"] },
];
const pedidos = [];
window.F4 = { api: {
  get: async (url) => {
    if (url === "/api/status") return { backends: { mediapipe: true, yolov8: true, whisper: true }, servicos: { speech_to_text: { provedor_efetivo: "azure" } } };
    if (url === "/api/pacientes") return pacientes;
    return pacientes.find((p) => url === `/api/pacientes/${p.paciente_id}`);
  },
  post: async (url, body) => {
    pedidos.push({ url, body });
    return { modo: "sincrono", resultado: { paciente_id: "PAC-0006", fusao: {}, achados: [], alertas: [], relatorios: [], midias: {} } };
  },
} };
const tick = () => new Promise((resolve) => setTimeout(resolve, 10));
const select = (label) => [...window.document.querySelectorAll("label.campo")].find((e) => e.firstElementChild.textContent === label)?.querySelector("select");
async function escolher(label, value) {
  const campo = select(label);
  assert.ok(campo, label);
  campo.value = value;
  campo.dispatchEvent(new window.Event("change"));
  await tick();
}
try {
  for (const nome of ["render", "app"]) window.eval(fs.readFileSync(`fase4/web/${nome}.js`, "utf8"));
  await tick();
  await escolher("Paciente", pacientes[1].paciente_id);
  assert.equal(select("Estimador de pose").value, "mediapipe");
  assert.equal(select("Detector de objetos").value, "yolov8");
  assert.match(window.document.body.textContent, /UCF101/);
  await escolher("Paciente", "PAC-0006");
  assert.equal(select("Estimador de pose").value, "anotado");
  assert.equal(select("Detector de objetos").value, "anotado");
  assert.deepEqual([...select("Transcrição e linguagem").options].map((o) => o.value), ["offline", "auto", "local", "azure"]);
  for (const provedor of ["offline", "auto", "local", "azure"]) {
    await escolher("Transcrição e linguagem", provedor);
    await escolher("Estimador de pose", "mediapipe");
    await escolher("Detector de objetos", "anotado");
    window.document.querySelector("form.formulario").dispatchEvent(new window.Event("submit", { cancelable: true }));
    await tick();
    assert.equal(pedidos.at(-1).body.provedor_azure, provedor);
    assert.equal(pedidos.at(-1).body.backend_pose, "mediapipe");
    assert.equal(pedidos.at(-1).body.backend_detector, "anotado");
  }
  console.log("OK: UCF101, padroes sinteticos e envio das quatro opcoes de provedor preservados.");
} finally {
  window.close();
}
