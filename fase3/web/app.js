const state = {
  patients: [],
  selected: null,
  status: null,
  sessions: new Map(),
  training: null,
  selectedAdapter: null,
  trainingPoll: null,
  nextVersion: null,
};

const $ = (selector) => document.querySelector(selector);
const elements = {
  status: $("#runtime-status"),
  patientSelect: $("#patient-select"),
  patientSummary: $("#patient-summary"),
  pendingCount: $("#pending-count"),
  pendingList: $("#pending-list"),
  alertCount: $("#active-alert-count"),
  alertList: $("#active-alert-list"),
  suggestions: $("#suggestions"),
  form: $("#question-form"),
  question: $("#question"),
  characterCount: $("#character-count"),
  submit: $("#submit-button"),
  chatPatientBadge: $("#chat-patient-badge"),
  welcome: $("#welcome"),
  conversation: $("#conversation"),
  error: $("#error-message"),
  emptyEvidence: $("#empty-evidence"),
  evidence: $("#evidence-content"),
  flow: $("#flow-list"),
  sources: $("#source-list"),
  generatedAlerts: $("#generated-alerts"),
  clinicalView: $("#clinical-view"),
  trainingView: $("#training-view"),
  viewTabs: [...document.querySelectorAll("[data-view]")],
  promotedVersion: $("#promoted-version"),
  promotedScale: $("#promoted-scale"),
  datasetTrain: $("#dataset-train"),
  datasetValidation: $("#dataset-validation"),
  trainingForm: $("#training-form"),
  trainVersion: $("#train-version"),
  rebuildDataset: $("#rebuild-dataset"),
  startTraining: $("#start-training"),
  jobStatus: $("#job-status"),
  jobTitle: $("#job-title"),
  jobStage: $("#job-stage"),
  jobProgress: $("#job-progress"),
  jobLog: $("#job-log"),
  cancelJob: $("#cancel-job"),
  opsError: $("#ops-error"),
  adapterCount: $("#adapter-count"),
  adapterSelect: $("#adapter-select"),
  adapterDetail: $("#adapter-detail"),
  evaluateLoss: $("#evaluate-loss"),
  calibrateAdapter: $("#calibrate-adapter"),
  promoteAdapter: $("#promote-adapter"),
  gateResult: $("#gate-result"),
  metricAcceptance: $("#metric-acceptance"),
  metricFallback: $("#metric-fallback"),
  metricQuality: $("#metric-quality"),
  metricSafety: $("#metric-safety"),
  gateList: $("#gate-list"),
};

const questionsByPatient = {
  "PAC-0001": ["Posso iniciar a quimioterapia hoje?", "Quais exames ainda impedem o primeiro ciclo?"],
  "PAC-0002": ["O checklist pre-tratamento esta completo?", "A paciente esta apta para iniciar o tratamento?"],
  "PAC-0003": ["Qual e a conduta para o BI-RADS 4?", "A biopsia precisa ser priorizada?"],
  "PAC-0004": ["O caso deve passar por equipe multidisciplinar?", "Quais cuidados o protocolo recomenda?"],
  "PAC-0005": ["A paciente esta com febre e taquicardia, qual conduta?", "Ha criterios para acionar o protocolo de sepse?"],
  "PAC-0006": ["Como conduzir a dor 7/10 no pos-operatorio?", "Essa dor exige reavaliacao da equipe cirurgica?"],
};

const flowLabels = {
  buscar_paciente: "Buscar paciente",
  verificar_exames_pendentes: "Verificar exames pendentes",
  alertar_exames_pendentes: "Registrar alerta de exames pendentes",
  sugerir_tratamento: "Recuperar protocolos e gerar resposta",
  checar_seguranca: "Avaliar resultado dos guardrails",
  emitir_alertas: "Consolidar alertas no estado",
  registrar_auditoria: "Registrar auditoria",
};

function textElement(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = text;
  return node;
}

function fillList(container, items, emptyText) {
  container.replaceChildren();
  if (!items.length) {
    container.append(textElement("li", "empty-item", emptyText));
    return;
  }
  items.forEach((item) => container.append(textElement("li", "", item.replaceAll("_", " "))));
}

function getSession(patientId) {
  if (!state.sessions.has(patientId)) {
    state.sessions.set(patientId, { messages: [], lastResult: null, draft: "" });
  }
  return state.sessions.get(patientId);
}

function renderPatient(patient) {
  if (state.selected) {
    getSession(state.selected.paciente_id).draft = elements.question.value;
  }
  state.selected = patient;
  const heading = document.createElement("div");
  heading.className = "patient-id-line";
  heading.append(textElement("strong", "", patient.paciente_id));
  heading.append(textElement("span", "stage-badge", `Estagio ${patient.estagio}`));

  const diagnosis = textElement("p", "diagnosis", patient.diagnostico);
  const facts = document.createElement("div");
  facts.className = "patient-facts";
  [["Idade", `${patient.idade} anos`], ["Sexo", patient.sexo === "F" ? "Feminino" : "Masculino"]].forEach(([label, value]) => {
    const item = document.createElement("div");
    item.append(textElement("span", "", label), textElement("strong", "", value));
    facts.append(item);
  });
  const observation = textElement("p", "patient-observation", patient.observacoes);
  elements.patientSummary.replaceChildren(heading, diagnosis, facts, observation);

  elements.pendingCount.textContent = patient.exames_pendentes.length;
  elements.alertCount.textContent = patient.alertas_ativos.length;
  fillList(elements.pendingList, patient.exames_pendentes, "Nenhum exame pendente");
  fillList(elements.alertList, patient.alertas_ativos, "Nenhum alerta ativo");
  renderSuggestions(patient.paciente_id);
  renderSession(patient.paciente_id);
}

function renderSuggestions(patientId) {
  elements.suggestions.replaceChildren();
  (questionsByPatient[patientId] || ["Qual conduta o protocolo recomenda para este caso?"]).forEach((question) => {
    const button = textElement("button", "suggestion-button", question);
    button.type = "button";
    button.addEventListener("click", () => {
      elements.question.value = question;
      elements.characterCount.textContent = question.length;
      getSession(patientId).draft = question;
      elements.question.focus();
    });
    elements.suggestions.append(button);
  });
}

function appendMessage(kind, text, meta = "") {
  const message = document.createElement("article");
  message.className = `message message-${kind}`;
  message.append(textElement("p", "message-label", kind === "user" ? "Sua pergunta" : "Assistente clinico"));
  message.append(textElement("div", "message-body", text));
  if (meta) message.append(textElement("div", "message-meta", meta));
  elements.conversation.append(message);
  elements.conversation.classList.add("visible");
  return message;
}

function clearEvidence() {
  elements.emptyEvidence.hidden = false;
  elements.evidence.hidden = true;
  elements.flow.replaceChildren();
  elements.sources.replaceChildren();
  elements.generatedAlerts.replaceChildren();
}

function renderSession(patientId) {
  const session = getSession(patientId);
  elements.chatPatientBadge.textContent = `Conversa ${patientId}`;
  elements.question.value = session.draft;
  elements.characterCount.textContent = session.draft.length;
  elements.error.hidden = true;
  elements.conversation.replaceChildren();
  elements.conversation.classList.toggle("visible", session.messages.length > 0);
  elements.welcome.hidden = session.messages.length > 0;

  session.messages.forEach(({ kind, text, meta, blocked }) => {
    const message = appendMessage(kind, text, meta);
    if (blocked) message.classList.add("message-blocked");
  });

  if (session.lastResult) renderEvidence(session.lastResult);
  else clearEvidence();
}

function showLoading() {
  const message = document.createElement("article");
  message.className = "message message-assistant loading-message";
  message.append(textElement("p", "message-label", "Processando fluxo clinico"));
  const body = document.createElement("div");
  body.className = "message-body";
  body.append(document.createElement("i"), document.createElement("i"), document.createElement("i"));
  body.append(textElement("span", "", state.status?.backend === "local" ? "Consultando modelo local..." : "Consultando protocolos..."));
  message.append(body);
  elements.conversation.append(message);
  elements.conversation.classList.add("visible");
  return message;
}

function renderEvidence(result) {
  elements.emptyEvidence.hidden = true;
  elements.evidence.hidden = false;
  elements.flow.replaceChildren();
  (result.etapas_executadas || []).forEach((step) => {
    elements.flow.append(textElement("li", "", flowLabels[step] || step.replaceAll("_", " ")));
  });

  elements.sources.replaceChildren();
  result.fontes.forEach((source) => {
    const card = document.createElement("div");
    card.className = "source-card";
    card.append(textElement("strong", "", source.id), textElement("span", "", source.titulo));
    elements.sources.append(card);
  });
  if (!result.fontes.length) elements.sources.append(textElement("p", "empty-item", "Nenhuma fonte retornada."));

  fillList(elements.generatedAlerts, result.alertas, "Nenhum alerta adicional");
}

function humanizeMode(mode) {
  return ({ llm: "Resposta direta do modelo", citacao_reparada: "Citacao reparada", fallback: "Fallback seguro", bloqueada: "Resposta bloqueada" })[mode] || "Modo nao informado";
}

async function submitQuestion(event) {
  event.preventDefault();
  const question = elements.question.value.trim();
  if (!question || !state.selected) return;
  const patientId = state.selected.paciente_id;
  const session = getSession(patientId);

  elements.error.hidden = true;
  session.messages.push({ kind: "user", text: question, meta: "", blocked: false });
  session.draft = question;
  renderSession(patientId);
  const loading = showLoading();
  elements.submit.disabled = true;
  elements.patientSelect.disabled = true;
  elements.conversation.setAttribute("aria-busy", "true");

  try {
    const response = await fetch("/api/consultas", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paciente_id: patientId, pergunta: question }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : "Falha ao analisar o caso.");
    session.messages.push({
      kind: "assistant",
      text: body.resposta || "Paciente nao encontrado no prontuario.",
      meta: humanizeMode(body.modo_resposta),
      blocked: body.bloqueado,
    });
    session.lastResult = body;
    session.draft = "";
    renderSession(patientId);
  } catch (error) {
    loading.remove();
    elements.error.textContent = error instanceof Error ? error.message : "Nao foi possivel consultar o assistente.";
    elements.error.hidden = false;
  } finally {
    elements.submit.disabled = false;
    elements.patientSelect.disabled = false;
    elements.conversation.setAttribute("aria-busy", "false");
  }
}

const activeJobStatuses = new Set(["aguardando", "executando", "cancelando"]);
const stageByJobType = {
  dataset: "data",
  treino: "train",
  loss: "loss",
  calibracao: "calibration",
};

async function apiRequest(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: options.body ? { "Content-Type": "application/json", ...(options.headers || {}) } : options.headers,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : "A operacao nao pode ser concluida.";
    throw new Error(detail);
  }
  return body;
}

function adapterVersion(path = "") {
  const match = path.replaceAll("\\", "/").match(/qwen2\.5-1\.5b-v\d+/);
  return match ? match[0] : "Adapter nao identificado";
}

function percent(value) {
  return Number.isFinite(value) ? `${Math.round(value * 1000) / 10}%` : "-";
}

function decimal(value, digits = 4) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : "-";
}

function showOpsError(error) {
  elements.opsError.textContent = error instanceof Error ? error.message : String(error);
  elements.opsError.hidden = false;
}

function clearOpsError() {
  elements.opsError.hidden = true;
  elements.opsError.textContent = "";
}

function calibrationMatches(version, calibration) {
  return Boolean(
    calibration
      && adapterVersion(calibration.adapter_path) === version,
  );
}

function renderPipeline(overview) {
  const completed = {
    data: overview.dataset.ready,
    train: overview.adapters.length > 0,
    loss: overview.adapters.some((adapter) => Number.isFinite(adapter.validation_loss)),
    calibration: Boolean(overview.latest_calibration),
    promotion: Boolean(overview.promoted?.adapter_path),
  };
  const job = overview.job;
  const activeStage = job && activeJobStatuses.has(job.status) ? stageByJobType[job.tipo] : null;
  const failedStage = job?.status === "falhou" ? stageByJobType[job.tipo] : null;

  Object.keys(completed).forEach((stage) => {
    const node = $(`#stage-${stage}`);
    node.classList.toggle("complete", completed[stage]);
    node.classList.toggle("active", activeStage === stage);
    node.classList.toggle("failed", failedStage === stage);
  });
  $("#stage-data-detail").textContent = overview.dataset.ready
    ? `${overview.dataset.train} treino / ${overview.dataset.validation} validacao`
    : "Splits ainda nao preparados";
  $("#stage-train-detail").textContent = overview.adapters.length
    ? `${overview.adapters.length} adapter(s) salvo(s)`
    : "Nenhum adapter salvo";
}

function renderJob(job) {
  const active = Boolean(job && activeJobStatuses.has(job.status));
  const statusLabels = {
    aguardando: "Preparando",
    executando: "Executando",
    cancelando: "Cancelando",
    concluido: "Concluido",
    falhou: "Falhou",
    cancelado: "Cancelado",
  };
  const statusClass = !job
    ? "idle"
    : active
      ? "running"
      : job.status === "concluido"
        ? "success"
        : "failed";

  elements.jobStatus.className = `job-status ${statusClass}`;
  elements.jobStatus.textContent = job ? (statusLabels[job.status] || job.status) : "Sem operacao";
  elements.jobTitle.textContent = job?.titulo || "Aguardando comando";
  elements.jobStage.textContent = job?.etapa || "As etapas e logs aparecerao aqui.";
  elements.jobProgress.style.width = `${Math.round((job?.progresso || 0) * 100)}%`;
  elements.jobLog.textContent = job?.logs?.length
    ? job.logs.join("\n")
    : "Nenhum job executado nesta sessao.";
  elements.jobLog.scrollTop = elements.jobLog.scrollHeight;
  elements.cancelJob.disabled = !active;
}

function detailItem(label, value) {
  const item = document.createElement("div");
  item.append(textElement("span", "", label), textElement("strong", "", value));
  return item;
}

function renderAdapter(overview) {
  const adapters = overview.adapters;
  const available = new Set(adapters.map((adapter) => adapter.version));
  if (!state.selectedAdapter || !available.has(state.selectedAdapter)) {
    state.selectedAdapter = adapters[0]?.version || null;
  }

  elements.adapterCount.textContent = adapters.length;
  elements.adapterSelect.replaceChildren();
  adapters.forEach((adapter) => {
    const option = document.createElement("option");
    option.value = adapter.version;
    option.textContent = adapter.version;
    option.selected = adapter.version === state.selectedAdapter;
    elements.adapterSelect.append(option);
  });
  elements.adapterSelect.disabled = adapters.length === 0;

  const selected = adapters.find((adapter) => adapter.version === state.selectedAdapter);
  elements.adapterDetail.replaceChildren();
  if (!selected) {
    elements.adapterDetail.append(textElement("p", "empty-item", "Nenhum adapter treinado foi encontrado."));
  } else {
    elements.adapterDetail.append(
      detailItem("Epocas", selected.epochs ?? "Nao informado"),
      detailItem("Learning rate", selected.learning_rate ?? "Nao informado"),
      detailItem("Loss validacao", decimal(selected.validation_loss)),
      detailItem("Criado em", new Date(selected.created_at).toLocaleString("pt-BR")),
    );
  }

  const active = Boolean(overview.job && activeJobStatuses.has(overview.job.status));
  const canPromote = Boolean(
    selected
      && calibrationMatches(selected.version, overview.latest_calibration)
      && overview.latest_calibration.aprovado,
  );
  elements.evaluateLoss.disabled = active || !selected;
  elements.calibrateAdapter.disabled = active || !selected;
  elements.promoteAdapter.disabled = active || !canPromote;
  elements.rebuildDataset.disabled = active;
  elements.startTraining.disabled = active;
  [...elements.trainingForm.elements].forEach((control) => {
    if (control !== elements.startTraining) control.disabled = active;
  });
}

function humanizeGate(name) {
  const labels = {
    aceitacao_bruta_regular_minima_0_80: "Aceitacao bruta regular >= 80%",
    fallback_regular_maximo_0_20: "Fallback regular <= 20%",
    seguranca_final_1_00: "Seguranca final = 100%",
    adversariais_seguros_1_00: "Casos adversariais seguros = 100%",
    qualidade_final_1_00: "Qualidade final = 100%",
    melhoria_adapter_minima: "Melhoria minima sobre o modelo-base",
  };
  return labels[name] || name.replaceAll("_", " ");
}

function renderMetrics(calibration) {
  const approved = Boolean(calibration?.aprovado);
  elements.gateResult.className = `gate-result ${calibration ? (approved ? "approved" : "rejected") : ""}`;
  elements.gateResult.textContent = calibration ? (approved ? "Aprovado" : "Reprovado") : "Sem resultado";
  elements.metricAcceptance.textContent = percent(calibration?.taxa_aceitacao_bruta_regular);
  elements.metricFallback.textContent = percent(calibration?.taxa_fallback_regular);
  elements.metricQuality.textContent = percent(calibration?.score_qualidade_final);
  elements.metricSafety.textContent = percent(calibration?.score_seguranca_final);
  elements.gateList.replaceChildren();
  if (!calibration?.gates) {
    elements.gateList.append(textElement("p", "empty-item", "Execute a calibracao para validar os gates."));
    return;
  }
  Object.entries(calibration.gates).forEach(([name, passed]) => {
    const row = document.createElement("div");
    row.className = passed ? "pass" : "fail";
    row.append(textElement("span", "", humanizeGate(name)), textElement("strong", "", passed ? "ATENDIDO" : "PENDENTE"));
    elements.gateList.append(row);
  });
}

function renderTraining(overview) {
  state.training = overview;
  const promoted = overview.promoted || {};
  elements.promotedVersion.textContent = adapterVersion(promoted.adapter_path);
  elements.promotedScale.textContent = `Escala LoRA ${promoted.lora_scale ?? "-"}`;
  elements.datasetTrain.textContent = overview.dataset.train;
  elements.datasetValidation.textContent = overview.dataset.validation;
  if (!elements.trainVersion.value || elements.trainVersion.value === state.nextVersion) {
    elements.trainVersion.value = overview.next_version;
  }
  state.nextVersion = overview.next_version;
  renderPipeline(overview);
  renderJob(overview.job);
  renderAdapter(overview);
  renderMetrics(
    calibrationMatches(state.selectedAdapter, overview.latest_calibration)
      ? overview.latest_calibration
      : null,
  );
}

function scheduleTrainingPoll() {
  clearTimeout(state.trainingPoll);
  const active = Boolean(state.training?.job && activeJobStatuses.has(state.training.job.status));
  if (!active) return;
  state.trainingPoll = setTimeout(() => loadTrainingOverview(), 1500);
}

async function loadTrainingOverview(clearPreviousError = true) {
  try {
    const overview = await apiRequest("/api/treinamento");
    renderTraining(overview);
    if (clearPreviousError) clearOpsError();
    scheduleTrainingPoll();
  } catch (error) {
    showOpsError(error);
  }
}

async function runTrainingOperation(path, payload) {
  clearOpsError();
  try {
    const options = { method: "POST" };
    if (payload) options.body = JSON.stringify(payload);
    const result = await apiRequest(path, options);
    if (result.id) {
      renderJob(result);
      state.training = { ...state.training, job: result };
    }
    await loadTrainingOverview();
  } catch (error) {
    showOpsError(error);
    await loadTrainingOverview(false);
  }
}

function switchView(view) {
  const training = view === "training";
  elements.clinicalView.hidden = training;
  elements.trainingView.hidden = !training;
  elements.viewTabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.view === view));
  if (training) loadTrainingOverview();
}

function trainingPayload() {
  const data = new FormData(elements.trainingForm);
  return {
    version: data.get("version"),
    epochs: Number(data.get("epochs")),
    batch_size: Number(data.get("batch_size")),
    gradient_accumulation_steps: Number(data.get("gradient_accumulation_steps")),
    learning_rate: Number(data.get("learning_rate")),
    max_length: Number(data.get("max_length")),
    seed: Number(data.get("seed")),
    lora_r: Number(data.get("lora_r")),
    lora_alpha: Number(data.get("lora_alpha")),
    lora_dropout: Number(data.get("lora_dropout")),
  };
}

async function initialize() {
  try {
    const [statusResponse, patientsResponse] = await Promise.all([fetch("/api/status"), fetch("/api/pacientes")]);
    if (!statusResponse.ok || !patientsResponse.ok) throw new Error("Servico indisponivel");
    state.status = await statusResponse.json();
    state.patients = await patientsResponse.json();
    elements.status.classList.add("ready");
    const loadNote = state.status.modelo_carregado ? "modelo carregado" : "carrega na primeira consulta";
    elements.status.querySelector("span:last-child").textContent = `${state.status.backend.toUpperCase()} | ${loadNote}`;

    elements.patientSelect.replaceChildren();
    state.patients.forEach((patient) => {
      const option = document.createElement("option");
      option.value = patient.paciente_id;
      option.textContent = `${patient.paciente_id} | ${patient.diagnostico}`;
      elements.patientSelect.append(option);
    });
    if (state.patients.length) renderPatient(state.patients[0]);
  } catch (error) {
    elements.status.classList.add("error");
    elements.status.querySelector("span:last-child").textContent = "Servico indisponivel";
    elements.error.textContent = "Nao foi possivel carregar os dados da aplicacao.";
    elements.error.hidden = false;
  }
}

elements.patientSelect.addEventListener("change", () => {
  const patient = state.patients.find((item) => item.paciente_id === elements.patientSelect.value);
  if (patient) renderPatient(patient);
});
elements.question.addEventListener("input", () => {
  elements.characterCount.textContent = elements.question.value.length;
  if (state.selected) getSession(state.selected.paciente_id).draft = elements.question.value;
});
elements.form.addEventListener("submit", submitQuestion);
elements.viewTabs.forEach((tab) => tab.addEventListener("click", () => switchView(tab.dataset.view)));
elements.trainingForm.addEventListener("submit", (event) => {
  event.preventDefault();
  runTrainingOperation("/api/treinamento/iniciar", trainingPayload());
});
elements.rebuildDataset.addEventListener("click", () => runTrainingOperation("/api/treinamento/dataset"));
elements.cancelJob.addEventListener("click", () => runTrainingOperation("/api/treinamento/cancelar"));
elements.adapterSelect.addEventListener("change", () => {
  state.selectedAdapter = elements.adapterSelect.value;
  if (state.training) {
    renderAdapter(state.training);
    renderMetrics(
      calibrationMatches(state.selectedAdapter, state.training.latest_calibration)
        ? state.training.latest_calibration
        : null,
    );
  }
});
elements.evaluateLoss.addEventListener("click", () => {
  if (state.selectedAdapter) runTrainingOperation("/api/treinamento/loss", { version: state.selectedAdapter });
});
elements.calibrateAdapter.addEventListener("click", () => {
  if (state.selectedAdapter) {
    runTrainingOperation("/api/treinamento/calibrar", {
      version: state.selectedAdapter,
      scales: [0.25, 0.5, 0.75, 1.0],
    });
  }
});
elements.promoteAdapter.addEventListener("click", async () => {
  if (!state.selectedAdapter) return;
  await runTrainingOperation("/api/treinamento/promover", { version: state.selectedAdapter });
  try {
    state.status = await apiRequest("/api/status");
  } catch (error) {
    showOpsError(error);
  }
});
initialize();
