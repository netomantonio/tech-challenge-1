const state = { patients: [], selected: null, status: null, sessions: new Map() };

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
};

const questionsByPatient = {
  "PAC-0001": ["Posso iniciar a quimioterapia hoje?", "Quais exames ainda impedem o primeiro ciclo?"],
  "PAC-0002": ["O checklist pre-tratamento esta completo?", "A paciente esta apta para iniciar o tratamento?"],
  "PAC-0003": ["Qual e a conduta para o BI-RADS 4?", "A biopsia precisa ser priorizada?"],
  "PAC-0004": ["O caso deve passar por equipe multidisciplinar?", "Quais cuidados o protocolo recomenda?"],
  "PAC-0005": ["A paciente esta com febre e taquicardia, qual conduta?", "Ha criterios para acionar o protocolo de sepse?"],
  "PAC-0006": ["Como conduzir a dor 7/10 no pos-operatorio?", "Essa dor exige reavaliacao da equipe cirurgica?"],
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
  const route = result.rota_exames === "com_pendencias" ? "Alertar exames pendentes" : "Seguir sem pendencias";
  ["Buscar paciente", "Verificar exames", route, "Recuperar protocolos", "Checar seguranca", "Registrar auditoria"].forEach((step) => {
    elements.flow.append(textElement("li", "", step));
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
initialize();
