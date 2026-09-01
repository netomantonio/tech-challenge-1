const state = {
  patients: [],
  selected: null,
  status: null,
  sessions: new Map(),
  training: null,
  selectedAdapter: null,
  trainingPoll: null,
  nextVersion: null,
  profiles: [],
  activeProfile: null,
  capabilities: null,
  setupStep: 0,
  setupValidated: null,
  tour: null,
  editingProfile: null,
};

const STORAGE_KEYS = {
  profiles: "fase3.backendProfiles.v1",
  active: "fase3.activeBackend.v1",
  tours: "fase3.completedTours.v1",
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
  profileSelect: $("#profile-select"),
  manageProfiles: $("#manage-profiles"),
  helpButton: $("#help-button"),
  activeModelBadge: $("#active-model-badge"),
  apiDocsLink: $("#api-docs-link"),
  setupDialog: $("#setup-dialog"),
  setupForm: $("#setup-form"),
  setupBack: $("#setup-back"),
  setupNext: $("#setup-next"),
  setupName: $("#setup-name"),
  setupUrl: $("#setup-url"),
  setupInstanceName: $("#setup-instance-name"),
  setupBindHost: $("#setup-bind-host"),
  setupPort: $("#setup-port"),
  setupRemoteTraining: $("#setup-remote-training"),
  setupOpenBrowser: $("#setup-open-browser"),
  setupCommand: $("#setup-command"),
  setupCommandWarning: $("#setup-command-warning"),
  setupNetworkHint: $("#setup-network-hint"),
  setupSteps: [...document.querySelectorAll("#setup-steps li")],
  setupPages: [...document.querySelectorAll(".setup-page")],
  testConnection: $("#test-connection"),
  connectionDiagnostic: $("#connection-diagnostic"),
  setupModel: $("#setup-model"),
  installSetupModel: $("#install-setup-model"),
  modelDiagnostic: $("#model-diagnostic"),
  setupSummary: $("#setup-summary"),
  copyCommand: $("#copy-command"),
  profilesDialog: $("#profiles-dialog"),
  profilesList: $("#profiles-list"),
  closeProfiles: $("#close-profiles"),
  newProfile: $("#new-profile"),
  resetProfiles: $("#reset-profiles"),
  trainModel: $("#train-model"),
  trainPrecision: $("#train-precision"),
  hardwareSummary: $("#hardware-summary"),
  modelCount: $("#model-count"),
  modelList: $("#model-list"),
  modelForm: $("#model-form"),
  tourLayer: $("#tour-layer"),
  tourFocus: $("#tour-focus"),
  tourPopover: $("#tour-popover"),
  tourProgress: $("#tour-progress"),
  tourTitle: $("#tour-title"),
  tourText: $("#tour-text"),
  tourNext: $("#tour-next"),
  tourClose: $("#tour-close"),
  fieldHelpTooltip: $("#field-help-tooltip"),
};

const fieldHelpDefinitions = [
  {
    selector: "#train-model",
    title: "Modelo-base",
    text: "Define os pesos sobre os quais o LoRA sera treinado. O adapter resultante fica vinculado a esse modelo e nao deve ser promovido sobre outra base. Modelos maiores podem elevar a capacidade, mas exigem mais memoria, armazenamento e tempo.",
  },
  {
    selector: "#train-precision",
    title: "Precisao",
    text: "Controla como pesos e calculos ocupam memoria. FP32 prioriza compatibilidade, FP16/BF16 reduzem memoria e NF4 usa QLoRA para economizar ainda mais VRAM. A opcao automatica escolhe conforme o hardware; formatos inadequados podem falhar ou deixar o treino mais lento.",
  },
  {
    selector: "#train-version",
    title: "Versao de saida",
    text: "Identifica de forma unica o novo adapter e seus artefatos. Deve seguir o alias do modelo com o sufixo -vN para manter rastreabilidade entre treino, avaliacao e promocao. O nome nao altera a qualidade do modelo.",
  },
  {
    selector: "#train-epochs",
    title: "Epocas",
    text: "Numero de passagens completas pelo conjunto de treino. Mais epocas aumentam tempo e adaptacao aos exemplos, mas em excesso podem causar sobreajuste. Deve ser analisado junto do learning rate e da loss de validacao.",
  },
  {
    selector: "#train-lr",
    title: "Learning rate",
    text: "Define o tamanho de cada atualizacao dos parametros LoRA. Valores altos aprendem mais rapido, mas podem desestabilizar o treino; valores baixos sao mais conservadores e podem exigir mais epocas. Loss de treino e validacao orientam o ajuste.",
  },
  {
    selector: "#train-batch",
    title: "Batch",
    text: "Quantidade de exemplos processados simultaneamente por passo. Batches maiores podem estabilizar o gradiente, mas consomem mais memoria. O batch efetivo e batch multiplicado pela acumulacao.",
  },
  {
    selector: "#train-accumulation",
    title: "Acumulacao",
    text: "Soma gradientes de varios microbatches antes de atualizar o modelo. Permite simular um batch efetivo maior sem ocupar toda a memoria de uma vez, com algum aumento no tempo entre atualizacoes.",
  },
  {
    selector: "#train-length",
    title: "Sequencia",
    text: "Limite de tokens por exemplo de treinamento. Sequencias menores economizam memoria e tempo, mas podem truncar contexto clinico; sequencias maiores preservam mais texto e elevam o custo, especialmente na atencao do modelo.",
  },
  {
    selector: "#train-seed",
    title: "Seed",
    text: "Controla a aleatoriedade da inicializacao, embaralhamento e amostragem. Reutilizar a mesma seed com dados e parametros iguais facilita reproduzir e comparar experimentos. Uma seed diferente nao e, por si so, melhor.",
  },
  {
    selector: "#train-r",
    title: "LoRA r",
    text: "Define o rank das matrizes adaptadoras e, portanto, sua capacidade e quantidade de parametros treinaveis. Valores maiores podem representar ajustes mais complexos, mas usam mais memoria e podem sobreajustar. Atua em conjunto com LoRA alpha.",
  },
  {
    selector: "#train-alpha",
    title: "LoRA alpha",
    text: "Escala a contribuicao aprendida pelo adapter, normalmente em relacao ao rank r. A razao alpha/r influencia a intensidade efetiva das atualizacoes; valores excessivos podem tornar o ajuste agressivo e instavel.",
  },
  {
    selector: "#train-dropout",
    title: "Dropout",
    text: "Desativa aleatoriamente parte do caminho LoRA durante o treino para reduzir sobreajuste. Pode ajudar em datasets pequenos; valores altos demais dificultam o aprendizado. Nao e aplicado da mesma forma durante a inferencia.",
  },
  {
    selector: "#train-checkpointing",
    title: "Gradient checkpointing",
    text: "Economiza VRAM ao recalcular ativacoes durante o backward em vez de mante-las todas na memoria. Facilita treinar modelos ou sequencias maiores, mas deixa cada passo mais lento. Nao muda sozinho a arquitetura do adapter.",
  },
  {
    selector: "#model-alias",
    title: "Alias do modelo",
    text: "Chave tecnica unica usada em versoes de treino, manifests, adapters e promocao. Escolha um identificador curto e estavel; altera-lo cria outra identidade logica, mesmo quando a origem dos pesos e a mesma.",
  },
  {
    selector: "#model-label",
    title: "Nome do modelo",
    text: "Nome amigavel exibido na interface. Ajuda o grupo a reconhecer arquitetura e tamanho, mas nao interfere no download, no treinamento ou na qualidade gerada.",
  },
  {
    selector: "#model-source-type",
    title: "Origem",
    text: "Indica se os pesos serao obtidos do Hugging Face ou de um diretorio local autorizado. Essa escolha determina como a origem e validada e instalada; os demais parametros de treino continuam os mesmos.",
  },
  {
    selector: "#model-source",
    title: "Repo ID ou caminho",
    text: "Local exato dos pesos e tokenizer. Repositorios precisam ser compativeis com AutoModelForCausalLM; caminhos locais devem estar dentro de FASE3_MODEL_ROOTS. A arquitetura escolhida define memoria, velocidade e modulos disponiveis para LoRA.",
  },
  {
    selector: "#model-revision",
    title: "Revisao fixa",
    text: "Commit ou revisao imutavel dos arquivos do modelo. Fixar a revisao melhora reproducibilidade e protege contra mudancas futuras no repositorio. E obrigatoria quando codigo remoto for autorizado.",
  },
  {
    selector: "#model-target-modules",
    title: "Modulos LoRA",
    text: "Camadas lineares que receberao os adapters, separadas por virgula. Em branco, o sistema detecta os modulos pela arquitetura e usa all-linear como fallback. Uma selecao incompleta limita a adaptacao; nomes invalidos fazem o treino falhar.",
  },
  {
    selector: "#model-trust-remote-code",
    title: "Autorizar codigo remoto",
    text: "Permite executar implementacoes Python fornecidas pelo repositorio do modelo quando Transformers nao possui suporte nativo. Use apenas com fonte confiavel e revisao fixa, pois esse codigo roda na infraestrutura do backend. Nao melhora o modelo quando nao e necessario.",
  },
  {
    selector: "#adapter-select",
    title: "Adapter para avaliar",
    text: "Escolhe qual versao sera usada nas operacoes de loss, calibracao, gates e promocao. A selecao nao inicia treino nem altera o adapter; ela direciona os proximos comandos. Confirme se a versao pertence ao modelo-base esperado.",
  },
];

let activeFieldHelp = null;
let fieldHelpPinned = false;
let fieldHelpHideTimer = null;

function positionFieldHelp(button) {
  const tooltip = elements.fieldHelpTooltip;
  const trigger = button.getBoundingClientRect();
  const box = tooltip.getBoundingClientRect();
  const padding = 10;
  const left = Math.min(Math.max(padding, trigger.left), innerWidth - box.width - padding);
  let top = trigger.bottom + 7;
  if (top + box.height > innerHeight - padding) top = Math.max(padding, trigger.top - box.height - 7);
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;
}

function showFieldHelp(button, definition, pinned = false) {
  clearTimeout(fieldHelpHideTimer);
  if (activeFieldHelp && activeFieldHelp !== button) activeFieldHelp.setAttribute("aria-expanded", "false");
  activeFieldHelp = button;
  fieldHelpPinned = pinned;
  elements.fieldHelpTooltip.querySelector("strong").textContent = definition.title;
  elements.fieldHelpTooltip.querySelector("p").textContent = definition.text;
  elements.fieldHelpTooltip.hidden = false;
  button.setAttribute("aria-expanded", "true");
  positionFieldHelp(button);
}

function hideFieldHelp(force = false) {
  if (fieldHelpPinned && !force) return;
  clearTimeout(fieldHelpHideTimer);
  if (activeFieldHelp) activeFieldHelp.setAttribute("aria-expanded", "false");
  activeFieldHelp = null;
  fieldHelpPinned = false;
  elements.fieldHelpTooltip.hidden = true;
}

function scheduleFieldHelpHide() {
  clearTimeout(fieldHelpHideTimer);
  fieldHelpHideTimer = setTimeout(() => hideFieldHelp(), 120);
}

function initializeFieldHelp() {
  fieldHelpDefinitions.forEach((definition, index) => {
    const field = document.querySelector(definition.selector);
    const label = field?.closest("label") || (field?.id ? document.querySelector(`label[for="${field.id}"]`) : null);
    const title = label?.querySelector(":scope > span");
    if (!field || !label || !title) return;
    label.classList.add("has-field-help");
    title.classList.add("field-title");
    const button = document.createElement("button");
    button.className = "field-help-button";
    button.type = "button";
    button.textContent = "?";
    button.setAttribute("aria-label", `Ajuda sobre ${definition.title}`);
    button.setAttribute("aria-describedby", elements.fieldHelpTooltip.id);
    button.setAttribute("aria-expanded", "false");
    button.dataset.helpIndex = String(index);
    button.addEventListener("pointerenter", () => showFieldHelp(button, definition));
    button.addEventListener("pointerleave", scheduleFieldHelpHide);
    button.addEventListener("focus", () => showFieldHelp(button, definition));
    button.addEventListener("blur", scheduleFieldHelpHide);
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (activeFieldHelp === button && fieldHelpPinned) hideFieldHelp(true);
      else showFieldHelp(button, definition, true);
    });
    button.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        hideFieldHelp(true);
        button.focus();
      }
    });
    title.append(button);
  });
  elements.fieldHelpTooltip.addEventListener("pointerenter", () => clearTimeout(fieldHelpHideTimer));
  elements.fieldHelpTooltip.addEventListener("pointerleave", scheduleFieldHelpHide);
  document.addEventListener("pointerdown", (event) => {
    if (fieldHelpPinned && !elements.fieldHelpTooltip.contains(event.target) && !activeFieldHelp?.contains(event.target)) hideFieldHelp(true);
  });
  addEventListener("resize", () => { if (activeFieldHelp) positionFieldHelp(activeFieldHelp); });
  addEventListener("scroll", () => hideFieldHelp(true), true);
}

function classifyBackendUrl(value) {
  let url;
  try { url = new URL(value); } catch { return { type: "invalid", reason: "URL invalida" }; }
  if (!['http:', 'https:'].includes(url.protocol)) return { type: "invalid", reason: "Use HTTP ou HTTPS" };
  const host = url.hostname.toLowerCase().replace(/^\[|\]$/g, "");
  const parts = host.split(".").map(Number);
  const ipv4 = parts.length === 4 && parts.every((part) => Number.isInteger(part) && part >= 0 && part <= 255);
  const loopback = host === "localhost" || host === "::1" || (ipv4 && parts[0] === 127);
  if (loopback) return { type: "loopback", url: url.origin };
  const privateIp = ipv4 && (
    parts[0] === 10
    || (parts[0] === 172 && parts[1] >= 16 && parts[1] <= 31)
    || (parts[0] === 192 && parts[1] === 168)
    || (parts[0] === 169 && parts[1] === 254)
    || (parts[0] === 100 && parts[1] >= 64 && parts[1] <= 127)
  );
  const privateHost = privateIp || host.endsWith(".local") || host === "0.0.0.0" || host.startsWith("fe80:") || host.startsWith("fc") || host.startsWith("fd");
  if (privateHost) return { type: "private", url: url.origin };
  if (url.protocol !== "https:") return { type: "invalid", reason: "Backend publico exige HTTPS" };
  return { type: "public", url: url.origin };
}

window.fase3ClassifyBackendUrl = classifyBackendUrl;

function loadProfiles() {
  try { state.profiles = JSON.parse(localStorage.getItem(STORAGE_KEYS.profiles) || "[]"); }
  catch { state.profiles = []; }
  const activeId = localStorage.getItem(STORAGE_KEYS.active);
  state.activeProfile = state.profiles.find((profile) => profile.id === activeId) || state.profiles[0] || null;
}

function persistProfiles() {
  localStorage.setItem(STORAGE_KEYS.profiles, JSON.stringify(state.profiles));
  if (state.activeProfile) localStorage.setItem(STORAGE_KEYS.active, state.activeProfile.id);
  else localStorage.removeItem(STORAGE_KEYS.active);
}

function renderProfiles() {
  elements.profileSelect.replaceChildren();
  state.profiles.forEach((profile) => {
    const option = document.createElement("option");
    option.value = profile.id;
    option.textContent = profile.name;
    option.selected = profile.id === state.activeProfile?.id;
    elements.profileSelect.append(option);
  });
  elements.profileSelect.disabled = state.profiles.length < 2;
  elements.profilesList.replaceChildren();
  state.profiles.forEach((profile) => {
    const row = document.createElement("div");
    row.className = "profile-row";
    row.append(detailItem(profile.name, `${profile.baseUrl} | ${profile.networkType}`));
    const test = textElement("button", "secondary-button", "Testar");
    test.type = "button";
    test.addEventListener("click", async () => {
      state.activeProfile = profile;
      persistProfiles();
      renderProfiles();
      await initializeBackend();
    });
    const edit = textElement("button", "secondary-button", "Editar");
    edit.type = "button";
    edit.addEventListener("click", () => { elements.profilesDialog.close(); openSetup(profile); });
    const actions = document.createElement("div");
    actions.className = "profile-actions";
    actions.append(test, edit);
    row.append(actions);
    elements.profilesList.append(row);
  });
}

function requestInitFor(profile, options) {
  const init = { ...options };
  if (profile.networkType === "loopback") init.targetAddressSpace = "loopback";
  if (profile.networkType === "private") init.targetAddressSpace = "local";
  return init;
}

async function profileFetch(path, options = {}, profile = state.activeProfile) {
  if (!profile) throw new Error("Configure um backend antes de continuar.");
  const url = `${profile.baseUrl}${path}`;
  const init = requestInitFor(profile, options);
  try {
    return await fetch(url, init);
  } catch (firstError) {
    const method = String(init.method || "GET").toUpperCase();
    if (!("targetAddressSpace" in init) || method !== "GET") throw firstError;
    const { targetAddressSpace, ...fallbackInit } = init;
    return fetch(url, fallbackInit);
  }
}

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

function formatStageLabel(stage) {
  const normalized = String(stage || "").trim().toLowerCase().replaceAll("_", " ");
  if (!normalized || ["a definir", "nao definido"].includes(normalized)) {
    return "Estagio em definicao";
  }
  return `Estagio ${stage}`;
}

function getSession(patientId) {
  const key = `${state.activeProfile?.id || "unconfigured"}:${patientId}`;
  if (!state.sessions.has(key)) {
    state.sessions.set(key, { messages: [], lastResult: null, draft: "" });
  }
  return state.sessions.get(key);
}

function renderPatient(patient) {
  if (state.selected) {
    getSession(state.selected.paciente_id).draft = elements.question.value;
  }
  state.selected = patient;
  const heading = document.createElement("div");
  heading.className = "patient-id-line";
  heading.append(textElement("strong", "", patient.paciente_id));
  heading.append(textElement("span", "stage-badge", formatStageLabel(patient.estagio)));

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
    const response = await profileFetch("/api/consultas", {
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
  const response = await profileFetch(path, {
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
  const match = path.replaceAll("\\", "/").match(/[a-z0-9][a-z0-9._-]{1,63}-v\d+/);
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
  const previousModel = elements.trainModel.value;
  elements.trainModel.replaceChildren();
  (overview.models || state.capabilities?.models || []).forEach((model) => {
    const option = document.createElement("option");
    option.value = model.alias;
    option.textContent = `${model.label}${model.installed ? "" : " (nao instalado)"}`;
    option.selected = model.alias === previousModel || (!previousModel && model.alias === "qwen2.5-1.5b");
    elements.trainModel.append(option);
  });
  const modelAlias = elements.trainModel.value || "qwen2.5-1.5b";
  const nextVersion = overview.next_versions?.[modelAlias] || overview.next_version;
  if (!elements.trainVersion.value || elements.trainVersion.value === state.nextVersion) {
    elements.trainVersion.value = nextVersion;
  }
  state.nextVersion = nextVersion;
  const hardware = overview.hardware || state.capabilities?.hardware || {};
  elements.hardwareSummary.textContent = hardware.device === "cuda"
    ? `${hardware.device_name} | ${Math.round((hardware.memory_bytes || 0) / 1073741824)} GB`
    : hardware.device_name || "CPU";
  [...elements.trainPrecision.options].forEach((option) => {
    option.disabled = !(hardware.quantization || ["auto", "fp32"]).includes(option.value);
  });
  renderModelCatalog(overview.models || []);
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
  if (training) {
    loadTrainingOverview();
    setTimeout(() => maybeStartTour("training"), 50);
  }
}

function trainingPayload() {
  const data = new FormData(elements.trainingForm);
  return {
    version: data.get("version"),
    model_alias: data.get("model_alias"),
    precision: data.get("precision"),
    gradient_checkpointing: data.get("gradient_checkpointing") === "on",
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

function renderModelCatalog(models) {
  elements.modelCount.textContent = models.length;
  elements.modelList.replaceChildren();
  models.forEach((model) => {
    const row = document.createElement("div");
    row.className = "model-row";
    const description = document.createElement("div");
    description.append(textElement("strong", "", model.label), textElement("span", "", `${model.source} | ${model.installed ? "pronto" : "nao instalado"}`));
    const actions = document.createElement("div");
    if (!model.installed) {
      const install = textElement("button", "secondary-button", "Instalar");
      install.type = "button";
      install.addEventListener("click", () => runTrainingOperation("/api/modelos/instalar", { alias: model.alias }));
      actions.append(install);
    }
    if (!model.builtin) {
      const remove = textElement("button", "danger-button", "Remover");
      remove.type = "button";
      remove.addEventListener("click", async () => {
        try { await apiRequest(`/api/modelos/${encodeURIComponent(model.alias)}`, { method: "DELETE" }); await loadTrainingOverview(); }
        catch (error) { showOpsError(error); }
      });
      actions.append(remove);
    }
    row.append(description, actions);
    elements.modelList.append(row);
  });
}

function populatePatients() {
  elements.patientSelect.replaceChildren();
  state.patients.forEach((patient) => {
    const option = document.createElement("option");
    option.value = patient.paciente_id;
    option.textContent = `${patient.paciente_id} | ${patient.diagnostico}`;
    elements.patientSelect.append(option);
  });
  if (state.patients.length) renderPatient(state.patients[0]);
}

async function initializeBackend() {
  try {
    elements.status.className = "runtime-status";
    elements.status.querySelector("span:last-child").textContent = "Conectando";
    const [capabilities, status, patients] = await Promise.all([
      apiRequest("/api/capabilities"), apiRequest("/api/status"), apiRequest("/api/pacientes"),
    ]);
    if (Number(String(capabilities.api_version).split(".")[0]) !== 2) throw new Error("Versao da API incompatível. Esperada: 2.x.");
    state.capabilities = capabilities;
    state.status = status;
    state.patients = patients;
    state.selected = null;
    state.training = null;
    state.activeProfile.lastCapabilities = capabilities;
    persistProfiles();
    elements.status.className = "runtime-status ready";
    const loadNote = state.status.modelo_carregado ? "modelo carregado" : "carrega na primeira consulta";
    elements.status.querySelector("span:last-child").textContent = `${capabilities.instance_name} | ${loadNote}`;
    elements.activeModelBadge.textContent = capabilities.promoted?.model_alias || status.modelo_base || "Modelo configurado";
    elements.apiDocsLink.href = `${state.activeProfile.baseUrl}/docs`;
    populatePatients();
    renderProfiles();
  } catch (error) {
    elements.status.className = "runtime-status error";
    elements.status.querySelector("span:last-child").textContent = "Servico indisponivel";
    elements.error.textContent = error instanceof Error ? error.message : "Nao foi possivel carregar os dados da aplicacao.";
    elements.error.hidden = false;
  }
}

function selectedTopology() {
  return elements.setupForm.querySelector('input[name="topology"]:checked')?.value || "loopback";
}

function startupConfigFromForm() {
  return {
    instanceName: elements.setupInstanceName.value.trim() || elements.setupName.value.trim() || "Backend Fase 3",
    bindHost: elements.setupBindHost.value,
    port: Math.min(65535, Math.max(1, Number(elements.setupPort.value) || 8010)),
    allowRemoteTraining: elements.setupRemoteTraining.checked,
    openBrowser: elements.setupOpenBrowser.checked,
  };
}

function quoteCommandArgument(value) {
  if (/^[a-zA-Z0-9._:/-]+$/.test(value)) return value;
  return JSON.stringify(value);
}

function buildSetupCommand() {
  const config = startupConfigFromForm();
  const args = ["python", "-m", "fase3"];
  if (config.bindHost !== "127.0.0.1") args.push("--host", config.bindHost);
  if (config.port !== 8010) args.push("--port", String(config.port));
  if (location.origin.startsWith("http")) args.push("--allowed-origin", location.origin);
  if (config.allowRemoteTraining) args.push("--allow-remote-training");
  if (config.instanceName) args.push("--instance-name", config.instanceName);
  if (!config.openBrowser) args.push("--no-open-browser");
  return args.map(quoteCommandArgument).join(" ");
}

window.fase3BuildStartCommand = buildSetupCommand;

function updateSetupCommand() {
  elements.setupCommand.textContent = buildSetupCommand();
  const networkExposed = elements.setupBindHost.value === "0.0.0.0";
  const remoteTraining = elements.setupRemoteTraining.checked;
  elements.setupCommandWarning.hidden = !networkExposed && !remoteTraining;
  elements.setupCommandWarning.textContent = remoteTraining
    ? "Atencao: outras maquinas com acesso a este backend poderao iniciar downloads, treinamentos e promocoes sem autenticacao."
    : networkExposed
      ? "O backend aceitara conexoes da rede. Mantenha o treinamento remoto desativado se ele nao for necessario."
      : "";
}

function syncDirectUrlPort() {
  if (selectedTopology() === "public") return;
  try {
    const url = new URL(elements.setupUrl.value.trim());
    url.port = String(startupConfigFromForm().port);
    elements.setupUrl.value = url.origin;
  } catch { /* A validacao da URL informa o erro ao avancar. */ }
}

function syncPortFromDirectUrl() {
  if (selectedTopology() === "public") return;
  try {
    const url = new URL(elements.setupUrl.value.trim());
    elements.setupPort.value = url.port || (url.protocol === "https:" ? "443" : "80");
  } catch { return; }
  updateSetupCommand();
}

function configureTopologyFields(resetValues = true) {
  const topology = selectedTopology();
  const defaults = {
    loopback: ["Backend local", "http://127.0.0.1:8010"],
    private: ["Backend da rede", "http://192.168.1.10:8010"],
    public: ["Backend remoto", "https://backend.exemplo.com"],
  };
  if (resetValues) {
    [elements.setupName.value, elements.setupUrl.value] = defaults[topology];
    elements.setupInstanceName.value = defaults[topology][0];
    elements.setupInstanceName.dataset.edited = "";
    elements.setupBindHost.value = topology === "private" ? "0.0.0.0" : "127.0.0.1";
    elements.setupPort.value = "8010";
    elements.setupRemoteTraining.checked = false;
    elements.setupOpenBrowser.checked = false;
  }
  updateSetupCommand();
  elements.setupNetworkHint.textContent = topology === "loopback"
    ? "O navegador pode solicitar permissao para acessar esta maquina."
    : topology === "private"
      ? "As maquinas precisam estar na mesma rede ou VPN; libere a porta 8010 no firewall."
      : "Use HTTPS. Sem uma rota publica, use VPN ou Cloudflare Tunnel.";
}

function renderSetupStep() {
  elements.setupPages.forEach((page, index) => { page.hidden = index !== state.setupStep; });
  elements.setupSteps.forEach((step, index) => {
    step.classList.toggle("active", index === state.setupStep);
    step.classList.toggle("complete", index < state.setupStep);
  });
  elements.setupBack.disabled = state.setupStep === 0;
  elements.setupNext.textContent = state.setupStep === 4 ? "Abrir consulta" : "Continuar";
  if (state.setupStep === 1) updateSetupCommand();
  if (state.setupStep === 4 && state.setupValidated) {
    elements.setupSummary.textContent = `${state.setupValidated.capabilities.instance_name} em ${state.setupValidated.profile.baseUrl}. O perfil e os chats ficam isolados neste navegador.`;
  }
}

function openSetup(profile = null) {
  state.editingProfile = profile;
  state.setupStep = profile ? 1 : 0;
  state.setupValidated = null;
  if (profile) {
    const radio = elements.setupForm.querySelector(`input[name="topology"][value="${profile.networkType}"]`);
    if (radio) radio.checked = true;
  }
  if (profile) {
    elements.setupName.value = profile.name;
    elements.setupUrl.value = profile.baseUrl;
    const startup = profile.startup || {};
    elements.setupInstanceName.value = startup.instanceName || profile.name;
    elements.setupInstanceName.dataset.edited = "true";
    elements.setupBindHost.value = startup.bindHost || (profile.networkType === "private" ? "0.0.0.0" : "127.0.0.1");
    elements.setupPort.value = String(startup.port || 8010);
    elements.setupRemoteTraining.checked = Boolean(startup.allowRemoteTraining);
    elements.setupOpenBrowser.checked = Boolean(startup.openBrowser);
    configureTopologyFields(false);
  } else {
    configureTopologyFields(true);
  }
  renderSetupStep();
  if (!elements.setupDialog.open) elements.setupDialog.showModal();
}

function connectionErrorMessage(error, classification) {
  const message = error instanceof Error ? error.message : String(error);
  if (location.protocol === "https:" && classification?.type === "public" && classification.url.startsWith("http:")) return "Mixed content: um site HTTPS nao pode chamar este backend publico por HTTP.";
  if (/aborted|timeout/i.test(message)) return "Tempo esgotado. Verifique endereco, rota VPN e firewall.";
  if (/failed to fetch|networkerror|fetch/i.test(message)) return "Sem resposta legivel. Verifique se o backend esta ativo, a porta, o firewall, a permissao de rede local e FASE3_ALLOWED_ORIGINS.";
  return message;
}

async function testSetupConnection() {
  const classification = classifyBackendUrl(elements.setupUrl.value.trim());
  if (classification.type === "invalid") {
    elements.connectionDiagnostic.className = "connection-diagnostic error";
    elements.connectionDiagnostic.textContent = classification.reason;
    return;
  }
  const temporaryProfile = {
    id: state.editingProfile?.id || `backend-${Date.now()}`,
    name: elements.setupName.value.trim() || "Backend",
    baseUrl: classification.url,
    networkType: classification.type,
    startup: startupConfigFromForm(),
  };
  elements.connectionDiagnostic.className = "connection-diagnostic running";
  elements.connectionDiagnostic.textContent = "Testando rota, CORS e versao da API...";
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await profileFetch("/api/capabilities", { signal: controller.signal }, temporaryProfile);
    const capabilities = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(capabilities.detail || `HTTP ${response.status}`);
    if (Number(String(capabilities.api_version).split(".")[0]) !== 2) throw new Error(`Backend incompatível: API ${capabilities.api_version || "desconhecida"}.`);
    state.setupValidated = { profile: temporaryProfile, capabilities };
    elements.connectionDiagnostic.className = "connection-diagnostic success";
    const hardware = capabilities.hardware || {};
    const trainingAccess = capabilities.features?.remote_training
      ? "controle de treinamento remoto habilitado"
      : temporaryProfile.networkType === "loopback"
        ? "treinamento nesta maquina disponivel; controle remoto desabilitado"
        : "controle de treinamento remoto desabilitado";
    elements.connectionDiagnostic.textContent = `Conectado a ${capabilities.instance_name}. ${hardware.device_name || "CPU"}; API ${capabilities.api_version}; ${trainingAccess}.`;
    elements.setupModel.replaceChildren();
    (capabilities.models || []).forEach((model) => {
      const option = document.createElement("option");
      option.value = model.alias;
      option.textContent = `${model.label}${model.installed ? " | pronto" : " | requer instalacao"}`;
      option.dataset.installed = String(model.installed);
      elements.setupModel.append(option);
    });
  } catch (error) {
    state.setupValidated = null;
    elements.connectionDiagnostic.className = "connection-diagnostic error";
    elements.connectionDiagnostic.textContent = connectionErrorMessage(error, classification);
  } finally { clearTimeout(timeout); }
}

async function pollSetupInstall() {
  const profile = state.setupValidated.profile;
  for (;;) {
    await new Promise((resolve) => setTimeout(resolve, 1200));
    const body = await (await profileFetch("/api/treinamento/job", {}, profile)).json();
    const job = body.job;
    elements.modelDiagnostic.textContent = `${job?.etapa || "Preparando"} ${Math.round((job?.progresso || 0) * 100)}%`;
    if (!job || !activeJobStatuses.has(job.status)) {
      if (job?.status !== "concluido") throw new Error(job?.logs?.at(-1) || "A instalacao falhou.");
      await testSetupConnection();
      elements.modelDiagnostic.textContent = "Modelo instalado e disponivel.";
      return;
    }
  }
}

async function installSetupModel() {
  if (!state.setupValidated) return;
  elements.modelDiagnostic.textContent = "Iniciando download confirmado...";
  try {
    await profileFetch("/api/modelos/instalar", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ alias: elements.setupModel.value }),
    }, state.setupValidated.profile).then(async (response) => {
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || "Falha ao iniciar instalacao.");
    });
    await pollSetupInstall();
  } catch (error) { elements.modelDiagnostic.textContent = connectionErrorMessage(error); }
}

async function setupNext() {
  if (state.setupStep === 1 && classifyBackendUrl(elements.setupUrl.value.trim()).type === "invalid") {
    elements.setupNetworkHint.textContent = classifyBackendUrl(elements.setupUrl.value.trim()).reason;
    return;
  }
  if (state.setupStep === 2 && !state.setupValidated) {
    await testSetupConnection();
    if (!state.setupValidated) return;
  }
  if (state.setupStep === 3) {
    const selected = elements.setupModel.selectedOptions[0];
    if (selected && selected.dataset.installed !== "true") {
      elements.modelDiagnostic.textContent = "Instale o modelo selecionado antes de concluir.";
      return;
    }
  }
  if (state.setupStep === 4) {
    const profile = state.setupValidated.profile;
    const existing = state.profiles.findIndex((item) => item.id === profile.id || item.baseUrl === profile.baseUrl);
    if (existing >= 0) state.profiles[existing] = profile;
    else state.profiles.push(profile);
    state.activeProfile = profile;
    persistProfiles();
    renderProfiles();
    elements.setupDialog.close();
    await initializeBackend();
    maybeStartTour("clinical");
    return;
  }
  state.setupStep += 1;
  renderSetupStep();
}

const tours = {
  clinical: [
    ["backend", "Backend ativo", "Troque de infraestrutura sem misturar pacientes, modelos ou conversas."],
    ["patient", "Contexto isolado", "Cada paciente possui uma conversa propria dentro deste backend."],
    ["suggestions", "Perguntas sugeridas", "Um clique apenas preenche a pergunta; o envio continua sob seu controle."],
    ["model", "Modelo em uso", "Aqui aparece o modelo-base ou adapter promovido pelo backend ativo."],
    ["question", "Analise clinica", "A pergunta percorre recuperacao, LangGraph, guardrails e auditoria."],
    ["evidence", "Explicabilidade", "Fontes, alertas e a rota decisoria ficam visiveis apos cada resposta."],
  ],
  training: [
    ["infrastructure", "Infraestrutura detectada", "CPU, GPU, memoria e permissoes pertencem ao backend selecionado."],
    ["models", "Modelo-base configuravel", "Use um preset ou um modelo cadastrado compativel com Transformers e PEFT."],
    ["pipeline", "Etapas controladas", "Dados, treino, loss, calibracao e promocao preservam seus gates."],
    ["job", "Execucao observavel", "Progresso, logs e cancelamento acompanham a operacao real do backend."],
    ["adapters", "Promocao segura", "Somente adapters calibrados e aprovados podem entrar em inferencia."],
  ],
};

function completedTours() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEYS.tours) || "{}"); } catch { return {}; }
}

function renderTour() {
  const item = tours[state.tour.name][state.tour.index];
  const target = document.querySelector(`[data-tour="${item[0]}"]`);
  if (!target) return closeTour();
  const rect = target.getBoundingClientRect();
  elements.tourLayer.hidden = false;
  Object.assign(elements.tourFocus.style, { left: `${rect.left - 5}px`, top: `${rect.top - 5}px`, width: `${rect.width + 10}px`, height: `${rect.height + 10}px` });
  const popoverTop = rect.bottom + 12 + 190 < innerHeight ? rect.bottom + 12 : Math.max(12, rect.top - 202);
  Object.assign(elements.tourPopover.style, { left: `${Math.min(Math.max(12, rect.left), innerWidth - 352)}px`, top: `${popoverTop}px` });
  elements.tourProgress.textContent = `${state.tour.index + 1} de ${tours[state.tour.name].length}`;
  elements.tourTitle.textContent = item[1];
  elements.tourText.textContent = item[2];
  elements.tourNext.textContent = state.tour.index === tours[state.tour.name].length - 1 ? "Concluir" : "Proximo";
}

function startTour(name) { state.tour = { name, index: 0 }; renderTour(); }
function closeTour(markComplete = false) {
  if (markComplete && state.tour) localStorage.setItem(STORAGE_KEYS.tours, JSON.stringify({ ...completedTours(), [state.tour.name]: true }));
  state.tour = null;
  elements.tourLayer.hidden = true;
}
function maybeStartTour(name) { if (!completedTours()[name]) startTour(name); }

async function initialize() {
  loadProfiles();
  renderProfiles();
  if (!state.activeProfile) { openSetup(); return; }
  await initializeBackend();
  maybeStartTour("clinical");
}

elements.patientSelect.addEventListener("change", () => {
  const patient = state.patients.find((item) => item.paciente_id === elements.patientSelect.value);
  if (patient) renderPatient(patient);
});
elements.question.addEventListener("input", () => {
  elements.characterCount.textContent = elements.question.value.length;
  if (state.selected) getSession(state.selected.paciente_id).draft = elements.question.value;
});
elements.question.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
  event.preventDefault();
  if (!elements.submit.disabled && elements.question.value.trim()) {
    elements.form.requestSubmit();
  }
});
elements.form.addEventListener("submit", submitQuestion);
elements.viewTabs.forEach((tab) => tab.addEventListener("click", () => switchView(tab.dataset.view)));
elements.profileSelect.addEventListener("change", async () => {
  state.activeProfile = state.profiles.find((profile) => profile.id === elements.profileSelect.value) || null;
  persistProfiles();
  state.selectedAdapter = null;
  clearTimeout(state.trainingPoll);
  await initializeBackend();
});
elements.manageProfiles.addEventListener("click", () => { renderProfiles(); elements.profilesDialog.showModal(); });
elements.closeProfiles.addEventListener("click", () => elements.profilesDialog.close());
elements.newProfile.addEventListener("click", () => { elements.profilesDialog.close(); openSetup(); });
elements.resetProfiles.addEventListener("click", () => {
  state.profiles = [];
  state.activeProfile = null;
  state.sessions.clear();
  localStorage.removeItem(STORAGE_KEYS.profiles);
  localStorage.removeItem(STORAGE_KEYS.active);
  elements.profilesDialog.close();
  openSetup();
});
elements.helpButton.addEventListener("click", () => startTour(elements.trainingView.hidden ? "clinical" : "training"));
elements.setupForm.querySelectorAll('input[name="topology"]').forEach((radio) => {
  radio.addEventListener("change", () => configureTopologyFields(true));
});
[elements.setupName, elements.setupUrl, elements.setupInstanceName].forEach((input) => {
  input.addEventListener("input", () => {
    if (input === elements.setupName && !elements.setupInstanceName.dataset.edited) {
      elements.setupInstanceName.value = elements.setupName.value;
    }
    if (input === elements.setupInstanceName) elements.setupInstanceName.dataset.edited = "true";
    state.setupValidated = null;
    updateSetupCommand();
  });
});
elements.setupUrl.addEventListener("change", syncPortFromDirectUrl);
[elements.setupBindHost, elements.setupPort, elements.setupRemoteTraining, elements.setupOpenBrowser].forEach((control) => {
  control.addEventListener("change", () => {
    if (control === elements.setupPort) syncDirectUrlPort();
    state.setupValidated = null;
    updateSetupCommand();
  });
});
elements.setupNext.addEventListener("click", setupNext);
elements.setupBack.addEventListener("click", () => { if (state.setupStep > 0) { state.setupStep -= 1; renderSetupStep(); } });
elements.testConnection.addEventListener("click", testSetupConnection);
elements.installSetupModel.addEventListener("click", installSetupModel);
elements.copyCommand.addEventListener("click", async () => {
  await navigator.clipboard?.writeText(elements.setupCommand.textContent);
  elements.copyCommand.textContent = "Copiado";
  setTimeout(() => { elements.copyCommand.textContent = "Copiar"; }, 1200);
});
elements.setupDialog.addEventListener("cancel", (event) => { if (!state.activeProfile) event.preventDefault(); });
elements.trainModel.addEventListener("change", () => {
  if (!state.training) return;
  const next = state.training.next_versions?.[elements.trainModel.value];
  if (next) { elements.trainVersion.value = next; state.nextVersion = next; }
});
elements.tourNext.addEventListener("click", () => {
  if (!state.tour) return;
  if (state.tour.index >= tours[state.tour.name].length - 1) closeTour(true);
  else { state.tour.index += 1; renderTour(); }
});
elements.tourClose.addEventListener("click", () => closeTour(false));
addEventListener("resize", () => { if (state.tour) renderTour(); });
elements.trainingForm.addEventListener("submit", (event) => {
  event.preventDefault();
  runTrainingOperation("/api/treinamento/iniciar", trainingPayload());
});
elements.modelForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData(elements.modelForm);
  const targetModules = String(data.get("target_modules") || "").split(",").map((item) => item.trim()).filter(Boolean);
  try {
    await apiRequest("/api/modelos", {
      method: "POST",
      body: JSON.stringify({
        alias: data.get("alias"), label: data.get("label"), source_type: data.get("source_type"),
        source: data.get("source"), revision: data.get("revision") || null, target_modules: targetModules,
        trust_remote_code: data.get("trust_remote_code") === "on",
      }),
    });
    elements.modelForm.reset();
    await loadTrainingOverview();
  } catch (error) { showOpsError(error); }
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
initializeFieldHelp();
initialize();
