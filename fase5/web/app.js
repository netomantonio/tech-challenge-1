// Interface da Guardia AI: so monta o formulario e mostra o que a API devolve.
const $ = (id) => document.getElementById(id);
let exemplos = [];
let analiseId = null;

function esc(texto) {
  return String(texto).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function pct(v) {
  return (v * 100).toLocaleString('pt-BR', { maximumFractionDigits: 1 }) + '%';
}

function num(v) {
  return Number(v).toLocaleString('pt-BR', { maximumSignificantDigits: 4 });
}

// Markdown minimo para a resposta da LLM. O texto e escapado antes de qualquer marcacao.
function markdown(texto, fontes, citadas) {
  // Os modelos gpt-oss citam com colchetes lenticulares; normaliza para [id].
  const linhas = esc(texto.replace(/【/g, '[').replace(/】/g, ']').replace(/[‐-―]/g, '-')).split('\n');
  const html = [];
  let lista = null;
  let tabela = [];
  const fechar = () => {
    if (lista) { html.push(`</${lista}>`); lista = null; }
    if (tabela.length) {
      const linhasTab = tabela.filter((l) => !/^\|\s*:?-{2,}/.test(l));
      html.push('<table>' + linhasTab.map((l, i) => '<tr>' + l.replace(/^\||\|$/g, '').split('|')
        .map((c) => (i === 0 ? `<th>${c.trim()}</th>` : `<td>${c.trim()}</td>`)).join('') + '</tr>').join('') + '</table>');
      tabela = [];
    }
  };
  for (const linha of linhas) {
    const t = linha.trim();
    if (t.startsWith('|')) { if (lista) { html.push(`</${lista}>`); lista = null; } tabela.push(t); continue; }
    if (tabela.length) fechar();
    let m;
    if ((m = t.match(/^#{1,6}\s+(.*)/))) { fechar(); html.push(`<h4>${m[1]}</h4>`); }
    else if ((m = t.match(/^[-*•]\s+(.*)/))) { if (lista !== 'ul') { fechar(); html.push('<ul>'); lista = 'ul'; } html.push(`<li>${m[1]}</li>`); }
    else if ((m = t.match(/^\d+[.)]\s+(.*)/))) { if (lista !== 'ol') { fechar(); html.push('<ol>'); lista = 'ol'; } html.push(`<li>${m[1]}</li>`); }
    else if (t === '' || /^-{3,}$/.test(t)) { fechar(); }
    else { fechar(); html.push(`<p>${t}</p>`); }
  }
  fechar();
  const ids = new Set(fontes.map((f) => f.id));
  return html.join('')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/&lt;br\s*\/?&gt;/g, '<br>')
    .replace(/\b([a-z0-9]+(?:-[a-z0-9]+)+)\b(?=[^<]*\])/g, (id) =>
      ids.has(id) ? `<a class="cit" href="#fonte-${id}">${id}</a>`
        : (citadas.invalidas.includes(id) ? `<span class="cit invalida" title="Fonte não recuperada">${id}</span>` : id));
}

function preencher(i) {
  const ex = exemplos[i];
  $('pergunta').value = ex.entrada.pergunta;
  $('relato').value = ex.entrada.relato;
  for (const [nome, valor] of Object.entries(ex.entrada.features)) {
    const campo = document.querySelector(`[data-feature="${nome}"]`);
    if (campo) campo.value = valor;
  }
  const real = ex.classe_real === 0 ? 'maligno' : 'benigno';
  $('exemplo-info').textContent = `Rótulo real no dataset: ${real}. Ele aparece aqui só para a demonstração; no atendimento não seria conhecido.`;
}

function montarMedidas(features) {
  $('medidas').innerHTML = Object.keys(features).map((nome, i) =>
    `<div><label for="m${i}">${esc(nome)}</label>
     <input id="m${i}" data-feature="${esc(nome)}" type="number" step="any" min="0" required></div>`).join('');
}

function coletar() {
  const features = {};
  document.querySelectorAll('[data-feature]').forEach((c) => { features[c.dataset.feature] = Number(c.value); });
  return { features, relato: $('relato').value, pergunta: $('pergunta').value };
}

function mostrar(r) {
  analiseId = r.id;
  const maligno = r.ml.classe_estimada === 0;
  $('classe').textContent = maligno ? 'Estimativa: padrão compatível com maligno' : 'Estimativa: padrão compatível com benigno';
  $('classe').className = 'classe ' + (maligno ? 'maligno' : 'benigno');
  // Largura em CSS precisa de ponto decimal; pct() formata com virgula para exibicao.
  $('barra-prob').style.width = `${(r.ml.probabilidade_maligno * 100).toFixed(1)}%`;
  $('prob').textContent = `Probabilidade estimada de maligno: ${pct(r.ml.probabilidade_maligno)} (não calibrada). ` +
    `Modelo ${r.ml.modelo}, versão ${r.ml.versao}. Não confirma nem exclui câncer.`;

  const maior = Math.max(...r.ml.contribuicoes.map((c) => Math.abs(c.delta_probabilidade)), 1e-9);
  $('contribuicoes').innerHTML = r.ml.contribuicoes.map((c) => {
    const largura = (Math.abs(c.delta_probabilidade) / maior) * 50;
    const lado = c.delta_probabilidade >= 0 ? 'pos' : 'neg';
    const sinal = c.delta_probabilidade >= 0 ? '+' : '−';
    return `<li><span class="nome">${esc(c.feature)}<small>valor ${num(c.valor)} · mediana ${num(c.mediana_treino)}</small></span>
      <span class="eixo"><span class="${lado}" style="width:${largura}%"></span></span>
      <span class="valor">${sinal}${(Math.abs(c.delta_probabilidade) * 100).toLocaleString('pt-BR', { maximumFractionDigits: 1 })} p.p.</span></li>`;
  }).join('');

  $('provedor').textContent = r.provedor_efetivo === 'demo-sem-llm' ? 'modo demo — sem LLM' : `LLM: ${r.provedor_efetivo}`;
  $('explicacao').innerHTML = markdown(r.explicacao, r.fontes, { invalidas: r.citacoes_invalidas });
  $('cit-invalidas').hidden = !r.citacoes_invalidas.length;
  $('cit-invalidas').textContent = `Atenção: a resposta citou fontes que não foram recuperadas: ${r.citacoes_invalidas.join(', ')}. Confira antes de usar.`;

  $('fontes-vazio').hidden = r.fontes.length > 0;
  $('fontes').innerHTML = r.fontes.map((f) => {
    const citada = r.fontes_citadas.includes(f.id) ? 'citada na explicação' : 'recuperada, não citada';
    return `<li id="fonte-${esc(f.id)}"><strong>${esc(f.titulo)}</strong>
      <div class="meta">${esc(f.id)} · ${f.contexto === 'saude' ? 'saúde' : 'segurança'} · similaridade ${num(f.score)} · ${citada}</div>
      <p class="trecho">${esc(f.texto)}</p>
      <a href="${esc(f.url)}" target="_blank" rel="noopener noreferrer">Fonte oficial</a> · consultada em ${esc(f.consultado_em)}</li>`;
  }).join('');

  document.querySelectorAll('[data-decisao]').forEach((b) => b.setAttribute('aria-pressed', 'false'));
  $('revisao-status').textContent = '';
  $('auditoria').textContent = `ID da análise na auditoria: ${r.id}`;
  $('vazio').hidden = true;
  $('saida').hidden = false;
}

async function analisar(evento) {
  evento.preventDefault();
  $('erro').hidden = true;
  $('analisar').disabled = true;
  $('analisar').textContent = 'Analisando…';
  try {
    const resp = await fetch('/analisar', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(coletar()),
    });
    const corpo = await resp.json();
    if (!resp.ok) throw new Error(typeof corpo.detail === 'string' ? corpo.detail : 'Entrada inválida.');
    mostrar(corpo);
  } catch (e) {
    $('erro').textContent = e.message;
    $('erro').hidden = false;
  } finally {
    $('analisar').disabled = false;
    $('analisar').textContent = 'Analisar';
  }
}

async function revisar(botao) {
  if (!analiseId) return;
  const resp = await fetch('/revisao', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id: analiseId, decisao: botao.dataset.decisao }),
  });
  if (!resp.ok) { $('revisao-status').textContent = 'Não foi possível registrar a decisão.'; return; }
  document.querySelectorAll('[data-decisao]').forEach((b) => b.setAttribute('aria-pressed', String(b === botao)));
  $('revisao-status').textContent = `Decisão "${botao.dataset.decisao}" registrada na auditoria.`;
}

async function iniciar() {
  const info = await (await fetch('/info')).json();
  $('backend').textContent = info.backend === 'groq' ? `LLM: ${info.modelo_llm}` : 'modo demo — sem LLM';
  exemplos = await (await fetch('/exemplos')).json();
  $('exemplo').innerHTML = exemplos.map((ex, i) => `<option value="${i}">${esc(ex.nome)}</option>`).join('');
  montarMedidas(exemplos[0].entrada.features);
  preencher(0);
  $('exemplo').addEventListener('change', (e) => preencher(Number(e.target.value)));
  $('form').addEventListener('submit', analisar);
  document.querySelectorAll('[data-decisao]').forEach((b) => b.addEventListener('click', () => revisar(b)));
}

iniciar().catch(() => { $('erro').textContent = 'Falha ao carregar a aplicação. Execute python -m fase5.train.'; $('erro').hidden = false; });
