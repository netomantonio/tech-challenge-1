import { Turnstile, type TurnstileInstance } from "@marsidev/react-turnstile";
import { FormEvent, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { requestAnalysis } from "./api";
import { academicExample, features } from "./features";
import type { ApiResponse, InterpretResponse } from "./types";
import { isInterpretResponse } from "./types";

type FormValues = Record<string, string>;

// Mapa rápido de feature → unidade para uso na tabela de evidências.
const featureUnitMap: Record<string, string> = Object.fromEntries(
  features.map(({ name, unit }) => [name, unit]),
);

const emptyValues = Object.fromEntries(features.map(({ name }) => [name, ""]));
const turnstileSiteKey = import.meta.env.VITE_TURNSTILE_SITE_KEY as string | undefined;

function normalizeValues(values: FormValues): Record<string, number> | null {
  const entries = features.map(({ name }) => [name, Number(values[name])]);
  if (entries.some(([, value]) => !Number.isFinite(value))) {
    return null;
  }
  return Object.fromEntries(entries) as Record<string, number>;
}

// As 30 medições descrevem 10 características em 3 variantes estatísticas
// cada (média, erro padrão, máximo). Agrupá-las por característica em vez de
// por variante evita repetir a mesma grade de campos três vezes.
const measurementRows = features
  .filter((feature) => feature.group === "Médias")
  .map((meanFeature) => {
    const base = meanFeature.name.slice(0, -"_mean".length);
    return {
      key: base,
      label: meanFeature.label,
      unit: meanFeature.unit,
      mean: meanFeature,
      se: features.find((feature) => feature.name === `${base}_se`)!,
      worst: features.find((feature) => feature.name === `${base}_worst`)!,
    };
  });

function formatPercentage(value: number): string {
  const clamped = Math.min(Math.max(value, 0), 1);
  // Trunca (floor) em 2 casas decimais percentuais em vez de arredondar.
  // Em um contexto clínico, arredondar 99,9999% para 100% (ou 0,0001% para 0%)
  // transmitiria uma certeza que o modelo não tem. O toFixed(6) elimina o ruído
  // de ponto flutuante (ex.: 0,91 * 10000 = 9099,9999...) antes do truncamento.
  const percentScaled = Number((clamped * 10000).toFixed(6));
  const truncated = Math.floor(percentScaled) / 10000;
  const format = (fraction: number) =>
    new Intl.NumberFormat("pt-BR", {
      style: "percent",
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(fraction);
  // Probabilidade pequena, porém não nula, que cairia em 0,00% após o
  // truncamento: sinaliza "< 0,01%" sem afirmar zero absoluto.
  if (clamped > 0 && truncated === 0) {
    return `< ${format(0.0001)}`;
  }
  return format(truncated);
}

// Largura da barra como valor CSS válido (ponto decimal, não a vírgula do pt-BR),
// truncada de forma idêntica ao rótulo para manter texto e barra consistentes.
function fillWidth(value: number): string {
  const clamped = Math.min(Math.max(value, 0), 1);
  const percent = Math.floor(Number((clamped * 10000).toFixed(6))) / 100;
  return `${percent.toFixed(2)}%`;
}

// Ilustração esquemática do núcleo celular medido pelas 30 características
// (raio, perímetro, concavidade...). Substitui um ornamento genérico por algo
// que explica o que o instrumento está de fato lendo.
function NucleusIllustration() {
  return (
    <svg
      viewBox="0 0 300 260"
      role="img"
      aria-label="Ilustração esquemática de um núcleo celular com raio, perímetro e ponto côncavo indicados"
    >
      <circle className="nucleus-guide" cx="150" cy="145" r="108" />
      <path
        className="nucleus-outline"
        d="M150,40 C190,40 220,60 235,90 C248,115 245,140 230,155 C250,165 260,185 250,205 C240,225 215,230 195,222 C205,245 185,265 155,262 C130,260 115,240 118,218 C95,225 65,215 55,190 C45,165 55,140 75,128 C60,110 65,80 90,65 C110,52 130,40 150,40 Z"
      />
      <line className="nucleus-leader" x1="150" y1="145" x2="150" y2="41" />
      <text className="nucleus-label" x="154" y="95">
        raio
      </text>
      <line className="nucleus-leader" x1="247" y1="196" x2="284" y2="196" />
      <circle className="nucleus-dot" cx="247" cy="196" r="3" />
      <text className="nucleus-label" x="188" y="18">
        perímetro
      </text>
      <text className="nucleus-label" x="200" y="176">
        concavidade
      </text>
    </svg>
  );
}

function ResultPanel({
  result,
  showInterpretationLink,
}: {
  result: ApiResponse;
  showInterpretationLink: boolean;
}) {
  const malignant = result.diagnosis === "Maligno";
  return (
    <section className={`result-panel ${malignant ? "result-risk" : "result-benign"}`}>
      <div className="result-heading">
        <div>
          <span className="eyebrow">Classificação estimada</span>
          <h2>{result.diagnosis}</h2>
        </div>
        <span className="model-tag">{result.model}</span>
      </div>

      <div className="probability-grid">
        <div>
          <div className="probability-label">
            <span>Probabilidade maligna</span>
            <strong>{formatPercentage(result.probability_malignant)}</strong>
          </div>
          <div className="probability-track" aria-hidden="true">
            <span
              className="probability-fill malignant-fill"
              style={{ width: fillWidth(result.probability_malignant) }}
            />
          </div>
        </div>
        <div>
          <div className="probability-label">
            <span>Probabilidade benigna</span>
            <strong>{formatPercentage(result.probability_benign)}</strong>
          </div>
          <div className="probability-track" aria-hidden="true">
            <span
              className="probability-fill benign-fill"
              style={{ width: fillWidth(result.probability_benign) }}
            />
          </div>
        </div>
      </div>

      {showInterpretationLink && (
        <a className="result-footnote" href="#interpretacao">
          Ver leitura completa da IA ↓
        </a>
      )}
    </section>
  );
}

// Região de largura total, irmã do .workspace: aproveita toda a largura útil da
// tela para a explicação assistida por IA, evidências, insights e disclaimer.
function InterpretationRegion({ result }: { result: InterpretResponse }) {
  return (
    <section className="interpretation-region" id="interpretacao" aria-live="polite">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Leitura assistida por IA</span>
          <h2>Resumo do resultado</h2>
        </div>
        <span className="pill">{result.llm_model}</span>
      </div>

      <div className="interpretation">
        <div className="interpretation-copy">
          <div className="explanation markdown-body">
            {/* react-markdown é seguro por padrão: não renderiza HTML cru e
                sanitiza URLs — adequado para conteúdo vindo do LLM. */}
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                table: ({ node: _node, ...props }) => (
                  <div className="table-scroll">
                    <table {...props} />
                  </div>
                ),
              }}
            >
              {result.explanation}
            </ReactMarkdown>
          </div>
        </div>

        <div className="interpretation-aside">
          <div className="interpretation-block">
            <h3>Evidências do modelo</h3>
            <table className="evidence">
              <thead>
                <tr>
                  <th>Variável</th>
                  <th>Valor</th>
                  <th>Contrib.</th>
                  <th>Direção</th>
                </tr>
              </thead>
              <tbody>
                {result.evidence.map((item) => {
                  const unit = featureUnitMap[item.feature];
                  const unitSuffix = unit && unit !== "adim." ? ` ${unit}` : "";
                  const directionClass =
                    item.direction === "Maligno" ? "direction-maligno" : "direction-benigno";
                  return (
                    <tr key={item.feature}>
                      <td className="feature">{item.feature}</td>
                      <td className="num">
                        {item.value.toPrecision(5)}
                        {unitSuffix}
                      </td>
                      <td className="num">{item.contribution.toFixed(4)}</td>
                      <td>
                        <span className={`direction-pill ${directionClass}`}>{item.direction}</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="interpretation-block">
            <h3>Insights para revisão profissional</h3>
            <ol className="insights-list">
              {result.insights_acionaveis.map((insight, index) => (
                <li key={`${insight.sinal}-${index}`}>
                  <h4>{insight.sinal}</h4>
                  <p>{insight.implicacao_para_revisao}</p>
                  <small>{insight.cautela}</small>
                </li>
              ))}
            </ol>
          </div>
        </div>

        <p className="disclaimer">
          <span className="icon" aria-hidden="true">
            ⚠
          </span>{" "}
          {result.disclaimer}
        </p>
      </div>
    </section>
  );
}

export default function App() {
  const [values, setValues] = useState<FormValues>(emptyValues);
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<"predict" | "interpret" | null>(null);
  const [turnstileToken, setTurnstileToken] = useState<string | null>(null);
  const turnstileRef = useRef<TurnstileInstance>(null);

  const completed = useMemo(
    () => features.filter(({ name }) => values[name].trim() !== "").length,
    [values],
  );

  function fillExample() {
    setValues(
      Object.fromEntries(
        features.map(({ name }) => [name, String(academicExample[name])]),
      ),
    );
    setError(null);
    setResult(null);
  }

  function reset() {
    setValues(emptyValues);
    setError(null);
    setResult(null);
  }

  function updateFeature(name: string, value: string) {
    setValues((current) => ({ ...current, [name]: value }));
  }

  async function submit(endpoint: "/predict" | "/interpret") {
    const normalized = normalizeValues(values);
    if (!normalized || completed !== features.length) {
      setError("Preencha as 30 medições com valores numéricos válidos.");
      return;
    }
    if (endpoint === "/interpret" && !turnstileToken) {
      setError("Conclua a verificação de segurança para solicitar a interpretação.");
      return;
    }

    setLoading(endpoint === "/predict" ? "predict" : "interpret");
    setError(null);
    try {
      setResult(
        await requestAnalysis(endpoint, normalized, turnstileToken ?? undefined),
      );
    } catch (requestError) {
      setResult(null);
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Não foi possível processar a solicitação.",
      );
    } finally {
      if (endpoint === "/interpret") {
        setTurnstileToken(null);
        turnstileRef.current?.reset();
      }
      setLoading(null);
    }
  }

  function preventDefault(event: FormEvent) {
    event.preventDefault();
  }

  return (
    <main>
      <div className="topbar">
        <div className="wordmark">
          Leitura Mamária <small>AI4DEVS · FIAP</small>
        </div>
        <a href="/docs" target="_blank" rel="noreferrer">
          Documentação da API →
        </a>
      </div>

      <header className="hero">
        <div className="hero-copy">
          <span className="eyebrow">Modelo acadêmico · regressão logística</span>
          <h1>Leitura orientada de características mamárias</h1>
          <p>
            Informe as 30 medições do Wisconsin Diagnostic Dataset para consultar
            o modelo acadêmico de regressão logística e, opcionalmente, receber uma
            explicação estruturada.
          </p>
          <div className="hero-notice">
            <span aria-hidden="true">i</span>
            <p>
              Ferramenta educacional sem validação clínica externa. Não substitui
              avaliação, diagnóstico ou conduta de profissionais de saúde.
            </p>
          </div>
        </div>

        <aside className="instrument-card" aria-label="Desempenho do modelo e leitura da amostra">
          <figure>
            <NucleusIllustration />
          </figure>
          <p className="instrument-caption">
            As <strong>30 medições</strong> descrevem a geometria do núcleo celular
            observado: raio, perímetro, área, concavidade e mais — em três variantes
            (média, erro padrão, máximo).
          </p>
          <div className="instrument-stat">
            <span className="stat-label">Recall da classe maligna, no teste reservado</span>
            <strong>97,62%</strong>
            <small>1 falso negativo em 42 casos malignos</small>
          </div>
        </aside>
      </header>

      <section className="workspace">
        <form onSubmit={preventDefault} className="form-panel" noValidate>
          <div className="section-heading">
            <div>
              <span className="eyebrow" style={{ marginBottom: 0 }}>
                Dados de entrada
              </span>
              <h2>Medições da amostra</h2>
            </div>
            <span className="completion">{completed}/30 preenchidas</span>
          </div>

          <div className="form-actions compact-actions">
            <button type="button" className="secondary-button" onClick={fillExample}>
              Preencher exemplo acadêmico
            </button>
            <button type="button" className="text-button" onClick={reset}>
              Limpar
            </button>
          </div>

          <div className="ledger-scroll">
            <table className="ledger">
              <caption className="sr-only">
                Medições agrupadas por característica, com média, erro padrão e máximo por coluna
              </caption>
              <thead>
                <tr>
                  <th scope="col">Característica</th>
                  <th scope="col" id="col-mean">
                    Média
                  </th>
                  <th scope="col" id="col-se">
                    Erro padrão
                  </th>
                  <th scope="col" id="col-worst">
                    Máximo
                  </th>
                </tr>
              </thead>
              <tbody>
                {measurementRows.map((row) => (
                  <tr key={row.key}>
                    <th scope="row" id={`row-${row.key}`}>
                      {row.label}
                      {row.unit !== "adim." && <span className="unit">{row.unit}</span>}
                    </th>
                    {(
                      [
                        [row.mean, "mean"],
                        [row.se, "se"],
                        [row.worst, "worst"],
                      ] as const
                    ).map(([feature, column]) => (
                      <td key={feature.name}>
                        <input
                          type="number"
                          inputMode="decimal"
                          step="any"
                          required
                          aria-labelledby={`row-${row.key} col-${column}`}
                          className="ledger-input"
                          value={values[feature.name]}
                          onChange={(event) => updateFeature(feature.name, event.target.value)}
                        />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="submission-panel">
            <div>
              <h3>Verificação para interpretação</h3>
              <p>A classificação simples não exige esta verificação.</p>
            </div>
            {turnstileSiteKey ? (
              <Turnstile
                ref={turnstileRef}
                siteKey={turnstileSiteKey}
                onSuccess={setTurnstileToken}
                onExpire={() => setTurnstileToken(null)}
                onError={() => setTurnstileToken(null)}
                options={{ theme: "light", language: "pt-BR" }}
              />
            ) : (
              <p className="configuration-warning">
                Configure VITE_TURNSTILE_SITE_KEY para habilitar a interpretação.
              </p>
            )}
          </div>

          {error && (
            <div className="error-message" role="alert">
              {error}
            </div>
          )}

          <div className="form-actions primary-actions">
            <button
              type="button"
              className="secondary-button"
              disabled={loading !== null}
              onClick={() => submit("/predict")}
            >
              {loading === "predict" ? "Classificando…" : "Classificar"}
            </button>
            <button
              type="button"
              className="primary-button"
              disabled={loading !== null || !turnstileToken}
              onClick={() => submit("/interpret")}
            >
              {loading === "interpret"
                ? "Gerando interpretação…"
                : "Classificar e interpretar"}
            </button>
          </div>
        </form>

        <aside className="result-column" aria-live="polite">
          {result ? (
            <ResultPanel result={result} showInterpretationLink={isInterpretResponse(result)} />
          ) : (
            <div className="empty-result">
              <span aria-hidden="true">30</span>
              <h2>Resultado da análise</h2>
              <p>
                Preencha as medições e envie a amostra. Nenhum dado é salvo pelo
                aplicativo.
              </p>
            </div>
          )}
        </aside>
      </section>

      {result && isInterpretResponse(result) && <InterpretationRegion result={result} />}

      <footer>
        <p>Projeto acadêmico · Wisconsin Breast Cancer Diagnostic Dataset</p>
        <a href="/docs" target="_blank" rel="noreferrer">
          Documentação da API
        </a>
      </footer>
    </main>
  );
}
