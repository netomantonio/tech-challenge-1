import { Turnstile, type TurnstileInstance } from "@marsidev/react-turnstile";
import { FormEvent, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { requestAnalysis } from "./api";
import { academicExample, featureGroups, features } from "./features";
import type { ApiResponse, InterpretResponse } from "./types";
import { isInterpretResponse } from "./types";

type FormValues = Record<string, string>;

// Mapa rápido de feature → unidade para uso nos cards de evidência.
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

function ResultPanel({ result }: { result: ApiResponse }) {
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
    </section>
  );
}

// Região de largura total, irmã do .workspace: aproveita toda a largura útil da
// tela para a explicação assistida por IA, evidências, insights e disclaimer.
// O JSX interno (markdown, evidências, insights, disclaimer) é preservado;
// apenas o container/posição no layout muda em relação à versão anterior.
function InterpretationRegion({ result }: { result: InterpretResponse }) {
  return (
    <section className="interpretation-region" aria-live="polite">
      <div className="interpretation">
        <div className="interpretation-copy">
          <span className="eyebrow">Explicação assistida por IA</span>
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
            <h3>Principais evidências do modelo</h3>
            <div className="evidence-list">
              {result.evidence.map((item) => (
                <article key={item.feature} className="evidence-card">
                  <strong>{item.feature}</strong>
                  <span>Direção: {item.direction}</span>
                  <small>
                    Valor {item.value.toPrecision(5)}{" "}
                    {featureUnitMap[item.feature] && featureUnitMap[item.feature] !== "adim."
                      ? featureUnitMap[item.feature]
                      : ""}
                    {" "}· contribuição{" "}
                    {item.contribution.toFixed(4)}
                  </small>
                </article>
              ))}
            </div>
          </div>

          <div className="interpretation-block">
            <h3>Insights para revisão profissional</h3>
            <div className="insights-list">
              {result.insights_acionaveis.map((insight, index) => (
                <article key={`${insight.sinal}-${index}`}>
                  <strong>{insight.sinal}</strong>
                  <p>{insight.implicacao_para_revisao}</p>
                  <small>{insight.cautela}</small>
                </article>
              ))}
            </div>
          </div>
        </div>

        <p className="disclaimer">{result.disclaimer}</p>
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
      <header className="hero">
        <div className="hero-copy">
          <span className="product-mark">AI4DEVS · FIAP</span>
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
        <aside className="hero-metric" aria-label="Desempenho do modelo no teste reservado">
          <span>Recall da classe maligna</span>
          <strong>97,62%</strong>
          <small>1 falso negativo no conjunto de teste</small>
        </aside>
      </header>

      <section className="workspace">
        <form onSubmit={preventDefault} className="form-panel" noValidate>
          <div className="section-heading">
            <div>
              <span className="eyebrow">Dados de entrada</span>
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

          {featureGroups.map((group) => (
            <fieldset key={group}>
              <legend>{group}</legend>
              <div className="feature-grid">
                {features
                  .filter((feature) => feature.group === group)
                  .map((feature) => {
                    const inputId = `feature-${feature.name.replaceAll(" ", "-")}`;
                    return (
                      <label key={feature.name} htmlFor={inputId}>
                        <span>{feature.label}</span>
                        <input
                          id={inputId}
                          name={feature.name}
                          type="number"
                          inputMode="decimal"
                          step="any"
                          required
                          value={values[feature.name]}
                          onChange={(event) =>
                            setValues((current) => ({
                              ...current,
                              [feature.name]: event.target.value,
                            }))
                          }
                          aria-describedby={`${inputId}-help`}
                        />
                        <small id={`${inputId}-help`}>{feature.description}</small>
                      </label>
                    );
                  })}
              </div>
            </fieldset>
          ))}

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
            <ResultPanel result={result} />
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

      {result && isInterpretResponse(result) && (
        <InterpretationRegion result={result} />
      )}

      <footer>
        <p>Projeto acadêmico · Wisconsin Breast Cancer Diagnostic Dataset</p>
        <a href="/docs" target="_blank" rel="noreferrer">
          Documentação da API
        </a>
      </footer>
    </main>
  );
}
