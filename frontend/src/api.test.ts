import { describe, expect, it, vi } from "vitest";

import { requestAnalysis } from "./api";

describe("Cliente da API", () => {
  it("envia o token Turnstile somente pelo cabeçalho", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          prediction: 1,
          diagnosis: "Benigno",
          probability_malignant: 0.1,
          probability_benign: 0.9,
          model: "Regressao Logistica",
          explanation: "Explicação",
          llm_model: "modelo",
          prompt_version: "v3",
          disclaimer: "Aviso",
          evidence: [],
          insights_acionaveis: [],
          quality_checks: {},
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    await requestAnalysis("/interpret", { radius_mean: 1 }, "token-seguro");
    const [, options] = fetchMock.mock.calls[0];
    expect(options.headers["X-Turnstile-Token"]).toBe("token-seguro");
    expect(String(options.body)).not.toContain("token-seguro");
  });

  it("traduz a resposta de rate limiting para uma mensagem acionável", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "Limite excedido" }), {
          status: 429,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );

    await expect(requestAnalysis("/predict", {})).rejects.toThrow("Limite excedido");
  });
});
