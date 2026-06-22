import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { forwardRef, useImperativeHandle } from "react";
import { beforeAll, describe, expect, it, vi } from "vitest";

import type { TurnstileInstance } from "@marsidev/react-turnstile";

vi.mock("@marsidev/react-turnstile", () => ({
  Turnstile: forwardRef<TurnstileInstance, { onSuccess: (token: string) => void }>(
    ({ onSuccess }, reference) => {
      useImperativeHandle(reference, () => ({
        execute: () => undefined,
        getResponse: () => "token-de-teste",
        getResponsePromise: async () => "token-de-teste",
        isExpired: () => false,
        remove: () => undefined,
        render: () => "widget-de-teste",
        reset: () => undefined,
      }));
      return (
        <button type="button" onClick={() => onSuccess("token-de-teste")}>
          Concluir verificação
        </button>
      );
    },
  ),
}));

describe("Interpretação", () => {
  let App: typeof import("./App").default;

  beforeAll(async () => {
    vi.stubEnv("VITE_TURNSTILE_SITE_KEY", "1x00000000000000000000AA");
    App = (await import("./App")).default;
  });

  it("renderiza explicação, evidências, insights e disclaimer", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            prediction: 0,
            diagnosis: "Maligno",
            probability_malignant: 0.91,
            probability_benign: 0.09,
            model: "Regressao Logistica",
            explanation: "RESUMO DO RESULTADO\nClassificação estimada para revisão.",
            llm_model: "modelo-de-teste",
            prompt_version: "clinical_explanation_v3",
            disclaimer: "Não constitui diagnóstico médico.",
            evidence: [
              {
                feature: "area_worst",
                value: 1200,
                contribution: -2.4,
                direction: "Maligno",
              },
            ],
            insights_acionaveis: [
              {
                sinal: "Sinal estatístico relevante.",
                evidencia_numerica: "contribuicao_local=-2.4",
                implicacao_para_revisao: "Priorizar revisão profissional.",
                cautela: "Não interpretar isoladamente.",
              },
            ],
            quality_checks: { score_objetivo: 1 },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      ),
    );

    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Preencher exemplo acadêmico" }));
    fireEvent.click(screen.getByRole("button", { name: "Concluir verificação" }));
    fireEvent.click(screen.getByRole("button", { name: "Classificar e interpretar" }));

    await waitFor(() =>
      expect(screen.getByText("Não constitui diagnóstico médico.")).toBeInTheDocument(),
    );
    expect(screen.getByText("area_worst")).toBeInTheDocument();
    expect(screen.getByText("Priorizar revisão profissional.")).toBeInTheDocument();
    expect(fetch).toHaveBeenCalledWith(
      "/interpret",
      expect.objectContaining({
        headers: expect.objectContaining({ "X-Turnstile-Token": "token-de-teste" }),
      }),
    );
  });
});
