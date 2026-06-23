import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

const prediction = {
  prediction: 0,
  diagnosis: "Maligno",
  probability_malignant: 0.91,
  probability_benign: 0.09,
  model: "Regressao Logistica",
};

describe("Aplicação", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify(prediction), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );
  });

  it("exibe exatamente as 30 medições exigidas pela API", () => {
    render(<App />);
    expect(screen.getAllByRole("spinbutton")).toHaveLength(30);
    expect(screen.getByText("0/30 preenchidas")).toBeInTheDocument();
  });

  it("preenche o exemplo e classifica sem enviar dados adicionais", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Preencher exemplo acadêmico" }));
    expect(screen.getByText("30/30 preenchidas")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Classificar" }));
    await waitFor(() => expect(screen.getByText("91,00%")).toBeInTheDocument());

    const fetchMock = vi.mocked(fetch);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [path, options] = fetchMock.mock.calls[0];
    expect(path).toBe("/predict");
    const body = JSON.parse(String(options?.body)) as { features: Record<string, number> };
    expect(Object.keys(body.features)).toHaveLength(30);
  });

  it("não arredonda probabilidade quase certa para 100% (dados clínicos)", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          ...prediction,
          probability_malignant: 0.9999999987841798,
          probability_benign: 1.2158202405207758e-9,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Preencher exemplo acadêmico" }));
    fireEvent.click(screen.getByRole("button", { name: "Classificar" }));

    await waitFor(() => expect(screen.getByText("99,99%")).toBeInTheDocument());
    expect(screen.queryByText("100,00%")).not.toBeInTheDocument();
    expect(screen.getByText("< 0,01%")).toBeInTheDocument();
  });

  it("impede o envio de um formulário incompleto", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Classificar" }));
    expect(
      screen.getByText("Preencha as 30 medições com valores numéricos válidos."),
    ).toBeInTheDocument();
    expect(fetch).not.toHaveBeenCalled();
  });
});
