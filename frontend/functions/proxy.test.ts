import { describe, expect, it } from "vitest";

import { proxyRequest } from "./[[path]]";

describe("Proxy da API", () => {
  it("preserva status e corpo e adiciona cabeçalhos de segurança", async () => {
    const response = await proxyRequest(
      new Request("https://preview.pages.dev/health"),
      async () =>
        new Response('{"status":"ready"}', {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
    );

    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ status: "ready" });
    expect(response.headers.get("X-Content-Type-Options")).toBe("nosniff");
    expect(response.headers.get("Cache-Control")).toBe("no-store");
  });

  it("converte falhas do Service Binding em 503 sem expor detalhes", async () => {
    const response = await proxyRequest(
      new Request("https://preview.pages.dev/predict"),
      async () => {
        throw new Error("segredo que não deve vazar");
      },
    );

    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({
      detail: "A API está temporariamente indisponível.",
    });
  });
});
