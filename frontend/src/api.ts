import type { ApiResponse } from "./types";

interface ErrorPayload {
  detail?: string | Record<string, unknown>;
}

function errorMessage(status: number, payload: ErrorPayload): string {
  if (typeof payload.detail === "string") {
    return payload.detail;
  }
  if (status === 422) {
    return "Confira se todas as 30 medições foram preenchidas corretamente.";
  }
  if (status === 429) {
    return "O limite temporário foi atingido. Aguarde um minuto e tente novamente.";
  }
  return "Não foi possível processar a solicitação.";
}

export async function requestAnalysis(
  endpoint: "/predict" | "/interpret",
  featureValues: Record<string, number>,
  turnstileToken?: string,
): Promise<ApiResponse> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (turnstileToken) {
    headers["X-Turnstile-Token"] = turnstileToken;
  }

  const response = await fetch(endpoint, {
    method: "POST",
    headers,
    body: JSON.stringify({ features: featureValues }),
  });
  const payload = (await response.json()) as ApiResponse | ErrorPayload;
  if (!response.ok) {
    throw new Error(errorMessage(response.status, payload as ErrorPayload));
  }
  return payload as ApiResponse;
}
