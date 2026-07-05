export interface PredictResponse {
  prediction: number;
  diagnosis: "Maligno" | "Benigno";
  probability_malignant: number;
  probability_benign: number;
  model: string;
}

export interface Evidence {
  feature: string;
  value: number;
  contribution: number;
  direction: string;
  unit: string;
}

export interface ActionableInsight {
  sinal: string;
  evidencia_numerica: string;
  implicacao_para_revisao: string;
  cautela: string;
  proximos_passos: string;
}

export interface InterpretResponse extends PredictResponse {
  explanation: string;
  llm_model: string;
  prompt_version: string;
  disclaimer: string;
  evidence: Evidence[];
  insights_acionaveis: ActionableInsight[];
  quality_checks: Record<string, boolean | number>;
}

export type ApiResponse = PredictResponse | InterpretResponse;

export function isInterpretResponse(response: ApiResponse): response is InterpretResponse {
  return "explanation" in response;
}
