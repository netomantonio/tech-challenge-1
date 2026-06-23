export type FeatureGroup = "Médias" | "Erro padrão" | "Piores valores";

export interface FeatureDefinition {
  name: string;
  label: string;
  description: string;
  unit: string;
  group: FeatureGroup;
}

// Unidade de medida de cada característica conforme o Wisconsin Diagnostic Dataset.
// Raio, perímetro: mm · Área: mm² · Demais: adimensional (razão).
const measurementMeta: Record<string, { label: string; unit: string }> = {
  radius:            { label: "Raio",              unit: "mm" },
  texture:           { label: "Textura",           unit: "adim." },
  perimeter:         { label: "Perímetro",         unit: "mm" },
  area:              { label: "Área",              unit: "mm²" },
  smoothness:        { label: "Suavidade",         unit: "adim." },
  compactness:       { label: "Compacidade",       unit: "adim." },
  concavity:         { label: "Concavidade",       unit: "adim." },
  "concave points":  { label: "Pontos côncavos",   unit: "adim." },
  symmetry:          { label: "Simetria",          unit: "adim." },
  fractal_dimension: { label: "Dimensão fractal",  unit: "adim." },
};

const groups: Array<[string, FeatureGroup, string]> = [
  ["mean",  "Médias",         "valor médio observado"],
  ["se",    "Erro padrão",    "erro padrão da medição"],
  ["worst", "Piores valores", "maior valor observado"],
];

export const features: FeatureDefinition[] = groups.flatMap(
  ([suffix, group, detail]) =>
    Object.entries(measurementMeta).map(([name, { label, unit }]) => ({
      name: `${name}_${suffix}`,
      label,
      unit,
      description: `${label}: ${detail}. Unidade: ${unit === "adim." ? "adimensional (razão)" : unit}.`,
      group,
    })),
);

export const featureGroups: FeatureGroup[] = [
  "Médias",
  "Erro padrão",
  "Piores valores",
];

export const academicExample: Record<string, number> = {
  radius_mean: 17.99,
  texture_mean: 10.38,
  perimeter_mean: 122.8,
  area_mean: 1001,
  smoothness_mean: 0.1184,
  compactness_mean: 0.2776,
  concavity_mean: 0.3001,
  "concave points_mean": 0.1471,
  symmetry_mean: 0.2419,
  fractal_dimension_mean: 0.07871,
  radius_se: 1.095,
  texture_se: 0.9053,
  perimeter_se: 8.589,
  area_se: 153.4,
  smoothness_se: 0.006399,
  compactness_se: 0.04904,
  concavity_se: 0.05373,
  "concave points_se": 0.01587,
  symmetry_se: 0.03003,
  fractal_dimension_se: 0.006193,
  radius_worst: 25.38,
  texture_worst: 17.33,
  perimeter_worst: 184.6,
  area_worst: 2019,
  smoothness_worst: 0.1622,
  compactness_worst: 0.6656,
  concavity_worst: 0.7119,
  "concave points_worst": 0.2654,
  symmetry_worst: 0.4601,
  fractal_dimension_worst: 0.1189,
};
