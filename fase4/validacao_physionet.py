"""Roda os detectores de sinais vitais da Fase 4 num registro real do PhysioNet.

Depois de `python -m fase4.data.baixar_physionet`, este script aplica ao
`sinais_vitais.csv` convertido exatamente o que o pipeline aplica as
amostras sinteticas: as tres regras por canal (`avaliar_serie`), o Isolation
Forest em lote (`detectar_isolation_forest`), o detector multivariado online
(`monitorar_em_tempo_real`) e o motor de alertas. Grava em
`resultados/fase4/physionet/<registro>/`:

* `relatorio_physionet.md` - fonte, mapeamento, resumo por canal, achados;
* `achados.json` / `achados.csv` - todos os achados;
* `alertas.json` - alertas consolidados;
* `sinais_vitais_scores.csv` - serie com o score do Isolation Forest.

    python -m fase4.validacao_physionet
    python -m fase4.validacao_physionet --registro p044083-2112-05-04-19-50n

O que isto valida e o comportamento dos detectores em dados reais de monitor
(taxa de achados, plausibilidade dos episodios, ausencia de erro em buracos
de sensor). Nao ha rotulo de desfecho no registro, entao nao se mede
sensibilidade nem especificidade.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fase4 import anomaly_detection
from fase4.alertas import Achado, Alerta, MotorDeAlertas
from fase4.config import DIR_PHYSIONET, DIR_RESULTADOS, LIMIARES_ANOMALIA
from fase4.data.baixar_physionet import REGISTRO_PADRAO
from fase4.pipeline import salvar_achados_csv

DIR_SAIDA_PADRAO = DIR_RESULTADOS / "physionet"


def validar_registro(
    registro: str = REGISTRO_PADRAO,
    dir_dados: Path | None = None,
    dir_saida: Path | None = None,
) -> dict:
    nome = registro.strip().strip("/").rsplit("/", 1)[-1]
    pasta_dados = (dir_dados or DIR_PHYSIONET) / nome
    caminho_csv = pasta_dados / "sinais_vitais.csv"
    if not caminho_csv.exists():
        raise FileNotFoundError(f"{caminho_csv} nao existe; rode antes python -m fase4.data.baixar_physionet")
    fonte = json.loads((pasta_dados / "fonte.json").read_text(encoding="utf-8")) if (pasta_dados / "fonte.json").exists() else {}

    df = anomaly_detection.carregar_sinais_vitais(caminho_csv)
    achados_regras = anomaly_detection.avaliar_serie(df)
    multivariados, enriquecido = anomaly_detection.detectar_isolation_forest(df)
    online = [
        a for _, lista in anomaly_detection.monitorar_em_tempo_real(df) for a in lista
        if a.tipo == "anomalia_multivariada"
    ]
    achados = achados_regras + multivariados
    alertas = MotorDeAlertas().gerar(nome, achados)

    saida = (dir_saida or DIR_SAIDA_PADRAO) / nome
    saida.mkdir(parents=True, exist_ok=True)
    enriquecido.to_csv(saida / "sinais_vitais_scores.csv", index=False)
    (saida / "achados.json").write_text(
        json.dumps([a.to_dict() for a in achados], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    salvar_achados_csv(achados, saida / "achados.csv")
    (saida / "alertas.json").write_text(
        json.dumps([a.to_dict() for a in alertas], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    relatorio = relatorio_markdown_physionet(nome, df, fonte, achados, multivariados, online, alertas)
    (saida / "relatorio_physionet.md").write_text(relatorio, encoding="utf-8")

    return {
        "registro": nome,
        "minutos": int(len(df)),
        "achados": len(achados),
        "achados_por_severidade": dict(Counter(a.severidade for a in achados)),
        "episodios_lote": len(multivariados),
        "episodios_online": len(online),
        "alertas": len(alertas),
        "saida": str(saida),
    }


def _resumo_por_canal(df: pd.DataFrame) -> list[str]:
    linhas = [
        "| Canal | Leituras | Mediana | Mín | Máx | Mediana do basal (120 min) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for canal in LIMIARES_ANOMALIA.canais:
        if canal not in df.columns:
            linhas.append(f"| {canal} | ausente | - | - | - | - |")
            continue
        serie = df[canal]
        basal = serie.iloc[: anomaly_detection.MINUTOS_REFERENCIA]
        linhas.append(
            f"| {canal} | {int(serie.notna().sum())} | {serie.median():.1f} | {serie.min():.1f} | "
            f"{serie.max():.1f} | {basal.median():.1f} |"
        )
    return linhas


def _achados_por_hora(achados: list[Achado]) -> list[str]:
    contagem: Counter = Counter()
    for achado in achados:
        if achado.severidade == "informativo" or not achado.instante:
            continue
        contagem[achado.instante[:13]] += 1
    if not contagem:
        return ["Nenhum achado notificável."]
    linhas = ["| Hora | Achados notificáveis |", "| --- | ---: |"]
    for hora in sorted(contagem):
        linhas.append(f"| {hora.replace('T', ' ')}h | {contagem[hora]} |")
    return linhas


def relatorio_markdown_physionet(
    nome: str,
    df: pd.DataFrame,
    fonte: dict,
    achados: list[Achado],
    multivariados: list[Achado],
    online: list[Achado],
    alertas: list[Alerta],
) -> str:
    notificaveis = [a for a in achados if a.severidade != "informativo"]
    horas = len(df) / 60.0
    linhas = [
        f"# Validação dos detectores em dados reais - {nome}",
        "",
        f"Gerado em {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Fonte",
        "",
        f"- Dataset: {fonte.get('dataset', 'PhysioNet')} (`{fonte.get('slug', '')}` v{fonte.get('versao', '')}), "
        f"DOI {fonte.get('doi', '-')}",
        f"- Registro: `{fonte.get('registro', nome)}` - {fonte.get('url_registro', '')}",
        f"- Licença: {fonte.get('licenca', '-')}",
        f"- Citação: {fonte.get('citacao', '-')}",
        f"- Amostragem original: {fonte.get('fs_hz', '?')} Hz, {fonte.get('amostras_lidas', '?')} amostras lidas; "
        f"exportados {fonte.get('minutos_exportados', len(df))} minutos "
        f"({fonte.get('minutos_descartados', 0)} descartados por falta de leitura)",
        "",
        "| Coluna do projeto | Sinal do MIMIC | Unidade | Preenchimento máx. |",
        "| --- | --- | --- | ---: |",
    ]
    for coluna, dados in fonte.get("canais", {}).items():
        linhas.append(
            f"| {coluna} | `{dados.get('fonte')}` | {dados.get('unidade')} | {dados.get('ffill_max_min')} min |"
        )
    for coluna in fonte.get("canais_ausentes", []):
        linhas.append(f"| {coluna} | ausente no registro | - | - |")

    linhas += ["", "## Resumo da série", "", *_resumo_por_canal(df), ""]

    linhas += [
        "## Achados",
        "",
        f"{len(achados)} achados em {horas:.1f} h de monitoramento, {len(notificaveis)} notificáveis "
        f"({len(notificaveis) / horas if horas else 0:.2f} por hora). "
        f"Alertas após consolidação: {len(alertas)}.",
        "",
        "| Tipo | Severidade | Quantidade |",
        "| --- | --- | ---: |",
    ]
    contagem = Counter((a.tipo, a.severidade) for a in achados)
    for (tipo, severidade), quantidade in sorted(contagem.items(), key=lambda i: (-i[1], i[0])):
        linhas.append(f"| {tipo} | {severidade} | {quantidade} |")

    linhas += ["", "### Por hora", "", *_achados_por_hora(achados), ""]

    linhas += [
        "### Episódios multivariados (Isolation Forest)",
        "",
        f"Lote: {len(multivariados)} episódio(s); tempo real: {len(online)} episódio(s).",
        "",
    ]
    for rotulo, lista in (("lote", multivariados), ("tempo real", online)):
        for achado in lista:
            linhas.append(
                f"- {rotulo}: {achado.evidencias.get('inicio', '')[11:16]}-{achado.evidencias.get('fim', '')[11:16]} "
                f"({achado.evidencias.get('minutos')} min) - {achado.descricao}"
            )
    if multivariados or online:
        linhas.append("")

    linhas += ["### Alertas consolidados", ""]
    if alertas:
        linhas += ["| Prioridade | Alerta | Destino |", "| --- | --- | --- |"]
        for alerta in alertas:
            linhas.append(f"| {alerta.prioridade} | {alerta.titulo} | {alerta.destino} |")
    else:
        linhas.append("Nenhum alerta.")

    linhas += ["", "### Todos os achados", "", "| Severidade | Instante | Achado |", "| --- | --- | --- |"]
    for achado in achados:
        linhas.append(f"| {achado.severidade} | `{achado.instante or '-'}` | {achado.descricao} |")

    linhas += [
        "",
        "> O registro não tem rótulo clínico de desfecho. Esta validação mostra como os detectores se "
        "comportam em dados reais de monitor (taxa de achados, plausibilidade dos episódios, robustez a "
        "buracos de sensor); não mede sensibilidade nem especificidade. Os limiares são os mesmos "
        "calibrados nas amostras sintéticas.",
        "",
    ]
    return "\n".join(linhas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registro", default=REGISTRO_PADRAO)
    parser.add_argument("--dir-dados", type=Path, default=None)
    parser.add_argument("--dir-saida", type=Path, default=None)
    args = parser.parse_args()
    resumo = validar_registro(args.registro, args.dir_dados, args.dir_saida)
    print(
        f"{resumo['registro']}: {resumo['minutos']} min, {resumo['achados']} achados "
        f"{resumo['achados_por_severidade']}, episodios IF lote={resumo['episodios_lote']} "
        f"online={resumo['episodios_online']}, alertas={resumo['alertas']}"
    )
    print(f"  {resumo['saida']}")


if __name__ == "__main__":
    main()
