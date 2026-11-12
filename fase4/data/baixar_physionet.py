"""Baixa um registro numerico do PhysioNet e o converte para o esquema da Fase 4.

O enunciado sugere o PhysioNet como fonte de dados. Este script traz um
registro de acesso aberto do **MIMIC-III Waveform Database Matched Subset**
(numericos de monitor de leito a 1 Hz: FC, pressao arterial invasiva e nao
invasiva, SpO2, frequencia respiratoria) e o grava no mesmo formato de
`sinais_vitais.csv` das amostras sinteticas, com uma leitura por minuto.
Assim os detectores de `fase4.anomaly_detection` rodam em dados reais sem
nenhuma mudanca.

    python -m fase4.data.baixar_physionet                 # registro padrao, 12 h
    python -m fase4.data.baixar_physionet --minutos 0     # registro inteiro (45,8 h)
    python -m fase4.data.baixar_physionet --registro p04/p044083/p044083-2112-05-04-19-50n

Saida em `fase4/data/physionet/<registro>/`: `sinais_vitais.csv` e
`fonte.json` (dataset, licenca, URL, mapeamento de sinais, preenchimentos).
Os arquivos WFDB originais (`.hea`/`.dat`) ficam na mesma pasta, fora do git.

Fonte e licenca: Moody, B., Moody, G., Villarroel, M., Clifford, G. D., &
Silva, I. (2020). MIMIC-III Waveform Database Matched Subset (version 1.0).
PhysioNet. https://doi.org/10.13026/c2294b - Open Data Commons Open Database
License v1.0 (ODbL). As datas sao deslocadas (anos 21xx) por anonimizacao.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fase4.config import DIR_PHYSIONET, LIMIARES_ANOMALIA

DATASET = "mimic3wdb-matched"
VERSAO = "1.0"
BASE_URL = f"https://physionet.org/files/{DATASET}/{VERSAO}/"
URL_DATASET = f"https://physionet.org/content/{DATASET}/{VERSAO}/"
DOI = "10.13026/c2294b"
LICENCA = "Open Data Commons Open Database License v1.0 (ODbL)"
REGISTRO_PADRAO = "p04/p044083/p044083-2112-05-04-19-50n"
MINUTOS_PADRAO = 720  # 12 h, como as amostras sinteticas

# Coluna do projeto -> (sinais candidatos do MIMIC, em ordem de preferencia; unidade)
MAPEAMENTO_CANAIS: dict[str, tuple[tuple[str, ...], str]] = {
    "frequencia_cardiaca": (("HR", "PULSE"), "bpm"),
    "pressao_sistolica": (("ABP Sys", "ABPSys", "NBP Sys", "NBPSys"), "mmHg"),
    "pressao_diastolica": (("ABP Dias", "ABPDias", "NBP Dias", "NBPDias"), "mmHg"),
    "saturacao_oxigenio": (("SpO2", "%SpO2"), "%"),
    "temperatura": (("Temp", "TEMP"), "C"),
    "frequencia_respiratoria": (("RESP", "RR"), "irpm"),
}
# Minutos que um valor pode ser arrastado para frente num buraco. A pressao
# nao invasiva e medida de tempos em tempos, por isso tolera mais.
LIMITE_FFILL_MIN = 5
LIMITE_FFILL_NBP_MIN = 60


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def resolver_registro(registro: str) -> tuple[str, str]:
    """Separa `pXX/pXXXXXX/nome` em (subpasta, nome); aceita so o nome."""
    registro = registro.strip().strip("/")
    if "/" in registro:
        subpasta, nome = registro.rsplit("/", 1)
        return subpasta, nome
    paciente = registro.split("-")[0]  # p044083
    if not (paciente.startswith("p") and paciente[1:].isdigit()):
        raise ValueError(f"nao sei derivar a pasta do registro '{registro}'; informe pXX/pXXXXXX/nome")
    return f"p{paciente[1:3]}/{paciente}", registro


def arquivos_do_header(texto_hea: str) -> list[str]:
    """Nomes dos arquivos de sinal citados num header WFDB (sem repeticao)."""
    arquivos: list[str] = []
    linhas = [l.strip() for l in texto_hea.splitlines() if l.strip() and not l.startswith("#")]
    for linha in linhas[1:]:  # a primeira linha e o cabecalho do registro
        nome = linha.split()[0]
        if nome.startswith("~"):
            continue
        if nome not in arquivos:
            arquivos.append(nome)
    return arquivos


def _baixar(url: str, destino: Path) -> int:
    """Baixa `url` para `destino`, pulando se o tamanho local ja bate com o remoto."""
    requisicao = urllib.request.Request(url, headers={"User-Agent": "fase4-physionet/1.0"})
    with urllib.request.urlopen(requisicao, timeout=120) as resposta:
        tamanho = int(resposta.headers.get("Content-Length", "0") or 0)
        if destino.exists() and tamanho and destino.stat().st_size == tamanho:
            return tamanho
        destino.parent.mkdir(parents=True, exist_ok=True)
        with destino.open("wb") as f:
            while True:
                bloco = resposta.read(1 << 20)
                if not bloco:
                    break
                f.write(bloco)
    return destino.stat().st_size


def baixar_registro(registro: str = REGISTRO_PADRAO, destino: Path | None = None) -> tuple[Path, list[str]]:
    """Baixa o `.hea` e os `.dat` do registro. Devolve (pasta, arquivos)."""
    subpasta, nome = resolver_registro(registro)
    pasta = (destino or DIR_PHYSIONET) / nome
    pasta.mkdir(parents=True, exist_ok=True)
    url_base = f"{BASE_URL}{subpasta}/"

    header = pasta / f"{nome}.hea"
    _baixar(f"{url_base}{nome}.hea", header)
    arquivos = [f"{nome}.hea"]
    for dat in arquivos_do_header(header.read_text(encoding="utf-8", errors="ignore")):
        _baixar(f"{url_base}{dat}", pasta / dat)
        arquivos.append(dat)
    return pasta, arquivos


# ---------------------------------------------------------------------------
# Leitura e conversao
# ---------------------------------------------------------------------------


def ler_registro(caminho_sem_extensao: Path | str, minutos: int | None = MINUTOS_PADRAO) -> tuple[pd.DataFrame, dict]:
    """Le o registro WFDB como DataFrame 1 Hz indexado por datetime."""
    import wfdb

    caminho = Path(caminho_sem_extensao)
    sampto = int(minutos * 60) if minutos else None
    registro = wfdb.rdrecord(str(caminho), sampto=sampto)
    fs = float(registro.fs)
    inicio = registro.base_datetime or datetime(2100, 1, 1, 0, 0, 0)
    indice = pd.to_datetime(inicio) + pd.to_timedelta(np.arange(registro.sig_len) / fs, unit="s")
    df = pd.DataFrame(registro.p_signal, index=indice, columns=list(registro.sig_name))
    meta = {
        "fs_hz": fs,
        "amostras_lidas": int(registro.sig_len),
        "sinais": list(registro.sig_name),
        "unidades": list(getattr(registro, "units", []) or []),
        "inicio": pd.Timestamp(inicio).isoformat(),
    }
    return df, meta


def converter_para_schema(
    df_1hz: pd.DataFrame,
    mapeamento: dict[str, tuple[tuple[str, ...], str]] = MAPEAMENTO_CANAIS,
    limite_ffill_min: int = LIMITE_FFILL_MIN,
) -> tuple[pd.DataFrame, dict]:
    """Reamostra para mediana por minuto e renomeia para o esquema do projeto.

    Para cada coluna do projeto usa o primeiro sinal candidato que tenha
    algum dado. Buracos curtos sao arrastados para frente; linhas que
    continuam sem valor em algum canal sao descartadas, porque os detectores
    precisam de leituras completas por minuto.
    """
    if not isinstance(df_1hz.index, pd.DatetimeIndex):
        raise ValueError("o DataFrame precisa ter indice datetime")

    colunas: dict[str, pd.Series] = {}
    fontes: dict[str, dict] = {}
    ausentes: list[str] = []
    for coluna, (candidatos, unidade) in mapeamento.items():
        fonte = next(
            (c for c in candidatos if c in df_1hz.columns and df_1hz[c].notna().any()), None
        )
        if fonte is None:
            ausentes.append(coluna)
            continue
        serie = pd.to_numeric(df_1hz[fonte], errors="coerce").resample("1min").median()
        limite = LIMITE_FFILL_NBP_MIN if fonte.upper().startswith("NBP") else limite_ffill_min
        serie = serie.ffill(limit=limite)
        colunas[coluna] = serie
        fontes[coluna] = {"fonte": fonte, "unidade": unidade, "ffill_max_min": limite}

    if not colunas:
        raise ValueError("nenhum sinal do mapeamento existe no registro")

    tabela = pd.DataFrame(colunas)
    faixas = {f.nome: f for f in LIMIARES_ANOMALIA.faixas}
    # Valores fisiologicamente impossiveis (sensor desconectado grava 0) viram NaN.
    for coluna in tabela.columns:
        faixa = faixas.get(coluna)
        if faixa is None:
            continue
        minimo = (faixa.critico_baixo or faixa.minimo) * 0.5
        maximo = (faixa.critico_alto or faixa.maximo) * 1.6 if coluna != "saturacao_oxigenio" else 100.0
        fora = (tabela[coluna] < minimo) | (tabela[coluna] > maximo)
        tabela.loc[fora, coluna] = np.nan

    antes = len(tabela)
    tabela = tabela.dropna(how="any").reset_index().rename(columns={"index": "timestamp"})
    tabela["timestamp"] = pd.to_datetime(tabela["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    ordem = ["timestamp"] + [c for c in mapeamento if c in tabela.columns]
    tabela = tabela[ordem].reset_index(drop=True)
    for coluna in ordem[1:]:
        tabela[coluna] = tabela[coluna].round(2)

    info = {
        "canais": fontes,
        "canais_ausentes": ausentes,
        "minutos_totais": int(antes),
        "minutos_exportados": int(len(tabela)),
        "minutos_descartados": int(antes - len(tabela)),
    }
    return tabela, info


def salvar(df: pd.DataFrame, info: dict, meta: dict, pasta: Path) -> tuple[Path, Path]:
    pasta.mkdir(parents=True, exist_ok=True)
    caminho_csv = pasta / "sinais_vitais.csv"
    df.to_csv(caminho_csv, index=False)
    fonte = {
        "dataset": "MIMIC-III Waveform Database Matched Subset",
        "slug": DATASET,
        "versao": VERSAO,
        "doi": DOI,
        "url_dataset": URL_DATASET,
        "licenca": LICENCA,
        "citacao": (
            "Moody, B., Moody, G., Villarroel, M., Clifford, G. D., & Silva, I. (2020). "
            "MIMIC-III Waveform Database Matched Subset (version 1.0). PhysioNet."
        ),
        "baixado_em": datetime.now(timezone.utc).isoformat(),
        **meta,
        **info,
        "observacao": (
            "Datas deslocadas por anonimizacao (anos 21xx). Nao ha rotulo clinico de "
            "desfecho: o registro serve para verificar o comportamento dos detectores "
            "em dados reais, nao para medir sensibilidade."
        ),
    }
    caminho_json = pasta / "fonte.json"
    caminho_json.write_text(json.dumps(fonte, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return caminho_csv, caminho_json


def preparar_registro(
    registro: str = REGISTRO_PADRAO, destino: Path | None = None, minutos: int | None = MINUTOS_PADRAO
) -> dict:
    """Baixa, converte e grava. Devolve um resumo com os caminhos."""
    pasta, arquivos = baixar_registro(registro, destino)
    _subpasta, nome = resolver_registro(registro)
    df_1hz, meta = ler_registro(pasta / nome, minutos)
    tabela, info = converter_para_schema(df_1hz)
    meta.update(
        {
            "registro": registro,
            "arquivos": arquivos,
            "url_registro": f"{BASE_URL}{_subpasta}/{nome}.hea",
            "minutos_pedidos": minutos or "todos",
        }
    )
    caminho_csv, caminho_json = salvar(tabela, info, meta, pasta)
    return {
        "pasta": str(pasta),
        "csv": str(caminho_csv),
        "fonte": str(caminho_json),
        "minutos_exportados": info["minutos_exportados"],
        "canais": list(info["canais"]),
        "canais_ausentes": info["canais_ausentes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registro", default=REGISTRO_PADRAO)
    parser.add_argument("--destino", type=Path, default=None, help="Pasta raiz (padrao: fase4/data/physionet).")
    parser.add_argument(
        "--minutos", type=int, default=MINUTOS_PADRAO,
        help=f"Minutos a exportar a partir do inicio (padrao {MINUTOS_PADRAO}; 0 = registro inteiro).",
    )
    args = parser.parse_args()
    resumo = preparar_registro(args.registro, args.destino, args.minutos or None)
    print(f"registro {args.registro}")
    print(f"  {resumo['minutos_exportados']} minutos exportados; canais: {', '.join(resumo['canais'])}")
    if resumo["canais_ausentes"]:
        print(f"  canais ausentes no registro: {', '.join(resumo['canais_ausentes'])}")
    print(f"  {resumo['csv']}")
    print(f"  {resumo['fonte']}")


if __name__ == "__main__":
    main()
