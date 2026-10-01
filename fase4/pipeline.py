"""Pipeline de monitoramento multimodal de um paciente (Fase 4).

Junta as tres modalidades numa execucao so: analisa o video da sessao de
fisioterapia, o audio da consulta e as series de sinais vitais, prescricoes e
movimentacao; funde os achados; emite os alertas para a equipe medica; e grava
os relatorios em `resultados/fase4/<paciente>/`.

As modalidades sao opcionais. Se o paciente nao tem video (como o PAC-0002),
o pipeline roda sem ela e registra isso no resumo, em vez de falhar. Esse foi
o comportamento escolhido porque, num hospital, a disponibilidade de cada
fonte de dado varia de leito para leito e de dia para dia.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fase4 import anomaly_detection, audio_analysis, video_analysis
from fase4.alertas import Achado, Alerta, MotorDeAlertas, grupo_do_achado
from fase4.config import DIR_RESULTADOS, LIMIARES_ANOMALIA, LimiaresAnomalia
from fase4.data.gerar_dados_sinteticos import DIR_AMOSTRAS
from fase4.logging_utils import registrar_evento
from fase4.multimodal_fusion import ResultadoFusao, fundir


@dataclass
class ResultadoPaciente:
    paciente_id: str
    modalidades_processadas: list[str]
    modalidades_ausentes: list[str]
    achados: list[Achado]
    alertas: list[Alerta]
    fusao: ResultadoFusao
    resultado_video: video_analysis.ResultadoVideo | None = None
    resultado_audio: audio_analysis.ResultadoAudio | None = None
    arquivos_gerados: dict[str, str] = field(default_factory=dict)

    def provedores_utilizados(self) -> dict[str, str]:
        audio = self.resultado_audio
        return {
            "speech_to_text": audio.transcricao.provedor if audio else "nao_executado",
            "text_analytics": audio.analise_texto.provedor if audio else "nao_executado",
        }

    def to_dict(self) -> dict:
        return {
            "paciente_id": self.paciente_id,
            "modalidades_processadas": self.modalidades_processadas,
            "modalidades_ausentes": self.modalidades_ausentes,
            "fusao": self.fusao.to_dict(),
            "total_achados": len(self.achados),
            "total_alertas": len(self.alertas),
            "achados": [a.to_dict() for a in self.achados],
            "alertas": [a.to_dict() for a in self.alertas],
            "arquivos_gerados": self.arquivos_gerados,
            "servicos_azure": self.provedores_utilizados(),
        }


def localizar_amostras(paciente_id: str, raiz: Path | None = None) -> dict[str, Path]:
    """Mapeia quais arquivos de amostra existem para o paciente."""
    base = (raiz or DIR_AMOSTRAS) / paciente_id
    candidatos = {
        "sinais_vitais": base / "sinais_vitais.csv",
        "prescricoes": base / "prescricoes.json",
        "movimentacao": base / "movimentacao.csv",
        "audio": base / "consulta.wav",
        "video": base / "fisioterapia.mp4",
        "keypoints": base / "fisioterapia.keypoints.json",
        "objetos": base / "fisioterapia.objetos.json",
    }
    return {nome: caminho for nome, caminho in candidatos.items() if caminho.exists()}


def processar_paciente(
    paciente_id: str,
    raiz_amostras: Path | None = None,
    dir_saida: Path | None = None,
    provedor_azure: str = "auto",
    backend_pose: str = "anotado",
    backend_detector: str = "anotado",
    limiares: LimiaresAnomalia = LIMIARES_ANOMALIA,
    gravar_relatorios: bool = True,
    gravar_video_anotado: bool = True,
) -> ResultadoPaciente:
    """Executa o monitoramento completo de um paciente."""
    amostras = localizar_amostras(paciente_id, raiz_amostras)
    saida = (dir_saida or DIR_RESULTADOS) / paciente_id
    if gravar_relatorios:
        saida.mkdir(parents=True, exist_ok=True)

    achados: list[Achado] = []
    processadas: list[str] = []
    ausentes: list[str] = []
    arquivos: dict[str, str] = {}
    resultado_video = None
    resultado_audio = None

    # --- Video ------------------------------------------------------------
    tem_video = "keypoints" in amostras or ("video" in amostras and backend_pose != "anotado")
    if tem_video:
        resultado_video = video_analysis.analisar_sessao(
            paciente_id=paciente_id,
            caminho_video=amostras.get("video"),
            caminho_keypoints=amostras.get("keypoints"),
            caminho_objetos=amostras.get("objetos"),
            backend_pose=backend_pose,
            backend_detector=backend_detector,
        )
        achados.extend(resultado_video.achados)
        processadas.append("video")
        if gravar_relatorios:
            caminho = saida / "relatorio_video.md"
            caminho.write_text(
                video_analysis.relatorio_markdown(resultado_video), encoding="utf-8"
            )
            arquivos["relatorio_video"] = str(caminho)
        if gravar_video_anotado and "video" in amostras:
            destino = saida / "fisioterapia_anotado.mp4"
            if video_analysis.renderizar_video_anotado(
                amostras["video"], resultado_video, destino
            ):
                arquivos["video_anotado"] = str(destino)
    else:
        ausentes.append("video")

    # --- Audio ------------------------------------------------------------
    if "audio" in amostras:
        resultado_audio = audio_analysis.analisar_consulta(
            amostras["audio"], paciente_id, provedor_azure=provedor_azure
        )
        achados.extend(resultado_audio.achados)
        processadas.append("audio")
        if gravar_relatorios:
            caminho = saida / "relatorio_audio.md"
            caminho.write_text(
                audio_analysis.relatorio_markdown(resultado_audio), encoding="utf-8"
            )
            arquivos["relatorio_audio"] = str(caminho)
    else:
        ausentes.append("audio")

    # --- Sinais vitais ----------------------------------------------------
    if "sinais_vitais" in amostras:
        df = anomaly_detection.carregar_sinais_vitais(amostras["sinais_vitais"])
        achados.extend(anomaly_detection.avaliar_serie(df, limiares))
        multivariados, enriquecido = anomaly_detection.detectar_isolation_forest(df, limiares)
        achados.extend(multivariados)
        processadas.append("sinais_vitais")
        if gravar_relatorios:
            caminho = saida / "sinais_vitais_scores.csv"
            enriquecido.to_csv(caminho, index=False)
            arquivos["sinais_vitais_scores"] = str(caminho)
    else:
        ausentes.append("sinais_vitais")

    # --- Prescricoes ------------------------------------------------------
    if "prescricoes" in amostras:
        with amostras["prescricoes"].open(encoding="utf-8") as f:
            eventos = json.load(f)
        achados.extend(anomaly_detection.detectar_anomalias_prescricoes(eventos, limiares))
        processadas.append("prescricoes")
    else:
        ausentes.append("prescricoes")

    # --- Movimentacao -----------------------------------------------------
    if "movimentacao" in amostras:
        movimentacao = anomaly_detection.carregar_movimentacao(amostras["movimentacao"])
        achados.extend(
            anomaly_detection.detectar_anomalias_movimentacao(movimentacao, limiares)
        )
        processadas.append("movimentacao")
    else:
        ausentes.append("movimentacao")

    # --- Fusao e alertas --------------------------------------------------
    fusao = fundir(paciente_id, achados)
    todos = achados + fusao.achados_multimodais

    motor = MotorDeAlertas()
    alertas = motor.gerar(
        paciente_id, todos, grupos_absorvidos=fusao.grupos_absorvidos
    )

    for alerta in alertas:
        registrar_evento(
            "alerta_emitido",
            paciente_id=paciente_id,
            alerta_id=alerta.alerta_id,
            prioridade=alerta.prioridade,
            titulo=alerta.titulo,
            destino=alerta.destino,
            modalidades=alerta.modalidades,
        )

    registrar_evento(
        "monitoramento_concluido",
        paciente_id=paciente_id,
        modalidades_processadas=processadas,
        modalidades_ausentes=ausentes,
        total_achados=len(todos),
        total_alertas=len(alertas),
        risco=round(fusao.risco, 1),
        nivel_risco=fusao.nivel_risco,
        sindromes=[s.nome for s in fusao.sindromes],
        servicos_azure={
            "speech_to_text": resultado_audio.transcricao.provedor if resultado_audio else "nao_executado",
            "text_analytics": resultado_audio.analise_texto.provedor if resultado_audio else "nao_executado",
        },
    )

    resultado = ResultadoPaciente(
        paciente_id=paciente_id,
        modalidades_processadas=processadas,
        modalidades_ausentes=ausentes,
        achados=todos,
        alertas=alertas,
        fusao=fusao,
        resultado_video=resultado_video,
        resultado_audio=resultado_audio,
        arquivos_gerados=arquivos,
    )

    if gravar_relatorios:
        caminho = saida / "relatorio_monitoramento.md"
        caminho.write_text(relatorio_markdown(resultado), encoding="utf-8")
        arquivos["relatorio_monitoramento"] = str(caminho)

        caminho = saida / "achados.json"
        caminho.write_text(
            json.dumps([a.to_dict() for a in todos], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        arquivos["achados"] = str(caminho)

        caminho = saida / "alertas.json"
        caminho.write_text(
            json.dumps([a.to_dict() for a in alertas], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        arquivos["alertas"] = str(caminho)

        caminho = saida / "achados.csv"
        _salvar_achados_csv(todos, caminho)
        arquivos["achados_csv"] = str(caminho)

    return resultado


def _salvar_achados_csv(achados: list[Achado], destino: Path) -> None:
    pd.DataFrame(
        [
            {
                "modalidade": a.modalidade,
                "tipo": a.tipo,
                "severidade": a.severidade,
                "score": round(a.score, 3),
                "instante": a.instante,
                "descricao": a.descricao,
            }
            for a in achados
        ]
    ).to_csv(destino, index=False)


def _descrever_familias(sindrome) -> str:
    """Famílias de achado do quadro, com quantos achados cada uma trouxe."""
    contagem: dict[str, int] = {}
    for achado in sindrome.achados:
        grupo = grupo_do_achado(achado.tipo)
        contagem[grupo] = contagem.get(grupo, 0) + 1
    ordenadas = sorted(contagem.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"`{grupo}` ({quantidade})" for grupo, quantidade in ordenadas)


def relatorio_markdown(resultado: ResultadoPaciente) -> str:
    """Relatorio consolidado do paciente: fluxo, achados, sindromes e alertas."""
    fusao = resultado.fusao
    linhas = [
        f"# Relatório de monitoramento multimodal - {resultado.paciente_id}",
        "",
        f"Gerado em {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"**Risco calculado:** {fusao.risco:.1f}/100 (nível **{fusao.nivel_risco}**)",
        "",
        f"Modalidades processadas: {', '.join(resultado.modalidades_processadas) or 'nenhuma'}  ",
        f"Modalidades ausentes: {', '.join(resultado.modalidades_ausentes) or 'nenhuma'}",
        "",
        "## Achados por modalidade",
        "",
        "| Modalidade | Achados |",
        "| --- | ---: |",
    ]
    for modalidade, quantidade in sorted(fusao.contagem_por_modalidade.items()):
        linhas.append(f"| {modalidade} | {quantidade} |")

    linhas += [
        "",
        "| Severidade | Achados |",
        "| --- | ---: |",
    ]
    for severidade, quantidade in fusao.contagem_por_severidade.items():
        linhas.append(f"| {severidade} | {quantidade} |")

    linhas += ["", "## Quadros identificados pela fusão", ""]
    if fusao.sindromes:
        linhas += [
            "Uma mesma família de achado pode sustentar mais de um quadro: a piora "
            "dos sinais vitais, por exemplo, entra tanto na suspeita de infecção "
            "quanto no risco medicamentoso. Os achados individuais estão listados "
            "na última seção.",
            "",
        ]
        for sindrome in fusao.sindromes:
            linhas += [
                f"### {sindrome.nome.replace('_', ' ')}",
                "",
                f"{sindrome.descricao}.",
                "",
                f"- Severidade: **{sindrome.severidade}**",
                f"- Modalidades envolvidas: {', '.join(sindrome.modalidades)}",
                f"- Famílias que sustentam ({len(sindrome.achados)} achados): "
                f"{_descrever_familias(sindrome)}",
                f"- Conduta sugerida: {sindrome.acao}",
                "",
            ]
    else:
        linhas += [
            "Nenhuma combinação entre modalidades diferentes atingiu os critérios das "
            "regras de fusão.",
            "",
        ]

    linhas += ["## Alertas gerados para a equipe", ""]
    if resultado.alertas:
        linhas += ["| Prioridade | Alerta | Destino | Instante |", "| --- | --- | --- | --- |"]
        for alerta in resultado.alertas:
            instante = alerta.achados[0].get("instante") if alerta.achados else None
            linhas.append(
                f"| {alerta.prioridade} | {alerta.titulo} | {alerta.destino} | "
                f"`{instante or '-'}` |"
            )
    else:
        linhas.append("Nenhum alerta foi emitido: não houve achado de atenção ou crítico.")

    linhas += ["", "## Todos os achados", "", "| Modalidade | Severidade | Instante | Achado |", "| --- | --- | --- | --- |"]
    for achado in resultado.achados:
        linhas.append(
            f"| {achado.modalidade} | {achado.severidade} | `{achado.instante or '-'}` | "
            f"{achado.descricao} |"
        )

    linhas += [
        "",
        "## Serviços utilizados",
        "",
        "| Serviço | Provedor efetivo |",
        "| --- | --- |",
    ]
    for servico, provedor in resultado.provedores_utilizados().items():
        linhas.append(f"| {servico} | `{provedor}` |")

    linhas += [
        "",
        "> Protótipo acadêmico. Nenhum alerta deste relatório "
        "substitui avaliação clínica.",
        "",
    ]
    return "\n".join(linhas)


def processar_lote(
    pacientes: list[str],
    raiz_amostras: Path | None = None,
    dir_saida: Path | None = None,
    provedor_azure: str = "auto",
    backend_pose: str = "anotado",
    backend_detector: str = "anotado",
) -> dict:
    """Processa varios pacientes e grava o resumo consolidado da execucao."""
    saida = dir_saida or DIR_RESULTADOS
    saida.mkdir(parents=True, exist_ok=True)

    resultados = []
    for paciente_id in pacientes:
        resultado = processar_paciente(
            paciente_id,
            raiz_amostras=raiz_amostras,
            dir_saida=saida,
            provedor_azure=provedor_azure,
            backend_pose=backend_pose,
            backend_detector=backend_detector,
        )
        resultados.append(resultado)

    resumo = {
        "executado_em": datetime.now(timezone.utc).isoformat(),
        "servicos_azure": {r.paciente_id: r.provedores_utilizados() for r in resultados},
        "backend_pose": backend_pose,
        "backend_detector": backend_detector,
        "pacientes": [
            {
                "paciente_id": r.paciente_id,
                "servicos_azure": r.provedores_utilizados(),
                "risco": round(r.fusao.risco, 1),
                "nivel_risco": r.fusao.nivel_risco,
                "modalidades_processadas": r.modalidades_processadas,
                "modalidades_ausentes": r.modalidades_ausentes,
                "achados_por_modalidade": r.fusao.contagem_por_modalidade,
                "achados_por_severidade": r.fusao.contagem_por_severidade,
                "sindromes": [s.nome for s in r.fusao.sindromes],
                "total_alertas": len(r.alertas),
                "alertas_por_prioridade": _contar_prioridades(r.alertas),
            }
            for r in resultados
        ],
    }

    (saida / "resumo_execucao.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return resumo


def _contar_prioridades(alertas: list[Alerta]) -> dict[str, int]:
    contagem: dict[str, int] = {}
    for alerta in alertas:
        contagem[alerta.prioridade] = contagem.get(alerta.prioridade, 0) + 1
    return contagem
