"""Demonstracao de ponta a ponta do monitoramento multimodal (Fase 4).

Pensado para a gravacao do video pedido no PDF. Roda as tres modalidades de um
paciente, imprime cada etapa e termina no que interessa a equipe: os alertas.
O modo `--tempo-real` percorre a serie de sinais vitais minuto a minuto e
mostra os alertas aparecendo conforme o quadro evolui.

Exemplos:

    python -m fase4.cli_demo --paciente-id PAC-0006
    python -m fase4.cli_demo --paciente-id PAC-0002
    python -m fase4.cli_demo --paciente-id PAC-0006 --tempo-real
    python -m fase4.cli_demo --paciente-id PAC-0006 --provedor-azure azure
    python -m fase4.cli_demo --todos
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from fase4 import anomaly_detection
from fase4.alertas import MotorDeAlertas, canais_padrao, descrever_canais
from fase4.azure_services import status_servicos
from fase4.data.gerar_dados_sinteticos import PACIENTES_PADRAO
from fase4.pipeline import (
    ARQUIVO_AUDIO_PADRAO,
    localizar_amostras,
    processar_lote,
    processar_paciente,
)
from fase4.video_analysis import parse_area_critica

LARGURA = 74


def _titulo(texto: str) -> None:
    print("=" * LARGURA)
    print(texto)
    print("=" * LARGURA)


def _secao(texto: str) -> None:
    print(f"\n--- {texto} " + "-" * max(0, LARGURA - len(texto) - 6))


def mostrar_servicos() -> None:
    _secao("Disponibilidade dos servicos gerenciados")
    for servico, estado in status_servicos().items():
        credencial = "sim" if estado["credencial"] else "nao"
        sdk = "sim" if estado["sdk"] else "nao"
        print(
            f"  {servico:16s} disponivel={estado['provedor_efetivo']:8s} "
            f"credencial={credencial:3s} sdk={sdk}"
        )
    if any(e["provedor_efetivo"] == "offline" for e in status_servicos().values()):
        print(
            "  (modo offline: defina AZURE_SPEECH_KEY/REGION e "
            "AZURE_LANGUAGE_KEY/ENDPOINT para usar a Azure)"
        )


def _parse_area_critica(texto: str | None) -> tuple[float, float, float, float] | None:
    """Converte "x1,y1,x2,y2" (pixels) na ROI manual da area critica."""
    try:
        return parse_area_critica(texto)
    except ValueError as erro:
        raise argparse.ArgumentTypeError(f"--area-critica: {erro}") from erro


def demonstrar_paciente(
    paciente_id: str, provedor_azure: str, backend_pose: str, backend_detector: str,
    raiz_amostras: Path | None = None, dir_saida: Path | None = None,
    area_critica: tuple[float, float, float, float] | None = None,
    arquivo_audio: str = ARQUIVO_AUDIO_PADRAO,
) -> None:
    _titulo(f"Monitoramento multimodal - paciente {paciente_id}")

    amostras = localizar_amostras(paciente_id, raiz_amostras, arquivo_audio=arquivo_audio)
    _secao("Amostras encontradas")
    if not amostras:
        print("  nenhuma. Rode primeiro: python -m fase4.data.gerar_dados_sinteticos")
        return
    for nome, caminho in amostras.items():
        print(f"  {nome:16s} {caminho.name}")

    mostrar_servicos()

    resultado = processar_paciente(
        paciente_id,
        raiz_amostras=raiz_amostras,
        dir_saida=dir_saida,
        provedor_azure=provedor_azure,
        backend_pose=backend_pose,
        backend_detector=backend_detector,
        area_critica=area_critica,
        arquivo_audio=arquivo_audio,
    )

    if resultado.resultado_video is not None:
        _secao("Analise de video (pose + deteccao de objetos)")
        video = resultado.resultado_video
        print(
            f"  {video.quadros_analisados} quadros, {video.duracao_s:.1f} s, "
            f"pose={video.backend_pose} ({video.quadros_com_pose} quadros com pose), "
            f"detector={video.backend_detector}"
        )
        print(
            f"  area critica: {video.origem_area_critica}"
            + (f" {video.area_critica_manual}" if video.area_critica_manual else "")
            + f"; mao do paciente por {video.regra_mao}"
        )
        for medida, valor in video.resumo_angulos.items():
            print(f"  {medida:34s} {valor:6.1f} graus")
        for achado in video.achados:
            print(f"  [{achado.severidade:11s}] {achado.instante or '-':>8s}  {achado.descricao}")

    if resultado.resultado_audio is not None:
        _secao("Analise de audio (acustica + Azure)")
        audio = resultado.resultado_audio
        atributos = audio.atributos
        print(
            f"  {atributos.duracao_total_s:.1f} s, {atributos.quantidade_segmentos} frases, "
            f"pausa={atributos.proporcao_pausa:.2f}, "
            f"taxa de fala={atributos.taxa_fala_silabas_s:.2f} sil/s, "
            f"jitter={atributos.jitter_f0:.4f}"
        )
        print(
            f"  transcricao ({audio.transcricao.provedor}, "
            f"confianca {audio.transcricao.confianca:.2f}): "
            f"{audio.transcricao.texto[:120]}"
        )
        print(
            f"  sentimento ({audio.analise_texto.provedor}): "
            f"{audio.analise_texto.sentimento}; termos criticos: "
            f"{', '.join(t.termo for t in audio.analise_texto.termos_criticos) or 'nenhum'}"
        )
        if audio.wer_referencia is not None:
            print(
                f"  taxa de erro de palavras contra a transcricao de referencia: "
                f"{audio.wer_referencia:.1%}"
            )
        for achado in audio.achados:
            print(f"  [{achado.severidade:11s}] {achado.instante or '-':>8s}  {achado.descricao}")

    _secao("Deteccao de anomalias (sinais vitais, prescricoes, movimentacao)")
    for achado in resultado.achados:
        if achado.modalidade in ("sinais_vitais", "prescricao", "movimentacao"):
            instante = (achado.instante or "-")[-8:]
            print(f"  [{achado.severidade:11s}] {instante:>8s}  {achado.descricao}")

    _secao("Fusao multimodal")
    fusao = resultado.fusao
    print(f"  risco: {fusao.risco:.1f}/100 (nivel {fusao.nivel_risco})")
    print(f"  achados por modalidade: {fusao.contagem_por_modalidade}")
    if fusao.sindromes:
        for sindrome in fusao.sindromes:
            print(f"  quadro: {sindrome.nome} ({', '.join(sindrome.modalidades)})")
            print(f"          {sindrome.descricao}")
            print(f"          conduta: {sindrome.acao}")
    else:
        print("  nenhum quadro multimodal identificado")

    _secao(f"Alertas para a equipe medica ({len(resultado.alertas)})")
    for alerta in resultado.alertas:
        entregue = f"  [entregue via {', '.join(alerta.canais_entrega)}]" if alerta.canais_entrega else ""
        print(f"  {alerta.resumo()}{entregue}")
        print(f"      {alerta.acao_recomendada}")

    _secao("Arquivos gerados")
    for nome, caminho in resultado.arquivos_gerados.items():
        print(f"  {nome:24s} {caminho}")
    print()


def demonstrar_tempo_real(
    paciente_id: str, atraso: float, raiz_amostras: Path | None = None,
    dir_saida: Path | None = None,
) -> None:
    """Percorre a serie de sinais vitais como se fosse o monitor do leito."""
    _titulo(f"Monitoramento em tempo real - paciente {paciente_id}")

    amostras = localizar_amostras(paciente_id, raiz_amostras)
    if "sinais_vitais" not in amostras:
        print("Serie de sinais vitais nao encontrada.")
        return

    df = anomaly_detection.carregar_sinais_vitais(amostras["sinais_vitais"])
    print(
        f"{len(df)} leituras de {df['timestamp'].iloc[0]:%H:%M} a "
        f"{df['timestamp'].iloc[-1]:%H:%M} (1 por minuto)"
    )
    canais = canais_padrao(dir_saida)
    print("canais de entrega: " + "; ".join(descrever_canais(canais)) + "\n")

    motor = MotorDeAlertas(canais=canais)
    total_alertas = 0
    for instante, achados in anomaly_detection.monitorar_em_tempo_real(df):
        for achado in achados:
            if achado.severidade == "informativo":
                # O achado se refere a leitura suspeita, que e um minuto antes
                # do instante em que a sequencia se mostrou nao confirmada.
                marca = (achado.instante or instante.isoformat())[11:16]
                print(f"{marca}  (nota) {achado.descricao[:88]}")
            elif achado.tipo == "anomalia_multivariada":
                print(
                    f"{instante:%H:%M}  (isolation forest) episodio multivariado desde "
                    f"{achado.evidencias.get('inicio', '')[11:16]}: {achado.descricao[:70]}"
                )
        alertas = motor.gerar(paciente_id, achados, agora=instante)
        for alerta in alertas:
            total_alertas += 1
            entregue = f" [{', '.join(alerta.canais_entrega)}]" if alerta.canais_entrega else ""
            print(f"{instante:%H:%M}  {alerta.resumo()}{entregue}")
        if atraso:
            time.sleep(atraso)

    print(f"\n{total_alertas} alertas emitidos durante o turno.")
    if motor.entregas_falhas:
        print(f"{motor.entregas_falhas} entrega(s) falharam; veja auditoria_monitoramento.jsonl.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--paciente-id", default="PAC-0006")
    parser.add_argument("--raiz-amostras", type=Path, help="Pasta com uma subpasta por paciente.")
    parser.add_argument("--dir-saida", type=Path, help="Pasta para os relatorios gerados.")
    parser.add_argument(
        "--todos",
        action="store_true",
        help="Processa todos os pacientes das amostras e grava o resumo consolidado.",
    )
    parser.add_argument(
        "--tempo-real",
        action="store_true",
        help="Mostra os alertas de sinais vitais surgindo minuto a minuto.",
    )
    parser.add_argument(
        "--atraso",
        type=float,
        default=0.0,
        help="Segundos entre leituras no modo tempo real (use 0.01 na gravacao).",
    )
    parser.add_argument(
        "--provedor-azure",
        default="auto",
        choices=["auto", "azure", "offline", "local"],
        help=(
            "Provedor de Speech to Text e Text Analytics (padrao: auto). "
            "'local' transcreve com faster-whisper na propria maquina, para "
            "demonstracao sem credencial Azure."
        ),
    )
    parser.add_argument(
        "--audio",
        default=ARQUIVO_AUDIO_PADRAO,
        metavar="ARQUIVO",
        help=(
            "Nome do WAV da consulta dentro da pasta do paciente (padrao: consulta.wav). "
            "Use consulta_fala.wav para o audio com fala inteligivel gerado por "
            "python -m fase4.data.gerar_fala_tts."
        ),
    )
    parser.add_argument(
        "--backend-pose",
        default="anotado",
        choices=["anotado", "mediapipe"],
        help="Estimador de pose (padrao: anotado, usado com as amostras sinteticas).",
    )
    parser.add_argument(
        "--backend-detector",
        default="anotado",
        choices=["anotado", "yolov8"],
        help="Detector de objetos (padrao: anotado).",
    )
    parser.add_argument(
        "--area-critica",
        type=_parse_area_critica,
        default=None,
        metavar="X1,Y1,X2,Y2",
        help=(
            "ROI manual da area critica em pixels (ex.: 250,178,296,220). Necessaria "
            "com --backend-detector yolov8, que nao reconhece drenos."
        ),
    )
    args = parser.parse_args()
    if args.atraso < 0:
        parser.error("--atraso deve ser maior ou igual a zero")

    # No modo tempo real o log de auditoria em stdout atrapalha a leitura.
    if args.tempo_real:
        os.environ.setdefault("FASE4_LOG_SILENCIOSO", "1")

    if args.tempo_real:
        demonstrar_tempo_real(args.paciente_id, args.atraso, args.raiz_amostras, args.dir_saida)
        return

    if args.todos:
        resumo = processar_lote(
            sorted(p.name for p in args.raiz_amostras.iterdir() if p.is_dir())
            if args.raiz_amostras else list(PACIENTES_PADRAO),
            raiz_amostras=args.raiz_amostras,
            dir_saida=args.dir_saida,
            provedor_azure=args.provedor_azure,
            backend_pose=args.backend_pose,
            backend_detector=args.backend_detector,
            area_critica=args.area_critica,
            arquivo_audio=args.audio,
        )
        _titulo("Resumo consolidado")
        for paciente in resumo["pacientes"]:
            print(
                f"  {paciente['paciente_id']}  risco {paciente['risco']:5.1f} "
                f"({paciente['nivel_risco']:9s})  alertas: {paciente['total_alertas']:2d}  "
                f"quadros: {', '.join(paciente['sindromes']) or '-'}"
            )
        print()
        return

    demonstrar_paciente(
        args.paciente_id, args.provedor_azure, args.backend_pose, args.backend_detector,
        args.raiz_amostras, args.dir_saida, area_critica=args.area_critica,
        arquivo_audio=args.audio,
    )


if __name__ == "__main__":
    main()
