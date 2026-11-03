"""Avalia o recorte UCF101 com MediaPipe e YOLOv8, sem inferencia clinica."""
from __future__ import annotations

import json
from importlib.metadata import version
from datetime import datetime, timezone

from fase4.data.baixar_ucf101 import DIR_UCF101
from fase4.config import DIR_RESULTADOS
from fase4.web_service import Deps, Paciente, executar_processamento


def main() -> None:
    resultados = []
    deps = Deps()
    for pasta in sorted(DIR_UCF101.iterdir()):
        if not (pasta / "fonte.json").exists():
            continue
        paciente = Paciente(pasta.name, "ucf101", DIR_UCF101, DIR_RESULTADOS / "ucf101")
        resultado = executar_processamento(deps, paciente, {
            "backend_pose": "mediapipe", "backend_detector": "yolov8",
            "provedor_azure": "offline", "gravar_video_anotado": True,
        })
        video = resultado["video"]
        fonte = json.loads((pasta / "fonte.json").read_text(encoding="utf-8"))
        resultados.append({"id": pasta.name, "fonte": fonte, "video": video})
        print(f"{pasta.name}: pose em {video['quadros_com_pose']}/{video['quadros_analisados']}", flush=True)
    if not resultados:
        raise ValueError("Prepare os clipes com python -m fase4.data.baixar_ucf101")
    saida = DIR_RESULTADOS / "ucf101"
    dados = {"executado_em": datetime.now(timezone.utc).isoformat(),
             "versoes": {n: version(n) for n in ("mediapipe", "ultralytics", "opencv-contrib-python")},
             "resultados": resultados}
    (saida / "avaliacao.json").write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
    linhas = ["# Avaliação exploratória — UCF101", "",
              "Utilizamos três clipes para verificar a extração de postura em pessoas filmadas. "
              "Os modelos foram usados com pesos pré-treinados, sem ajuste ao recorte. "
              "Os rótulos de ação não indicam anomalias clínicas; por isso, não aplicamos os limites de pós-operatório.", "",
              "| Clipe | Quadros | Com pose | Cobertura |", "| --- | ---: | ---: | ---: |"]
    for r in resultados:
        v = r["video"]
        linhas.append(f"| {r['fonte']['clipe']} | {v['quadros_analisados']} | {v['quadros_com_pose']} | {100 * v['quadros_com_pose'] / max(1, v['quadros_analisados']):.1f}% |")
    linhas += ["", "A cobertura mede disponibilidade da pose, não acurácia dos pontos. "
               "Não avaliamos classificação das 101 ações, qualidade clínica do exercício ou identidade profissional. "
               "A revisão do espelho, os hashes e as versões estão em `avaliacao.json`.", ""]
    (saida / "relatorio_ucf101.md").write_text("\n".join(linhas), encoding="utf-8")


if __name__ == "__main__":
    main()
