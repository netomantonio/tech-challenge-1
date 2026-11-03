"""Backends reais de video da Fase 4 (MediaPipe e YOLOv8) com modulos falsos.

Os pacotes `mediapipe` e `ultralytics` sao opcionais e pesados, entao os
testes injetam substitutos em `sys.modules` (mesmo padrao de
`tests/test_fase4_azure.py`). O que se verifica aqui e o codigo do projeto:
mapeamento dos 33 landmarks para as 13 articulacoes, escala para pixels,
leitura das caixas do YOLO, a ROI manual de area critica, a regra da mao por
keypoint do punho e o criterio de "duas pessoas em cena" para o detector
generico. O video sintetico `fisioterapia.mp4` e lido de verdade com o OpenCV.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fase4 import video_analysis
from fase4.cli_demo import _parse_area_critica
from fase4.data import gerar_dados_sinteticos as gerador
from fase4.video_analysis import (
    CLASSE_AREA_CRITICA_MANUAL,
    QuadroObjetos,
    QuadroPose,
    Articulacao,
    Objeto,
    analisar_sessao,
    detectar_objetos_yolov8,
    extrair_pose_mediapipe,
    injetar_area_critica_manual,
)

try:  # O OpenCV ja esta no ambiente do projeto; sem ele os testes de video nao rodam.
    import cv2  # noqa: F401
    TEM_CV2 = True
except ImportError:  # pragma: no cover
    TEM_CV2 = False

TOTAL_QUADROS = int(gerador.DURACAO_VIDEO_S * gerador.FPS_VIDEO)


class AmostraDeVideo(unittest.TestCase):
    """Gera o PAC-0006 uma vez e carrega os keypoints anotados como gabarito."""

    diretorio: tempfile.TemporaryDirectory
    raiz: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.diretorio = tempfile.TemporaryDirectory()
        cls.raiz = Path(cls.diretorio.name)
        gerador.gerar_amostras_paciente("PAC-0006", seed=42, destino_raiz=cls.raiz)
        cls.pasta = cls.raiz / "PAC-0006"
        cls.video = cls.pasta / "fisioterapia.mp4"
        with (cls.pasta / "fisioterapia.keypoints.json").open(encoding="utf-8") as f:
            cls.keypoints = json.load(f)["quadros"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.diretorio.cleanup()

    # --- landmarks falsos construidos a partir das anotacoes -----------------

    @classmethod
    def landmarks_do_quadro(cls, indice: int) -> list:
        """33 landmarks normalizados; as 13 articulacoes do projeto vem da anotacao."""
        pontos = [SimpleNamespace(x=0.0, y=0.0, visibility=0.0) for _ in range(33)]
        quadro = cls.keypoints[min(indice, len(cls.keypoints) - 1)]
        for nome, idx in video_analysis._INDICES_MEDIAPIPE.items():
            ponto = quadro["articulacoes"][nome]
            pontos[idx] = SimpleNamespace(
                x=ponto["x"] / gerador.LARGURA_VIDEO,
                y=ponto["y"] / gerador.ALTURA_VIDEO,
                visibility=ponto["confianca"],
            )
        return pontos


@contextmanager
def mediapipe_legado_falso(landmarks_por_quadro):
    """`mediapipe` com a API `solutions.pose` (serie 0.10)."""
    contador = {"quadro": 0}

    class PoseFalsa:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def process(self, _rgb):
            pontos = landmarks_por_quadro(contador["quadro"])
            contador["quadro"] += 1
            if pontos is None:
                return SimpleNamespace(pose_landmarks=None)
            return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=pontos))

    mp = SimpleNamespace(solutions=SimpleNamespace(pose=SimpleNamespace(Pose=PoseFalsa)))
    with patch.dict("sys.modules", {"mediapipe": mp}):
        yield contador


@contextmanager
def mediapipe_tasks_falso(landmarks_por_quadro):
    """`mediapipe` 1.x: sem `solutions`, com `tasks.python.vision.PoseLandmarker`."""
    contador = {"quadro": 0, "timestamps": []}

    class LandmarkerFalso:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def detect_for_video(self, _imagem, timestamp_ms):
            contador["timestamps"].append(timestamp_ms)
            pontos = landmarks_por_quadro(contador["quadro"])
            contador["quadro"] += 1
            return SimpleNamespace(pose_landmarks=[pontos] if pontos is not None else [])

    vision = SimpleNamespace(
        PoseLandmarkerOptions=lambda **kwargs: SimpleNamespace(**kwargs),
        RunningMode=SimpleNamespace(VIDEO="VIDEO"),
        PoseLandmarker=SimpleNamespace(create_from_options=lambda _opcoes: LandmarkerFalso()),
    )
    mp_python = SimpleNamespace(BaseOptions=lambda **kwargs: SimpleNamespace(**kwargs), vision=vision)
    tasks = SimpleNamespace(python=mp_python)
    mp = SimpleNamespace(
        tasks=tasks,
        Image=lambda **kwargs: SimpleNamespace(**kwargs),
        ImageFormat=SimpleNamespace(SRGB="SRGB"),
    )
    modules = {
        "mediapipe": mp,
        "mediapipe.tasks": tasks,
        "mediapipe.tasks.python": mp_python,
        "mediapipe.tasks.python.vision": vision,
    }
    with patch.dict("sys.modules", modules):
        yield contador


@contextmanager
def ultralytics_falso(caixas_por_quadro, nomes=None):
    """`ultralytics.YOLO` cujo `predict(stream=True)` rende um quadro por chamada."""
    nomes = nomes or {0: "person", 1: "dreno", 2: "bed"}

    class Caixa:
        def __init__(self, classe, conf, xyxy):
            self.cls = classe
            self.conf = conf
            self.xyxy = [xyxy]

    class YOLOFalso:
        def __init__(self, modelo):
            self.modelo = modelo
            self.names = nomes

        def predict(self, source, stream, verbose):
            assert stream is True
            for quadro in range(TOTAL_QUADROS):
                yield SimpleNamespace(boxes=[Caixa(*c) for c in caixas_por_quadro(quadro)])

    modulo = SimpleNamespace(YOLO=YOLOFalso)
    with patch.dict("sys.modules", {"ultralytics": modulo}):
        yield


def _pessoas_no_quadro(quadro: int) -> list:
    """Duas pessoas em cena, menos na janela em que o profissional sai."""
    inicio, fim = gerador.JANELA_SEM_PROFISSIONAL
    caixas = [(0, 0.9, (150.0, 100.0, 330.0, 320.0))]  # paciente
    if not (inicio <= quadro < fim):
        caixas.append((0, 0.88, (20.0, 110.0, 100.0, 300.0)))  # profissional
    return caixas


@unittest.skipUnless(TEM_CV2, "OpenCV nao instalado")
class MediaPipeBackendTests(AmostraDeVideo):
    def test_api_legada_mapeia_landmarks_para_pixels(self) -> None:
        with mediapipe_legado_falso(self.landmarks_do_quadro) as contador:
            quadros, fps = extrair_pose_mediapipe(self.video)
        self.assertEqual(len(quadros), TOTAL_QUADROS)
        self.assertEqual(contador["quadro"], TOTAL_QUADROS)
        self.assertAlmostEqual(fps, gerador.FPS_VIDEO, places=3)
        primeiro = quadros[0].articulacoes
        self.assertEqual(set(primeiro), set(video_analysis._INDICES_MEDIAPIPE))
        esperado = self.keypoints[0]["articulacoes"]["punho_esquerdo"]
        self.assertAlmostEqual(primeiro["punho_esquerdo"].x, esperado["x"], places=3)
        self.assertAlmostEqual(primeiro["punho_esquerdo"].y, esperado["y"], places=3)
        self.assertAlmostEqual(primeiro["punho_esquerdo"].confianca, esperado["confianca"], places=5)

    def test_api_tasks_usa_timestamps_crescentes_e_nao_baixa_modelo(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".task", delete=False) as modelo:
            caminho_modelo = Path(modelo.name)
        try:
            with mediapipe_tasks_falso(self.landmarks_do_quadro) as contador, \
                    patch("fase4.video_analysis.urllib.request.urlretrieve") as download:
                quadros, _ = extrair_pose_mediapipe(self.video, caminho_modelo=caminho_modelo)
            download.assert_not_called()
            self.assertEqual(len(quadros), TOTAL_QUADROS)
            timestamps = contador["timestamps"]
            self.assertEqual(timestamps[0], 0)
            self.assertTrue(all(b > a for a, b in zip(timestamps, timestamps[1:])))
            self.assertEqual(timestamps[30], 1000)
        finally:
            caminho_modelo.unlink(missing_ok=True)

    def test_quadro_sem_pessoa_fica_sem_articulacoes(self) -> None:
        def sem_pessoa(indice):
            return None if indice % 2 else self.landmarks_do_quadro(indice)

        with mediapipe_legado_falso(sem_pessoa):
            quadros, _ = extrair_pose_mediapipe(self.video)
        self.assertEqual(len(quadros), TOTAL_QUADROS)
        self.assertEqual(quadros[1].articulacoes, {})
        self.assertTrue(quadros[0].articulacoes)
        medidas = video_analysis.medir_quadros(quadros)
        self.assertIsNone(medidas[1].inclinacao_tronco)
        self.assertIsNotNone(medidas[0].inclinacao_tronco)

    def test_sem_mediapipe_instalado_falha_com_mensagem_clara(self) -> None:
        with patch.dict("sys.modules", {"mediapipe": None}):
            with self.assertRaises(video_analysis.BackendIndisponivelError):
                extrair_pose_mediapipe(self.video)


@unittest.skipUnless(TEM_CV2, "OpenCV nao instalado")
class YoloBackendTests(AmostraDeVideo):
    def test_le_caixas_e_marca_classes_criticas(self) -> None:
        def caixas(quadro):
            return _pessoas_no_quadro(quadro) + [(1, 0.7, (250.0, 178.0, 296.0, 220.0))]

        with ultralytics_falso(caixas):
            quadros = detectar_objetos_yolov8(self.video, classes_criticas=("dreno",))
        self.assertEqual(len(quadros), TOTAL_QUADROS)
        dreno = next(o for o in quadros[0].objetos if o.classe == "dreno")
        self.assertTrue(dreno.area_critica)
        self.assertEqual(dreno.caixa, (250.0, 178.0, 296.0, 220.0))
        self.assertFalse(any(o.area_critica for o in quadros[0].objetos if o.classe == "person"))
        self.assertAlmostEqual(quadros[30].instante_s, 1.0, places=3)

    def test_sem_ultralytics_instalado_falha_com_mensagem_clara(self) -> None:
        with patch.dict("sys.modules", {"ultralytics": None}):
            with self.assertRaises(video_analysis.BackendIndisponivelError):
                detectar_objetos_yolov8(self.video)


class AreaCriticaManualTests(AmostraDeVideo):
    def _objetos_sem_mao_nem_dreno(self) -> Path:
        with (self.pasta / "fisioterapia.objetos.json").open(encoding="utf-8") as f:
            dados = json.load(f)
        for quadro in dados["quadros"]:
            quadro["objetos"] = [
                o for o in quadro["objetos"] if o["classe"] not in ("mao_paciente", "dreno")
            ]
        destino = self.pasta / "fisioterapia.objetos_sem_mao.json"
        destino.write_text(json.dumps(dados), encoding="utf-8")
        return destino

    def test_roi_manual_com_punho_reproduz_a_invasao_anotada(self) -> None:
        referencia = analisar_sessao(
            "PAC-0006",
            caminho_keypoints=self.pasta / "fisioterapia.keypoints.json",
            caminho_objetos=self.pasta / "fisioterapia.objetos.json",
        )
        self.assertEqual(referencia.origem_area_critica, "anotada")
        self.assertEqual(referencia.regra_mao, "caixa_detector")

        resultado = analisar_sessao(
            "PAC-0006",
            caminho_keypoints=self.pasta / "fisioterapia.keypoints.json",
            caminho_objetos=self._objetos_sem_mao_nem_dreno(),
            area_critica=gerador.AREA_CRITICA_DRENO,
        )
        self.assertEqual(resultado.origem_area_critica, "manual")
        self.assertEqual(resultado.regra_mao, "keypoint_punho")

        invasoes_ref = [a for a in referencia.achados if a.tipo == "invasao_area_critica"]
        invasoes = [a for a in resultado.achados if a.tipo == "invasao_area_critica"]
        self.assertTrue(invasoes_ref)
        self.assertEqual(len(invasoes), len(invasoes_ref))
        for ref, novo in zip(invasoes_ref, invasoes):
            self.assertEqual(novo.evidencias["origem_mao"], "keypoint_punho")
            self.assertAlmostEqual(
                novo.evidencias["duracao_s"], ref.evidencias["duracao_s"], delta=0.5
            )
        # As demais regras nao mudam quando so a origem da mao muda.
        self.assertEqual(
            sorted(a.tipo for a in referencia.achados if a.tipo != "invasao_area_critica"),
            sorted(a.tipo for a in resultado.achados if a.tipo != "invasao_area_critica"),
        )

    def test_roi_manual_sem_arquivo_de_objetos_cria_os_quadros(self) -> None:
        resultado = analisar_sessao(
            "PAC-0006",
            caminho_keypoints=self.pasta / "fisioterapia.keypoints.json",
            area_critica=gerador.AREA_CRITICA_DRENO,
        )
        self.assertEqual(resultado.regra_mao, "keypoint_punho")
        self.assertIn("invasao_area_critica", {a.tipo for a in resultado.achados})
        # Sem detector nao ha como saber se o profissional esta em cena.
        self.assertNotIn("profissional_ausente", {a.tipo for a in resultado.achados})

    def test_roi_fora_do_caminho_da_mao_nao_gera_invasao(self) -> None:
        resultado = analisar_sessao(
            "PAC-0006",
            caminho_keypoints=self.pasta / "fisioterapia.keypoints.json",
            area_critica=(0.0, 0.0, 20.0, 20.0),
        )
        self.assertNotIn("invasao_area_critica", {a.tipo for a in resultado.achados})

    def test_injecao_valida_a_roi(self) -> None:
        pose = [QuadroPose(0, 0.0, {"punho_esquerdo": Articulacao(1, 1, 0.9)})]
        with self.assertRaises(ValueError):
            injetar_area_critica_manual([], pose, (10, 10, 5, 20))
        with self.assertRaises(ValueError):
            injetar_area_critica_manual([], pose, (1, 2, 3))  # type: ignore[arg-type]
        quadros = injetar_area_critica_manual(
            [QuadroObjetos(0, 0.0, [Objeto("person", 0.9, (0, 0, 10, 10))])], pose, (1, 1, 5, 5)
        )
        classes = [o.classe for o in quadros[0].objetos]
        self.assertEqual(classes, ["person", CLASSE_AREA_CRITICA_MANUAL])
        # Injetar duas vezes nao duplica a ROI.
        quadros = injetar_area_critica_manual(quadros, pose, (1, 1, 5, 5))
        self.assertEqual([o.classe for o in quadros[0].objetos], ["person", CLASSE_AREA_CRITICA_MANUAL])

    def test_relatorio_registra_origem_da_area_e_da_mao(self) -> None:
        resultado = analisar_sessao(
            "PAC-0006",
            caminho_keypoints=self.pasta / "fisioterapia.keypoints.json",
            area_critica=gerador.AREA_CRITICA_DRENO,
        )
        relatorio = video_analysis.relatorio_markdown(resultado)
        self.assertIn("`manual`", relatorio)
        self.assertIn("`keypoint_punho`", relatorio)
        self.assertIn("pose detectada em", relatorio)


class PessoasGenericasTests(unittest.TestCase):
    def _quadros(self, pessoas_por_quadro: list[int], fps: float = 30.0) -> list[QuadroObjetos]:
        return [
            QuadroObjetos(
                quadro=i,
                instante_s=i / fps,
                objetos=[Objeto("person", 0.9, (0, 0, 10, 10)) for _ in range(n)],
            )
            for i, n in enumerate(pessoas_por_quadro)
        ]

    def test_uma_pessoa_nao_comprova_ausencia_profissional(self) -> None:
        quadros = self._quadros([2] * 30 + [1] * 90 + [2] * 30)
        achados = video_analysis.avaliar_eventos(quadros, 30.0, pessoas_genericas=True)
        tipos = [a.tipo for a in achados]
        self.assertEqual(tipos, ["acompanhamento_nao_confirmado"])
        self.assertEqual(achados[0].severidade, "informativo")
        self.assertAlmostEqual(achados[0].evidencias["duracao_s"], 3.0, places=2)
        self.assertIn("sem identificacao", achados[0].evidencias["criterio"])

    def test_zero_pessoas_nao_gera_alerta_de_ausencia(self) -> None:
        from fase4.alertas import MotorDeAlertas
        achados = video_analysis.avaliar_eventos(self._quadros([0] * 600), 30, pessoas_genericas=True)
        self.assertEqual([a.tipo for a in achados], ["deteccao_pessoas_insuficiente"])
        self.assertEqual(MotorDeAlertas().gerar("PAC-X", achados), [])

    def test_lacuna_nao_soma_tempo_de_pessoa_unica(self) -> None:
        quadros = self._quadros([1] * 45 + [0] * 30 + [1] * 45)
        self.assertEqual(video_analysis.avaliar_eventos(quadros, 30, pessoas_genericas=True), [])

    def test_duas_pessoas_nao_geram_queixa(self) -> None:
        self.assertEqual(video_analysis.avaliar_eventos(self._quadros([2] * 90), 30, pessoas_genericas=True), [])

    def test_sem_pessoas_genericas_a_classe_person_nao_conta(self) -> None:
        quadros = self._quadros([2] * 150)
        achados = video_analysis.avaliar_eventos(quadros, 30.0, pessoas_genericas=False)
        self.assertEqual([a.tipo for a in achados], ["profissional_ausente"])

    def test_ausencia_curta_nao_dispara(self) -> None:
        quadros = self._quadros([2] * 30 + [1] * 30 + [2] * 30)
        achados = video_analysis.avaliar_eventos(quadros, 30.0, pessoas_genericas=True)
        self.assertEqual(achados, [])


@unittest.skipUnless(TEM_CV2, "OpenCV nao instalado")
class SessaoComBackendsReaisTests(AmostraDeVideo):
    def test_sem_pose_medidas_sao_nulas_e_relatorio_nao_inventa_zero(self) -> None:
        with patch.object(video_analysis, "extrair_pose_mediapipe", return_value=([QuadroPose(i, i / 30, {}) for i in range(90)], 30)), patch.object(video_analysis, "detectar_objetos_yolov8", return_value=[QuadroObjetos(i, i / 30, []) for i in range(90)]):
            resultado = analisar_sessao("PAC-X", caminho_video=self.video, backend_pose="mediapipe", backend_detector="yolov8")
        self.assertTrue(all(v is None for v in resultado.to_dict()["resumo_angulos"].values()))
        self.assertIn("Não avaliado", video_analysis.relatorio_markdown(resultado))
        self.assertTrue(resultado.avisos)
        self.assertNotIn("profissional_ausente", {a.tipo for a in resultado.achados})

    def test_contexto_exploratorio_mede_sem_alerta_clinico(self) -> None:
        with mediapipe_legado_falso(self.landmarks_do_quadro), ultralytics_falso(_pessoas_no_quadro):
            resultado = analisar_sessao("UCF-X", caminho_video=self.video, backend_pose="mediapipe", backend_detector="yolov8", contexto="exploratorio")
        self.assertGreater(resultado.resumo_angulos["abducao_maxima_ombro_esquerdo"], 120)
        self.assertTrue(all(a.severidade == "informativo" for a in resultado.achados))
        from fase4.multimodal_fusion import fundir
        self.assertEqual(fundir("UCF-X", resultado.achados).risco, 0)

    def test_mediapipe_e_yolov8_com_roi_manual_reproduzem_os_eventos(self) -> None:
        with mediapipe_legado_falso(self.landmarks_do_quadro), ultralytics_falso(_pessoas_no_quadro):
            resultado = analisar_sessao(
                "PAC-0006",
                caminho_video=self.video,
                backend_pose="mediapipe",
                backend_detector="yolov8",
                area_critica=gerador.AREA_CRITICA_DRENO,
            )
        self.assertEqual(resultado.backend_pose, "mediapipe")
        self.assertEqual(resultado.backend_detector, "yolov8")
        self.assertEqual(resultado.quadros_com_pose, TOTAL_QUADROS)
        self.assertEqual(resultado.regra_mao, "keypoint_punho")
        tipos = {a.tipo for a in resultado.achados}
        for esperado in (
            "desvio_postural_tronco",
            "amplitude_ombro_esquerdo_excedida",
            "assimetria_entre_ombros",
            "invasao_area_critica",
            "acompanhamento_nao_confirmado",
        ):
            self.assertIn(esperado, tipos)
        ausente = next(a for a in resultado.achados if a.tipo == "acompanhamento_nao_confirmado")
        self.assertIn("sem identificacao", ausente.evidencias["criterio"])
        self.assertAlmostEqual(ausente.evidencias["duracao_s"], 4.0, delta=0.1)

    def test_yolov8_sem_roi_nao_tem_como_avaliar_area_critica(self) -> None:
        with mediapipe_legado_falso(self.landmarks_do_quadro), ultralytics_falso(_pessoas_no_quadro):
            resultado = analisar_sessao(
                "PAC-0006", caminho_video=self.video,
                backend_pose="mediapipe", backend_detector="yolov8",
            )
        self.assertEqual(resultado.origem_area_critica, "nenhuma")
        self.assertEqual(resultado.regra_mao, "nenhuma")
        self.assertNotIn("invasao_area_critica", {a.tipo for a in resultado.achados})


class ParseAreaCriticaTests(unittest.TestCase):
    def test_aceita_quatro_numeros(self) -> None:
        self.assertEqual(_parse_area_critica("250,178,296,220"), (250.0, 178.0, 296.0, 220.0))
        self.assertIsNone(_parse_area_critica(None))

    def test_rejeita_formatos_invalidos(self) -> None:
        import argparse

        for texto in ("1,2,3", "a,b,c,d", "10,10,5,20"):
            with self.assertRaises(argparse.ArgumentTypeError):
                _parse_area_critica(texto)


if __name__ == "__main__":
    unittest.main()
