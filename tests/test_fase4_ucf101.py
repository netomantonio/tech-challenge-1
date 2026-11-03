"""Procedencia e integridade do recorte UCF101, sem acesso a rede."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fase4.data import baixar_ucf101 as ucf


class Ucf101Tests(unittest.TestCase):
    def test_id_rejeita_caminho_e_classe_fora_do_recorte(self):
        for nome in ("../v_JumpingJack_g01_c01.avi", "v_Archery_g01_c01.avi", "x.mp4"):
            with self.assertRaises(ValueError):
                ucf.id_do_clipe(nome)

    def test_download_valida_hash_e_reutiliza_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            nome = "v_JumpingJack_g01_c01.avi"
            conteudo = b"video de teste"
            manifesto = raiz / "manifesto.json"
            manifesto.write_text(json.dumps([{"clipe": nome, "url": "https://example.org/video", "sha256_original": hashlib.sha256(conteudo).hexdigest(), "revisao_espelho": "rev"}]), encoding="utf-8")
            with patch.object(ucf, "MANIFESTO", manifesto), patch("requests.get", return_value=Mock(content=conteudo)) as get:
                self.assertEqual(list(ucf._baixar_clipes(raiz))[0][1], conteudo)
                pasta = raiz / ucf.id_do_clipe(nome)
                pasta.mkdir()
                (pasta / nome).write_bytes(conteudo)
                get.reset_mock()
                self.assertEqual(list(ucf._baixar_clipes(raiz))[0][1], conteudo)
                get.assert_not_called()

    def test_download_corrompido_e_rejeitado(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            manifesto = raiz / "manifesto.json"
            manifesto.write_text(json.dumps([{"clipe": "v_JumpingJack_g01_c01.avi", "url": "https://example.org/video", "sha256_original": "0" * 64, "revisao_espelho": "rev"}]), encoding="utf-8")
            with patch.object(ucf, "MANIFESTO", manifesto), patch("requests.get", return_value=Mock(content=b"corrompido")):
                with self.assertRaisesRegex(ValueError, "Hash inesperado"):
                    list(ucf._baixar_clipes(raiz))
