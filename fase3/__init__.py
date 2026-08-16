"""Modulo da Fase 3: assistente medico virtual."""

from __future__ import annotations

import os

# O backend local usa PyTorch/PEFT. Em ambientes onde TensorFlow tambem esta
# instalado, transformers pode tentar importa-lo cedo demais e falhar por
# incompatibilidades de dependencias opcionais.
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ.setdefault("USE_TORCH", "1")
