"""Backend de LLM plugavel para o assistente medico da Fase 3.

Tres modos, escolhidos por ``backend=`` ou pela variavel de ambiente
``FASE3_LLM_BACKEND``:

- ``"groq"`` (padrao): reaproveita o padrao de chamada HTTP + retry em 429
  ja usado em ``src/llm_interpretation.py`` (Fase 2), so que exposto como um
  LLM compativel com LangChain.
- ``"local"``: carrega o modelo base + adapter LoRA treinado em
  ``fase3/finetuning/train_lora.py`` via ``transformers``/``peft``, exposto
  como um ``HuggingFacePipeline`` do LangChain. Requer
  ``requirements-fase3.txt`` instalado.
- ``"fake"``: LLM determinístico usado em testes automatizados, sem rede e
  sem dependencias pesadas.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Optional

import httpx
from langchain_core.language_models.llms import LLM

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT_CLINICO = (
    "Voce e um assistente virtual de apoio clinico interno do hospital. "
    "Responda apenas com base nos protocolos e informacoes de contexto "
    "fornecidos. Nunca prescreva medicamentos, doses ou tratamentos "
    "diretamente: toda sugestao terapeutica deve ser explicitamente "
    "condicionada a validacao de um medico responsavel. Se a informacao "
    "necessaria nao estiver no contexto fornecido, diga isso claramente "
    "em vez de inventar uma resposta."
)


class LLMUnavailableError(RuntimeError):
    """Erro ao chamar o provedor de LLM (chave ausente, rede indisponivel, etc.)."""


def _parse_retry_delay(mensagem: str, default: float = 2.0) -> float:
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", mensagem, re.IGNORECASE)
    return float(match.group(1)) if match else default


class GroqLLM(LLM):
    """LLM compativel com LangChain que chama a API de chat da Groq via HTTP."""

    api_key: Optional[str] = None
    model: str = DEFAULT_GROQ_MODEL
    system_prompt: str = SYSTEM_PROMPT_CLINICO
    temperature: float = 0.2
    max_retries: int = 3
    timeout_seconds: float = 30.0

    @property
    def _llm_type(self) -> str:  # noqa: D401
        return "groq"

    def _call(self, prompt: str, stop: Optional[list[str]] = None, run_manager: Any = None, **kwargs: Any) -> str:
        api_key = self.api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise LLMUnavailableError("GROQ_API_KEY nao configurada.")

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {api_key}"}

        ultimo_erro: Optional[Exception] = None
        for tentativa in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    resposta = client.post(GROQ_CHAT_COMPLETIONS_URL, json=payload, headers=headers)
                if resposta.status_code == 429:
                    delay = _parse_retry_delay(resposta.text)
                    time.sleep(delay)
                    continue
                resposta.raise_for_status()
                dados = resposta.json()
                return dados["choices"][0]["message"]["content"]
            except httpx.HTTPError as exc:
                ultimo_erro = exc
                break

        raise LLMUnavailableError(f"Falha ao chamar a API da Groq: {ultimo_erro}")


class FakeLLM(LLM):
    """LLM deterministico para testes: devolve respostas pre-configuradas em ordem."""

    respostas: list[str] = []
    resposta_padrao: str = (
        "Nao ha informacao suficiente no contexto fornecido para responder com seguranca."
    )
    prompts_recebidos: list[str] = []

    @property
    def _llm_type(self) -> str:  # noqa: D401
        return "fake"

    def _call(self, prompt: str, stop: Optional[list[str]] = None, run_manager: Any = None, **kwargs: Any) -> str:
        self.prompts_recebidos.append(prompt)
        indice = len(self.prompts_recebidos) - 1
        if indice < len(self.respostas):
            return self.respostas[indice]
        return self.resposta_padrao


def _criar_llm_local(base_model: Optional[str], adapter_path: Optional[str], **kwargs: Any) -> LLM:
    os.environ.setdefault("USE_TF", "0")
    try:
        from langchain_community.llms import HuggingFacePipeline
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except ImportError as exc:  # pragma: no cover - exercitado so sem requirements-fase3.txt
        raise SystemExit(
            "Backend 'local' requer as dependencias de requirements-fase3.txt "
            f"(erro original: {exc})"
        ) from exc

    base_model = base_model or os.getenv("FASE3_LOCAL_BASE_MODEL", "distilgpt2")
    adapter_path = adapter_path or os.getenv("FASE3_LOCAL_ADAPTER_PATH")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    modelo = AutoModelForCausalLM.from_pretrained(base_model)
    if adapter_path:
        modelo = PeftModel.from_pretrained(modelo, adapter_path)

    text_gen = pipeline(
        "text-generation",
        model=modelo,
        tokenizer=tokenizer,
        max_new_tokens=kwargs.get("max_new_tokens", 200),
    )
    return HuggingFacePipeline(pipeline=text_gen)


def get_llm(backend: Optional[str] = None, **kwargs: Any) -> LLM:
    """Fabrica o LLM plugavel do assistente medico de acordo com o backend escolhido."""
    backend = backend or os.getenv("FASE3_LLM_BACKEND", "groq")
    if backend == "groq":
        return GroqLLM(
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model", os.getenv("GROQ_LLM_MODEL", DEFAULT_GROQ_MODEL)),
        )
    if backend == "local":
        return _criar_llm_local(kwargs.get("base_model"), kwargs.get("adapter_path"), **kwargs)
    if backend == "fake":
        return FakeLLM(respostas=kwargs.get("respostas", []))
    raise ValueError(f"Backend de LLM desconhecido: {backend!r} (use 'groq', 'local' ou 'fake')")
