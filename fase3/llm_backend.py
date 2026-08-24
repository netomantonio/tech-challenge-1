"""Backend de LLM plugavel para o assistente medico da Fase 3.

Tres modos, escolhidos por ``backend=`` ou, para usos avancados, pela variavel
de ambiente ``FASE3_LLM_BACKEND``:

- ``"local"`` (padrao): carrega o modelo base instrucional + adapter LoRA treinado em
  ``fase3/finetuning/train_lora.py`` via ``transformers``/``peft``. A classe
  local aplica o chat template do tokenizer, usa decodificacao deterministica e devolve
  somente os tokens novos. Requer ``requirements-fase3.txt`` instalado.
- ``"groq"``: reaproveita o padrao de chamada HTTP + retry em 429
  ja usado em ``src/llm_interpretation.py`` (Fase 2), so que exposto como um
  LLM compativel com LangChain.
- ``"fake"``: LLM determinístico usado em testes automatizados, sem rede e
  sem dependencias pesadas.
"""

from __future__ import annotations

import os
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from langchain_core.language_models.llms import LLM

from fase3.prompting import SYSTEM_PROMPT_CLINICO

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_LLM_BACKEND = "local"
DEFAULT_LOCAL_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_LOCAL_LORA_SCALE = 0.75
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_CACHE_POINTER = PROJECT_ROOT / ".cache" / "fase3-hf-home.txt"
PROMOTED_MODEL_CONFIG_PATH = PROJECT_ROOT / ".cache" / "fase3-promoted-model.json"
DEFAULT_LOCAL_ADAPTER_PATH = (
    PROJECT_ROOT
    / "resultados"
    / "fase3"
    / "finetuning"
    / "qwen2.5-1.5b-v4"
    / "lora_adapter"
)
LEGACY_DISTILGPT2_ADAPTER_PATH = (
    Path(__file__).resolve().parent.parent
    / "resultados"
    / "fase3"
    / "finetuning"
    / "smoke"
    / "lora_adapter"
)
GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"

class LLMUnavailableError(RuntimeError):
    """Erro ao chamar o provedor de LLM (chave ausente, rede indisponivel, etc.)."""


def _parse_retry_delay(mensagem: str, default: float = 2.0) -> float:
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", mensagem, re.IGNORECASE)
    return float(match.group(1)) if match else default


def get_promoted_local_config() -> dict[str, Any]:
    """Retorna adapter/escala promovidos, com v4 como fallback versionado."""
    fallback = {
        "base_model": DEFAULT_LOCAL_BASE_MODEL,
        "adapter_path": str(DEFAULT_LOCAL_ADAPTER_PATH),
        "lora_scale": DEFAULT_LOCAL_LORA_SCALE,
        "promovido_em": None,
    }
    try:
        config = json.loads(PROMOTED_MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return fallback
    adapter = Path(str(config.get("adapter_path", "")))
    if not adapter.is_absolute():
        adapter = PROJECT_ROOT / adapter
    if not (adapter / "adapter_model.safetensors").exists():
        return fallback
    try:
        scale = float(config["lora_scale"])
    except (KeyError, TypeError, ValueError):
        return fallback
    if not 0.0 < scale <= 1.0:
        return fallback
    return {**config, "adapter_path": str(adapter.resolve()), "lora_scale": scale}


def _resolver_local_adapter_path(base_model: str, adapter_path: Optional[str]) -> Optional[str]:
    adapter_resolvido = adapter_path or os.getenv("FASE3_LOCAL_ADAPTER_PATH")
    if adapter_resolvido:
        return adapter_resolvido
    promovido = get_promoted_local_config()
    if base_model == promovido.get("base_model") and Path(promovido["adapter_path"]).exists():
        return str(promovido["adapter_path"])
    if base_model == "distilgpt2" and LEGACY_DISTILGPT2_ADAPTER_PATH.exists():
        return str(LEGACY_DISTILGPT2_ADAPTER_PATH)
    return None


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


class LocalAdapterLLM(LLM):
    """LLM LangChain para inferencia deterministica com modelo + adapter LoRA."""

    hf_model: Any
    tokenizer: Any
    max_new_tokens: int = 160
    max_input_tokens: int = 2048
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    lora_scale: float = DEFAULT_LOCAL_LORA_SCALE

    @property
    def _llm_type(self) -> str:  # noqa: D401
        return "local_lora"

    def _formatar_chat(self, prompt: str) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT_CLINICO},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        return (
            f"Sistema: {SYSTEM_PROMPT_CLINICO}\n\n"
            f"Pergunta e contexto:\n{prompt}\n\nResposta:"
        )

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        import torch

        texto = self._formatar_chat(prompt)
        ids = self.tokenizer(
            texto,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]

        limite_modelo = int(
            getattr(self.hf_model.config, "max_position_embeddings", self.max_input_tokens)
            or self.max_input_tokens
        )
        limite_entrada = max(
            64,
            min(self.max_input_tokens, limite_modelo - self.max_new_tokens),
        )
        if len(ids) > limite_entrada:
            # Preserva o system prompt no inicio e a pergunta no final.
            cabeca = max(32, limite_entrada // 4)
            ids = torch.cat((ids[:cabeca], ids[-(limite_entrada - cabeca) :]))

        device = next(self.hf_model.parameters()).device
        input_ids = ids.unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = self.hf_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        resposta = self.tokenizer.decode(
            output[0, input_ids.shape[1] :],
            skip_special_tokens=True,
        ).strip()
        if stop:
            cortes = [resposta.find(token) for token in stop if token and token in resposta]
            if cortes:
                resposta = resposta[: min(cortes)].rstrip()
        return resposta or "Nao ha informacao suficiente para responder com seguranca."


def _criar_llm_local(base_model: Optional[str], adapter_path: Optional[str], **kwargs: Any) -> LLM:
    if not os.getenv("HF_HOME") and LOCAL_CACHE_POINTER.exists():
        cache_configurado = LOCAL_CACHE_POINTER.read_text(encoding="utf-8").strip()
        if cache_configurado:
            os.environ["HF_HOME"] = cache_configurado
            # O setup e responsavel pelo download. Inferencia nunca deve iniciar
            # uma transferencia grande no meio de uma consulta clinica.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

    os.environ["USE_TF"] = "0"
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    os.environ.setdefault("USE_TORCH", "1")
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercitado so sem requirements-fase3.txt
        raise SystemExit(
            "Backend 'local' requer as dependencias de requirements-fase3.txt "
            f"(erro original: {exc})"
        ) from exc

    promovido = get_promoted_local_config()
    base_model = base_model or os.getenv("FASE3_LOCAL_BASE_MODEL") or promovido.get("base_model") or DEFAULT_LOCAL_BASE_MODEL
    revision = kwargs.get("revision") or promovido.get("revision")
    trust_remote_code = bool(kwargs.get("trust_remote_code", promovido.get("trust_remote_code", False)))
    use_adapter = bool(kwargs.get("use_adapter", True))
    adapter_path = _resolver_local_adapter_path(base_model, adapter_path) if use_adapter else None
    if use_adapter and not adapter_path:
        raise LLMUnavailableError(
            "Adapter LoRA local nao encontrado. Treine o modelo com "
            "`python -m fase3.finetuning.train_lora` ou informe `adapter_path`."
        )

    load_options = {"revision": revision, "trust_remote_code": trust_remote_code}
    tokenizer = AutoTokenizer.from_pretrained(base_model, **load_options)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    usar_cuda = torch.cuda.is_available()
    model_kwargs = {**load_options, **({"dtype": torch.float16} if usar_cuda else {})}
    modelo = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    if use_adapter:
        modelo = PeftModel.from_pretrained(modelo, adapter_path)
    escala_padrao = (
        promovido["lora_scale"]
        if use_adapter and base_model == promovido.get("base_model")
        else DEFAULT_LOCAL_LORA_SCALE
    )
    escala_informada = kwargs.get("lora_scale")
    lora_scale = float(
        escala_informada
        if escala_informada is not None
        else os.getenv("FASE3_LOCAL_LORA_SCALE", str(escala_padrao))
    )
    if use_adapter and not 0.0 < lora_scale <= 1.0:
        raise ValueError("lora_scale deve estar no intervalo (0, 1].")
    if not use_adapter:
        lora_scale = 0.0
    else:
        for modulo in modelo.modules():
            scaling = getattr(modulo, "scaling", None)
            if isinstance(scaling, dict):
                for adapter, valor in scaling.items():
                    scaling[adapter] = valor * lora_scale
    if usar_cuda:
        modelo = modelo.to("cuda")
    modelo.eval()

    return LocalAdapterLLM(
        hf_model=modelo,
        tokenizer=tokenizer,
        max_new_tokens=kwargs.get("max_new_tokens", 160),
        max_input_tokens=kwargs.get("max_input_tokens", 2048),
        # O modelo redige o plano factual recebido no prompt; penalizar repeticao
        # degrada a preservacao de fatos e identificadores autorizados.
        repetition_penalty=kwargs.get("repetition_penalty", 1.0),
        no_repeat_ngram_size=kwargs.get("no_repeat_ngram_size", 0),
        lora_scale=lora_scale,
    )


def resolver_backend(backend: Optional[str] = None) -> str:
    """Resolve o backend; uma escolha explicita sempre vence o ambiente."""
    return (backend or os.getenv("FASE3_LLM_BACKEND") or DEFAULT_LLM_BACKEND).strip().lower()


def get_llm(backend: Optional[str] = None, **kwargs: Any) -> LLM:
    """Fabrica o LLM plugavel do assistente medico de acordo com o backend escolhido."""
    backend = resolver_backend(backend)
    if backend == "groq":
        return GroqLLM(
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model", os.getenv("GROQ_LLM_MODEL", DEFAULT_GROQ_MODEL)),
        )
    if backend == "local":
        local_kwargs = {k: v for k, v in kwargs.items() if k not in {"base_model", "adapter_path"}}
        return _criar_llm_local(kwargs.get("base_model"), kwargs.get("adapter_path"), **local_kwargs)
    if backend == "fake":
        return FakeLLM(respostas=kwargs.get("respostas", []))
    raise ValueError(f"Backend de LLM desconhecido: {backend!r} (use 'groq', 'local' ou 'fake')")
