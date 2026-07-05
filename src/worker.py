"""Entrypoint do Python Worker com segurança e encaminhamento ASGI."""

from __future__ import annotations

import hashlib
import json
from urllib.parse import urlparse

import asgi
from js import Object
from pyodide.ffi import to_js
from workers import Response, WorkerEntrypoint

from api import app
from edge_security import validate_turnstile


def _json_response(status: int, detail: str, headers: dict[str, str] | None = None):
    return Response.from_json(
        {"detail": detail},
        status=status,
        headers=headers,
    )


def _client_key(request, route: str) -> str:
    """Gera uma chave pseudônima sem persistir o IP em logs ou bancos."""

    address = request.headers.get("CF-Connecting-IP") or "desconhecido"
    digest = hashlib.sha256(address.encode("utf-8")).hexdigest()
    return f"{route}:{digest}"


class Default(WorkerEntrypoint):
    """Aplica controles de borda antes de encaminhar ao FastAPI."""

    async def _apply_rate_limit(self, request, path: str):
        binding_name = {
            "/predict": "PREDICT_RATE_LIMITER",
            "/interpret": "INTERPRET_RATE_LIMITER",
        }.get(path)
        if binding_name is None:
            return None

        limiter = getattr(self.env, binding_name)
        options = to_js(
            {"key": _client_key(request, path)},
            dict_converter=Object.fromEntries,
        )
        result = await limiter.limit(options)
        if result.success:
            return None

        print(
            json.dumps(
                {"event": "rate_limited", "endpoint": path},
                ensure_ascii=False,
            )
        )
        return _json_response(
            429,
            "Limite temporário de requisições excedido.",
            headers={"Retry-After": "60"},
        )

    async def _validate_interpretation_access(self, request, path: str):
        if path != "/interpret":
            return None

        token = request.headers.get("X-Turnstile-Token") or ""
        secret = str(getattr(self.env, "TURNSTILE_SECRET_KEY", ""))
        if not secret:
            print(json.dumps({"event": "turnstile_secret_missing"}))
            return _json_response(503, "Proteção Turnstile não configurada.")
        try:
            valid = await validate_turnstile(
                token,
                secret,
                request.headers.get("CF-Connecting-IP"),
            )
        except Exception as error:
            print(
                json.dumps(
                    {
                        "event": "turnstile_validation_failed",
                        "error_type": type(error).__name__,
                    }
                )
            )
            return _json_response(503, "Não foi possível validar o Turnstile.")
        if not valid:
            return _json_response(403, "Validação Turnstile inválida ou expirada.")
        return None

    async def fetch(self, request):
        path = urlparse(request.url).path.rstrip("/") or "/"
        rate_limit_response = await self._apply_rate_limit(request, path)
        if rate_limit_response is not None:
            return rate_limit_response
        turnstile_response = await self._validate_interpretation_access(request, path)
        if turnstile_response is not None:
            return turnstile_response
        return await asgi.fetch(app, request, self.env)
