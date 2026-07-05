"""Validação assíncrona do Turnstile na borda."""

from __future__ import annotations

from typing import Any

import httpx


TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


async def validate_turnstile(
    token: str,
    secret: str,
    remote_ip: str | None = None,
    client: Any | None = None,
) -> bool:
    """Valida um token sem registrar seu conteúdo nem o endereço do cliente."""

    if not token or not secret:
        return False
    payload = {"secret": secret, "response": token}
    if remote_ip:
        payload["remoteip"] = remote_ip

    if client is not None:
        response = await client.post(TURNSTILE_VERIFY_URL, data=payload)
    else:
        async with httpx.AsyncClient(timeout=10.0) as http_client:
            response = await http_client.post(TURNSTILE_VERIFY_URL, data=payload)
    response.raise_for_status()
    result = response.json()
    return bool(result.get("success", False))
