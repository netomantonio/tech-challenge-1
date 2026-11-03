"""Confere as credenciais da Azure com uma chamada minima em cada servico.

    python -m fase4.verificar_azure
    python -m fase4.verificar_azure --audio fase4/data/amostras/PAC-0006/consulta_fala.wav

Le o `.env` da raiz (via `fase4.config`), mostra o que esta configurado e,
quando ha credencial e SDK, envia uma frase curta ao Text Analytics e um WAV
ao Speech to Text. Sai com codigo 1 se algum servico configurado falhar.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fase4.azure_services import (
    ServicoAzureIndisponivelError,
    analisar_texto,
    status_servicos,
    transcrever_audio,
)
from fase4.config import (
    DIR_DADOS,
    ENV_LANGUAGE_ENDPOINT,
    ENV_LANGUAGE_KEY,
    ENV_SPEECH_KEY,
    ENV_SPEECH_REGION,
    IDIOMA_PADRAO,
)

FRASE_TESTE = (
    "O paciente relata dor forte no peito e falta de ar desde a madrugada, "
    "mas diz que o sono melhorou depois da medicacao."
)
AUDIO_PADRAO = DIR_DADOS / "amostras" / "PAC-0006" / "consulta_fala.wav"


def _mascarar(valor: str | None) -> str:
    if not valor:
        return "(nao definida)"
    if len(valor) <= 8:
        return "*" * len(valor)
    return f"{valor[:4]}...{valor[-4:]} ({len(valor)} caracteres)"


def _mostrar_configuracao() -> None:
    import os

    print("Variaveis de ambiente")
    print(f"  {ENV_SPEECH_KEY:24s} {_mascarar(os.getenv(ENV_SPEECH_KEY))}")
    print(f"  {ENV_SPEECH_REGION:24s} {os.getenv(ENV_SPEECH_REGION) or '(nao definida)'}")
    print(f"  {ENV_LANGUAGE_KEY:24s} {_mascarar(os.getenv(ENV_LANGUAGE_KEY))}")
    print(f"  {ENV_LANGUAGE_ENDPOINT:24s} {os.getenv(ENV_LANGUAGE_ENDPOINT) or '(nao definida)'}")
    print("\nServicos")
    for servico, estado in status_servicos().items():
        print(
            f"  {servico:22s} credencial={'sim' if estado['credencial'] else 'nao':3s} "
            f"sdk={'sim' if estado['sdk'] else 'nao':3s} provedor={estado['provedor_efetivo']}"
        )


def _testar_text_analytics() -> bool:
    print("\nText Analytics: enviando uma frase de teste...")
    try:
        resultado = analisar_texto(FRASE_TESTE, provedor="azure", idioma=IDIOMA_PADRAO)
    except ServicoAzureIndisponivelError as erro:
        print(f"  nao configurado: {erro}")
        return False
    except Exception as erro:  # noqa: BLE001 - mostra o erro do servico ao usuario
        print(f"  FALHOU: {type(erro).__name__}: {erro}")
        return False
    scores = ", ".join(f"{k}={v:.2f}" for k, v in resultado.scores_sentimento.items())
    print(f"  sentimento: {resultado.sentimento} ({scores})")
    print(f"  frases-chave: {', '.join(resultado.frases_chave[:6]) or '-'}")
    print(f"  entidades clinicas: {len(resultado.entidades)}")
    print(f"  termos criticos: {[t.termo for t in resultado.termos_criticos] or '-'}")
    return True


def _testar_speech(caminho: Path) -> bool:
    print(f"\nSpeech to Text: transcrevendo {caminho}...")
    if not caminho.is_file():
        print("  arquivo nao encontrado; gere com `python -m fase4.data.gerar_fala_tts` "
              "ou informe --audio")
        return False
    try:
        resultado = transcrever_audio(caminho, provedor="azure", idioma=IDIOMA_PADRAO)
    except ServicoAzureIndisponivelError as erro:
        print(f"  nao configurado: {erro}")
        return False
    except Exception as erro:  # noqa: BLE001
        print(f"  FALHOU: {type(erro).__name__}: {erro}")
        return False
    texto = resultado.texto.strip()
    print(f"  segmentos: {len(resultado.segmentos)}  confianca media: {resultado.confianca:.2f}")
    print(f"  texto: {texto[:200]}{'...' if len(texto) > 200 else ''}")
    return bool(texto)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audio", type=Path, default=AUDIO_PADRAO,
                        help="WAV PCM 16 bits mono para o teste de transcricao.")
    parser.add_argument("--sem-chamadas", action="store_true",
                        help="So mostra a configuracao, sem chamar a Azure.")
    args = parser.parse_args(argv)

    _mostrar_configuracao()
    if args.sem_chamadas:
        return 0
    ok_texto = _testar_text_analytics()
    ok_fala = _testar_speech(args.audio)
    print("\nResultado:", "OK" if ok_texto and ok_fala else "FALHA")
    return 0 if ok_texto and ok_fala else 1


if __name__ == "__main__":
    sys.exit(main())
