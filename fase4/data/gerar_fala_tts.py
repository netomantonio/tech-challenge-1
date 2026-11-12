"""Gera a consulta com fala inteligivel em pt-BR usando a voz do Windows.

O `consulta.wav` das amostras e um sinal vozeado sintetico, otimo para
calibrar o extrator de atributos acusticos, mas sem palavras. Para demonstrar
Speech to Text em fala real - com o provedor `local` (faster-whisper) agora e
com a Azure quando houver credencial - este script le o texto de referencia
(`consulta.transcricao.json`) e sintetiza cada frase com o TTS do sistema
(SAPI via `pyttsx3`; PowerShell `System.Speech` como reserva), preservando as
pausas da referencia.

Saida, na pasta do paciente: `consulta_fala.wav` (mono, 16 kHz, 16 bits) e
`consulta_fala.transcricao.json` com os instantes reais de cada frase.

    python -m fase4.data.gerar_fala_tts                    # PAC-0006 e PAC-0002
    python -m fase4.data.gerar_fala_tts --paciente-id PAC-0006 --voz Maria

Limitacao assumida: a voz sintetica e estavel e bem articulada. Ela serve
para a camada de linguagem (transcricao, termos criticos, sentimento); as
regras acusticas de fadiga e disartria nao devem disparar nela, e isso e o
comportamento esperado, nao um defeito.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fase4.data.gerar_dados_sinteticos import DIR_AMOSTRAS, PACIENTES_PADRAO, TAXA_AUDIO

NOME_SAIDA = "consulta_fala"
PAUSA_PADRAO_S = 0.6
SILENCIO_INICIAL_S = 0.5
SILENCIO_FINAL_S = 0.8
PISTAS_PT_BR = ("pt-br", "pt_br", "ptb", "brazil", "brasil", "portuguese", "maria", "daniel")


class SinteseIndisponivelError(RuntimeError):
    """Nenhum mecanismo de TTS pode ser usado nesta maquina."""


# ---------------------------------------------------------------------------
# Mecanismos de sintese
# ---------------------------------------------------------------------------


def _voz_e_pt_br(voz) -> bool:
    campos = [str(getattr(voz, "id", "")), str(getattr(voz, "name", ""))]
    for lingua in getattr(voz, "languages", None) or []:
        campos.append(lingua.decode("utf-8", "ignore") if isinstance(lingua, bytes) else str(lingua))
    texto = " ".join(campos).lower()
    return any(pista in texto for pista in PISTAS_PT_BR)


def escolher_voz_pyttsx3(engine, preferida: str | None = None):
    """Devolve a voz pedida ou a primeira pt-BR; None usa a padrao do sistema."""
    vozes = engine.getProperty("voices") or []
    if preferida:
        for voz in vozes:
            if preferida.lower() in f"{voz.id} {voz.name}".lower():
                return voz
        raise SinteseIndisponivelError(
            f"voz '{preferida}' nao encontrada; disponiveis: "
            + ", ".join(v.name for v in vozes)
        )
    for voz in vozes:
        if _voz_e_pt_br(voz):
            return voz
    return None


def sintetizar_pyttsx3(frases: list[str], destino_dir: Path, voz: str | None, velocidade: int | None) -> tuple[list[Path], str]:
    import pyttsx3

    engine = pyttsx3.init()
    escolhida = escolher_voz_pyttsx3(engine, voz)
    if escolhida is not None:
        engine.setProperty("voice", escolhida.id)
    if velocidade:
        engine.setProperty("rate", velocidade)
    arquivos: list[Path] = []
    for indice, frase in enumerate(frases):
        caminho = destino_dir / f"frase_{indice:02d}.wav"
        engine.save_to_file(frase, str(caminho))
        arquivos.append(caminho)
    engine.runAndWait()
    try:
        engine.stop()
    except Exception:  # pragma: no cover - alguns drivers nao implementam stop
        pass
    faltando = [a for a in arquivos if not a.exists() or a.stat().st_size == 0]
    if faltando:
        raise SinteseIndisponivelError("pyttsx3 nao gravou todas as frases")
    nome_voz = escolhida.name if escolhida is not None else "padrao do sistema"
    return arquivos, nome_voz


_SCRIPT_POWERSHELL = r"""
param([string]$Entrada, [string]$Saida, [string]$Voz)
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
if ($Voz) { $s.SelectVoice($Voz) } else {
  $pt = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'pt-BR' } | Select-Object -First 1
  if ($pt) { $s.SelectVoice($pt.VoiceInfo.Name) }
}
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$linhas = Get-Content -Path $Entrada -Encoding UTF8
$i = 0
foreach ($linha in $linhas) {
  $arquivo = Join-Path $Saida ("frase_{0:D2}.wav" -f $i)
  $s.SetOutputToWaveFile($arquivo, $fmt)
  $s.Speak($linha)
  $s.SetOutputToNull()
  $i++
}
Write-Output $s.Voice.Name
$s.Dispose()
"""


def sintetizar_powershell(frases: list[str], destino_dir: Path, voz: str | None) -> tuple[list[Path], str]:
    """Reserva para quando o pyttsx3 nao esta instalado: System.Speech via PowerShell."""
    if sys.platform != "win32":
        raise SinteseIndisponivelError("a reserva por PowerShell so existe no Windows")
    entrada = destino_dir / "frases.txt"
    entrada.write_text("\n".join(frases), encoding="utf-8")
    script = destino_dir / "tts.ps1"
    script.write_text(_SCRIPT_POWERSHELL, encoding="utf-8")
    comando = [
        "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
        "-Entrada", str(entrada), "-Saida", str(destino_dir), "-Voz", voz or "",
    ]
    processo = subprocess.run(comando, capture_output=True, text=True, timeout=300)
    if processo.returncode != 0:
        raise SinteseIndisponivelError(processo.stderr.strip() or "PowerShell falhou")
    arquivos = [destino_dir / f"frase_{i:02d}.wav" for i in range(len(frases))]
    if not all(a.exists() for a in arquivos):
        raise SinteseIndisponivelError("System.Speech nao gravou todas as frases")
    return arquivos, processo.stdout.strip().splitlines()[-1] if processo.stdout.strip() else "System.Speech"


# ---------------------------------------------------------------------------
# Montagem do WAV final
# ---------------------------------------------------------------------------


def ler_wav_mono(caminho: Path) -> tuple[np.ndarray, int]:
    """Le um WAV PCM e devolve float em [-1, 1] mono, mais a taxa."""
    with wave.open(str(caminho), "rb") as wav:
        canais, largura, taxa, quadros = (
            wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
        )
        bruto = wav.readframes(quadros)
    if largura == 2:
        sinal = np.frombuffer(bruto, dtype="<i2").astype(np.float32) / 32768.0
    elif largura == 1:
        sinal = (np.frombuffer(bruto, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif largura == 4:
        sinal = np.frombuffer(bruto, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise SinteseIndisponivelError(f"largura de amostra nao suportada: {largura} bytes")
    if canais > 1:
        sinal = sinal.reshape(-1, canais).mean(axis=1)
    return sinal, taxa


def reamostrar(sinal: np.ndarray, taxa_origem: int, taxa_destino: int) -> np.ndarray:
    """Reamostragem linear; suficiente para voz a 16 kHz."""
    if taxa_origem == taxa_destino or sinal.size == 0:
        return sinal.astype(np.float32)
    duracao = sinal.size / taxa_origem
    novos = int(round(duracao * taxa_destino))
    posicoes_origem = np.linspace(0.0, duracao, num=sinal.size, endpoint=False)
    posicoes_destino = np.linspace(0.0, duracao, num=novos, endpoint=False)
    return np.interp(posicoes_destino, posicoes_origem, sinal).astype(np.float32)


def aparar_silencio(sinal: np.ndarray, taxa: int, limiar: float = 0.01, margem_s: float = 0.05) -> np.ndarray:
    """Remove o silencio que o TTS acrescenta antes e depois da frase."""
    acima = np.flatnonzero(np.abs(sinal) > limiar)
    if acima.size == 0:
        return sinal
    margem = int(margem_s * taxa)
    inicio = max(0, int(acima[0]) - margem)
    fim = min(sinal.size, int(acima[-1]) + 1 + margem)
    return sinal[inicio:fim]


def montar_consulta(
    trechos: list[np.ndarray], pausas: list[float], taxa: int
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Concatena as frases com as pausas e devolve o sinal e os instantes."""
    partes = [np.zeros(int(SILENCIO_INICIAL_S * taxa), dtype=np.float32)]
    instantes: list[tuple[float, float]] = []
    cursor = SILENCIO_INICIAL_S
    for indice, trecho in enumerate(trechos):
        instantes.append((cursor, cursor + trecho.size / taxa))
        partes.append(trecho)
        cursor += trecho.size / taxa
        pausa = pausas[indice] if indice < len(pausas) else SILENCIO_FINAL_S
        partes.append(np.zeros(int(pausa * taxa), dtype=np.float32))
        cursor += pausa
    sinal = np.concatenate(partes)
    pico = float(np.max(np.abs(sinal))) if sinal.size else 0.0
    if pico > 0:
        sinal = sinal / pico * 0.9
    return sinal.astype(np.float32), instantes


def salvar_wav(sinal: np.ndarray, caminho: Path, taxa: int = TAXA_AUDIO) -> None:
    inteiros = np.clip(sinal * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(caminho), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(taxa)
        wav.writeframes(inteiros.tobytes())


def pausas_da_referencia(segmentos: list[dict]) -> list[float]:
    pausas: list[float] = []
    for atual, proximo in zip(segmentos, segmentos[1:]):
        pausa = float(proximo["inicio_s"]) - float(atual["fim_s"])
        pausas.append(max(0.2, pausa) if pausa > 0 else PAUSA_PADRAO_S)
    pausas.append(SILENCIO_FINAL_S)
    return pausas


# ---------------------------------------------------------------------------
# Orquestracao
# ---------------------------------------------------------------------------


def gerar_fala(
    paciente_id: str,
    raiz_amostras: Path | None = None,
    nome_saida: str = NOME_SAIDA,
    voz: str | None = None,
    velocidade: int | None = None,
    mecanismo: str = "auto",
) -> dict:
    pasta = (raiz_amostras or DIR_AMOSTRAS) / paciente_id
    referencia = pasta / "consulta.transcricao.json"
    if not referencia.exists():
        raise FileNotFoundError(
            f"{referencia} nao existe; rode antes python -m fase4.data.gerar_dados_sinteticos"
        )
    with referencia.open(encoding="utf-8") as f:
        dados = json.load(f)
    segmentos = dados.get("segmentos", [])
    frases = [str(s["texto"]) for s in segmentos]
    if not frases:
        raise ValueError(f"{referencia} nao tem segmentos")

    with tempfile.TemporaryDirectory() as temporario:
        destino_dir = Path(temporario)
        erros: list[str] = []
        arquivos: list[Path] = []
        nome_voz = ""
        mecanismo_usado = ""
        tentativas = ["pyttsx3", "powershell"] if mecanismo == "auto" else [mecanismo]
        for tentativa in tentativas:
            try:
                if tentativa == "pyttsx3":
                    arquivos, nome_voz = sintetizar_pyttsx3(frases, destino_dir, voz, velocidade)
                else:
                    arquivos, nome_voz = sintetizar_powershell(frases, destino_dir, voz)
                mecanismo_usado = tentativa
                break
            except Exception as erro:  # noqa: BLE001 - queremos tentar a reserva
                erros.append(f"{tentativa}: {erro}")
        if not arquivos:
            raise SinteseIndisponivelError("; ".join(erros))

        trechos = []
        for arquivo in arquivos:
            sinal, taxa = ler_wav_mono(arquivo)
            sinal = aparar_silencio(reamostrar(sinal, taxa, TAXA_AUDIO), TAXA_AUDIO)
            trechos.append(sinal)

    sinal, instantes = montar_consulta(trechos, pausas_da_referencia(segmentos), TAXA_AUDIO)
    caminho_wav = pasta / f"{nome_saida}.wav"
    salvar_wav(sinal, caminho_wav)

    transcricao = {
        "paciente_id": paciente_id,
        "idioma": dados.get("idioma", "pt-BR"),
        "duracao_s": round(sinal.size / TAXA_AUDIO, 3),
        "origem": f"tts_{mecanismo_usado}",
        "voz": nome_voz,
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "segmentos": [
            {
                "inicio_s": round(inicio, 3),
                "fim_s": round(fim, 3),
                "texto": frase,
                "confianca": 0.95,
            }
            for (inicio, fim), frase in zip(instantes, frases)
        ],
    }
    caminho_json = pasta / f"{nome_saida}.transcricao.json"
    caminho_json.write_text(json.dumps(transcricao, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "wav": str(caminho_wav),
        "transcricao": str(caminho_json),
        "voz": nome_voz,
        "mecanismo": mecanismo_usado,
        "duracao_s": transcricao["duracao_s"],
        "frases": len(frases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paciente-id", action="append", help="Repita para mais de um paciente.")
    parser.add_argument("--raiz-amostras", type=Path, default=None)
    parser.add_argument("--voz", default=None, help="Trecho do nome da voz SAPI (ex.: Maria).")
    parser.add_argument("--velocidade", type=int, default=None, help="Palavras por minuto (pyttsx3).")
    parser.add_argument("--mecanismo", choices=["auto", "pyttsx3", "powershell"], default="auto")
    args = parser.parse_args()

    pacientes = args.paciente_id or list(PACIENTES_PADRAO)
    for paciente in pacientes:
        try:
            info = gerar_fala(
                paciente, args.raiz_amostras, voz=args.voz, velocidade=args.velocidade,
                mecanismo=args.mecanismo,
            )
        except SinteseIndisponivelError as erro:
            print(f"{paciente}: sintese indisponivel ({erro})")
            print("Instale pyttsx3 (pip install -r requirements-fase4.txt) ou rode no Windows.")
            raise SystemExit(1) from erro
        print(f"{paciente}: {info['frases']} frases, {info['duracao_s']:.1f} s, voz '{info['voz']}' ({info['mecanismo']})")
        print(f"  {info['wav']}")
        print(f"  {info['transcricao']}")


if __name__ == "__main__":
    main()
