"""Conversor Markdown -> HTML para os relatorios gerados pela Fase 4.

Cobre exatamente o subconjunto que os geradores do projeto escrevem: titulos
`#`..`###`, paragrafos, listas com `-`, citacoes com `>`, blocos de codigo
cercados, tabelas com linha de alinhamento (`---:` vira coluna numerica),
negrito, codigo em linha, enfase e links `https://`. O texto e escapado
**antes** da conversao, por isso a saida pode ir para `innerHTML` sob uma
CSP estrita. Nao e um Markdown completo de proposito: a interface nao
recebe texto de terceiros, so os relatorios do proprio pipeline.
"""

from __future__ import annotations

import html
import re

_RE_CODIGO = re.compile(r"`([^`]+)`")
_RE_NEGRITO = re.compile(r"\*\*(.+?)\*\*")
_RE_ENFASE = re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])")
_RE_LINK = re.compile(r"(https?://[^\s<>\"')\]]+)")
_RE_TITULO = re.compile(r"^(#{1,3})\s+(.*)$")
_RE_SEPARADOR_TABELA = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def _inline(texto: str) -> str:
    """Formatacao em linha sobre texto ja escapado."""
    protegidos: list[str] = []

    def guardar(match: re.Match[str]) -> str:
        protegidos.append(f"<code>{match.group(1)}</code>")
        return f"\x00{len(protegidos) - 1}\x00"

    def ligar(match: re.Match[str]) -> str:
        url = match.group(1)
        cauda = ""
        while url and url[-1] in ".,;:":
            cauda = url[-1] + cauda
            url = url[:-1]
        return f'<a href="{url}" rel="noopener" target="_blank">{url}</a>{cauda}'

    texto = _RE_CODIGO.sub(guardar, texto)
    texto = _RE_LINK.sub(ligar, texto)
    texto = _RE_NEGRITO.sub(r"<strong>\1</strong>", texto)
    texto = _RE_ENFASE.sub(r"<em>\1</em>", texto)
    for indice, trecho in enumerate(protegidos):
        texto = texto.replace(f"\x00{indice}\x00", trecho)
    return texto


def _celulas(linha: str) -> list[str]:
    linha = linha.strip()
    if linha.startswith("|"):
        linha = linha[1:]
    if linha.endswith("|"):
        linha = linha[:-1]
    return [c.strip() for c in linha.split("|")]


def _alinhamentos(separador: str) -> list[str]:
    classes = []
    for celula in _celulas(separador):
        direita = celula.endswith(":")
        esquerda = celula.startswith(":")
        if direita and not esquerda:
            classes.append("num")
        elif direita and esquerda:
            classes.append("centro")
        else:
            classes.append("")
    return classes


def _tabela(linhas: list[str]) -> str:
    cabecalho = _celulas(linhas[0])
    classes = _alinhamentos(linhas[1]) if len(linhas) > 1 else [""] * len(cabecalho)
    while len(classes) < len(cabecalho):
        classes.append("")

    def abrir(tag: str, indice: int) -> str:
        classe = classes[indice] if indice < len(classes) else ""
        return f'<{tag} class="{classe}">' if classe else f"<{tag}>"

    saida = ["<table>", "<thead><tr>"]
    for indice, celula in enumerate(cabecalho):
        saida.append(f"{abrir('th', indice)}{_inline(celula)}</th>")
    saida.append("</tr></thead><tbody>")
    for linha in linhas[2:]:
        saida.append("<tr>")
        for indice, celula in enumerate(_celulas(linha)):
            saida.append(f"{abrir('td', indice)}{_inline(celula)}</td>")
        saida.append("</tr>")
    saida.append("</tbody></table>")
    return "".join(saida)


def _paragrafo(linhas: list[str]) -> str:
    partes = []
    for indice, linha in enumerate(linhas):
        quebra = linha.endswith("  ") and indice < len(linhas) - 1
        partes.append(_inline(linha.strip()) + ("<br>" if quebra else ""))
    return "<p>" + " ".join(p for p in partes if p) + "</p>"


def markdown_para_html(texto: str) -> str:
    """Converte o Markdown dos relatorios em HTML seguro para `innerHTML`."""
    linhas = html.escape(texto, quote=False).replace("\r\n", "\n").split("\n")
    saida: list[str] = []
    paragrafo: list[str] = []
    lista: list[str] = []
    citacao: list[str] = []

    def fechar_blocos() -> None:
        nonlocal paragrafo, lista, citacao
        if paragrafo:
            saida.append(_paragrafo(paragrafo))
            paragrafo = []
        if lista:
            saida.append("<ul>" + "".join(f"<li>{_inline(item)}</li>" for item in lista) + "</ul>")
            lista = []
        if citacao:
            saida.append("<blockquote><p>" + _inline(" ".join(citacao)) + "</p></blockquote>")
            citacao = []

    indice = 0
    total = len(linhas)
    while indice < total:
        linha = linhas[indice]
        bruta = linha.rstrip()

        if bruta.startswith("```"):
            fechar_blocos()
            indice += 1
            codigo: list[str] = []
            while indice < total and not linhas[indice].rstrip().startswith("```"):
                codigo.append(linhas[indice])
                indice += 1
            saida.append("<pre><code>" + "\n".join(codigo) + "</code></pre>")
            indice += 1
            continue

        titulo = _RE_TITULO.match(bruta)
        if titulo:
            fechar_blocos()
            nivel = len(titulo.group(1))
            saida.append(f"<h{nivel}>{_inline(titulo.group(2).strip())}</h{nivel}>")
            indice += 1
            continue

        if bruta.startswith("|") and indice + 1 < total and _RE_SEPARADOR_TABELA.match(linhas[indice + 1].strip()):
            fechar_blocos()
            tabela = [bruta, linhas[indice + 1].rstrip()]
            indice += 2
            while indice < total and linhas[indice].strip().startswith("|"):
                tabela.append(linhas[indice].rstrip())
                indice += 1
            saida.append(_tabela(tabela))
            continue

        if bruta.startswith("- ") or bruta.startswith("* "):
            if paragrafo or citacao:
                fechar_blocos()
            lista.append(bruta[2:].strip())
            indice += 1
            continue

        if bruta.startswith("&gt;"):
            if paragrafo or lista:
                fechar_blocos()
            citacao.append(bruta[4:].strip())
            indice += 1
            continue

        if not bruta.strip():
            fechar_blocos()
            indice += 1
            continue

        if lista and bruta.startswith("  "):
            lista[-1] += " " + bruta.strip()
            indice += 1
            continue
        if lista or citacao:
            fechar_blocos()
        paragrafo.append(linha)
        indice += 1

    fechar_blocos()
    return "\n".join(saida)
