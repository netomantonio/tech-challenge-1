from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output" / "entrega_tech_challenge_fase3.md"
TARGET = ROOT / "output" / "pdf" / "entrega_tech_challenge_fase3.pdf"


def section(markdown: str, title: str) -> str:
    marker = f"## {title}"
    start = markdown.index(marker) + len(marker)
    rest = markdown[start:]
    next_heading = rest.find("\n## ")
    return rest[:next_heading].strip() if next_heading >= 0 else rest.strip()


def parse_table(block: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line.startswith("|") or "---" in line:
            continue
        cells = [cell.strip().replace("`", "") for cell in line.strip("|").split("|")]
        rows.append(cells)
    return rows


def parse_paragraphs(block: str) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if line.startswith("|") or line.startswith("#"):
            continue
        current.append(line.strip("`"))
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def make_styles() -> dict[str, ParagraphStyle]:
    fonts = {
        "regular": r"C:\Windows\Fonts\arial.ttf",
        "bold": r"C:\Windows\Fonts\arialbd.ttf",
    }
    pdfmetrics.registerFont(TTFont("Arial", fonts["regular"]))
    pdfmetrics.registerFont(TTFont("Arial-Bold", fonts["bold"]))

    base = getSampleStyleSheet()
    return {
        "cover": ParagraphStyle(
            name="CoverTitle",
            parent=base["Title"],
            fontName="Arial-Bold",
            fontSize=24,
            leading=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#174A7C"),
            spaceAfter=14,
        ),
        "subtitle": ParagraphStyle(
            name="SubTitle",
            parent=base["Normal"],
            fontName="Arial",
            fontSize=11,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#5D6673"),
            spaceAfter=6,
        ),
        "section": ParagraphStyle(
            name="SectionTitle",
            parent=base["Heading2"],
            fontName="Arial-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#174A7C"),
            spaceBefore=12,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            name="BodyText2",
            parent=base["BodyText"],
            fontName="Arial",
            fontSize=9.5,
            leading=13.2,
            alignment=TA_LEFT,
            spaceAfter=7,
        ),
        "small": ParagraphStyle(
            name="Small",
            parent=base["BodyText"],
            fontName="Arial",
            fontSize=8.2,
            leading=10.8,
            alignment=TA_LEFT,
        ),
        "url": ParagraphStyle(
            name="Url",
            parent=base["BodyText"],
            fontName="Arial",
            fontSize=7.2,
            leading=9.2,
            textColor=colors.HexColor("#174A7C"),
            wordWrap="CJK",
        ),
        "table_head": ParagraphStyle(
            name="TableHead",
            parent=base["BodyText"],
            fontName="Arial-Bold",
            fontSize=8.5,
            leading=10.5,
            textColor=colors.white,
        ),
        "table_cell": ParagraphStyle(
            name="TableCell",
            parent=base["BodyText"],
            fontName="Arial",
            fontSize=8.2,
            leading=10.4,
        ),
        "table_cell_bold": ParagraphStyle(
            name="TableCellBold",
            parent=base["BodyText"],
            fontName="Arial-Bold",
            fontSize=8.2,
            leading=10.4,
        ),
    }


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text).replace("`", ""), style)


def build_table(
    rows: list[list[str]],
    styles: dict[str, ParagraphStyle],
    col_widths: list[float],
    header_color: colors.Color,
) -> Table:
    data = [
        [Paragraph(escape(cell), styles["table_head"]) for cell in rows[0]]
    ]
    for row in rows[1:]:
        data.append(
            [
                Paragraph(escape(row[0]), styles["table_cell_bold"]),
                *[Paragraph(escape(cell), styles["table_cell"]) for cell in row[1:]],
            ]
        )
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), header_color),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D5DBE3")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#F2F5F8")],
                ),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def add_header_footer(canvas, doc) -> None:
    primary = colors.HexColor("#174A7C")
    muted = colors.HexColor("#5D6673")
    width, height = A4
    canvas.saveState()
    canvas.setFillColor(primary)
    canvas.rect(0, height - 1.05 * cm, width, 1.05 * cm, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont("Arial-Bold", 8.5)
    canvas.drawString(doc.leftMargin, height - 0.65 * cm, "Entrega Tech Challenge - Fase 3")
    canvas.setFont("Arial", 8.5)
    canvas.drawRightString(width - doc.rightMargin, height - 0.65 * cm, f"Página {doc.page}")
    canvas.setFillColor(muted)
    canvas.setFont("Arial", 7.5)
    canvas.drawString(
        doc.leftMargin,
        0.8 * cm,
        "Uso acadêmico - dados sintéticos - não validado para uso assistencial real",
    )
    canvas.restoreState()


def main() -> None:
    markdown = SOURCE.read_text(encoding="utf-8")
    styles = make_styles()
    primary = colors.HexColor("#174A7C")
    accent = colors.HexColor("#0B6E4F")

    title = markdown.splitlines()[0].lstrip("# ").strip()
    meta = parse_paragraphs(markdown.split("## Integrantes", maxsplit=1)[0])
    project = meta[0].removeprefix("Projeto: ").strip()
    course = meta[1].removeprefix("Curso: ").strip()
    turma = meta[2].removeprefix("Turma: ").strip()
    version = meta[3].removeprefix("Data da versão: ").strip()

    integrantes = [
        line.strip("- ").strip()
        for line in section(markdown, "Integrantes").splitlines()
        if line.strip().startswith("- ")
    ]
    links = parse_table(section(markdown, "Links principais"))
    entregaveis = parse_table(section(markdown, "Atendimento dos entregáveis solicitados"))
    resumo = parse_paragraphs(section(markdown, "Resumo técnico do trabalho"))
    resultados = parse_table(section(markdown, "Resultados principais"))
    observacao = parse_paragraphs(section(markdown, "Observação final"))

    story = []
    story.append(Spacer(1, 1.0 * cm))
    story.append(Paragraph(escape(title), styles["cover"]))
    story.append(Paragraph(escape(project), styles["subtitle"]))
    story.append(
        Paragraph(
            escape(f"{course} | Turma {turma} | Data da versão: {version}"),
            styles["subtitle"],
        )
    )
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Integrantes", styles["section"]))
    for person in integrantes:
        story.append(p(f"- {person}", styles["body"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("Links principais", styles["section"]))
    for artefato, url, nota in links[1:]:
        if url == "PENDENTE":
            content = (
                f"<b>{escape(artefato)}</b><br/>"
                '<font color="#9A3412"><b>PENDENTE</b></font><br/>'
                f"{escape(nota)}"
            )
            link_para = Paragraph(content, styles["small"])
        else:
            safe_url = escape(url)
            content = (
                f"<b>{escape(artefato)}</b><br/>"
                f'<a href="{safe_url}" color="#174A7C">{safe_url}</a><br/>'
                f"{escape(nota)}"
            )
            link_para = Paragraph(content, styles["url"])
        story.append(KeepTogether([link_para, Spacer(1, 0.16 * cm)]))

    story.append(PageBreak())
    story.append(Paragraph("Atendimento dos entregáveis solicitados", styles["section"]))
    story.append(build_table(entregaveis, styles, [5.0 * cm, 11.2 * cm], primary))
    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph("Resumo técnico do trabalho", styles["section"]))
    for paragraph in resumo:
        story.append(p(paragraph, styles["body"]))

    story.append(PageBreak())
    story.append(Paragraph("Resultados principais", styles["section"]))
    story.append(build_table(resultados, styles, [11.7 * cm, 4.5 * cm], accent))
    story.append(Spacer(1, 0.4 * cm))
    for paragraph in parse_paragraphs(section(markdown, "Resultados principais")):
        story.append(p(paragraph, styles["body"]))
    story.append(Paragraph("Observação final", styles["section"]))
    for paragraph in observacao:
        story.append(p(paragraph, styles["body"]))

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(TARGET),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.4 * cm,
        title=title,
        author="Equipe 8IADT",
        subject="Índice de entrega da Fase 3",
    )
    doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)
    print(TARGET)


if __name__ == "__main__":
    main()
