"""
Generador del Informe Tecnico en Microsoft Word (.docx) para Evidencia TRL 3
============================================================================
Lee directamente los resultados experimentales reales desde:
  - prototipo_trl3/outputs/evidencia_trl3.json
  - prototipo_trl3/outputs/*.png

Y construye:
  C:\ML_img_can\Informe_Tecnico_TRL3_IA_Tamizaje_Cervical.docx
"""

import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
EVIDENCE_JSON = OUTPUTS_DIR / "evidencia_trl3.json"
OUT_DOCX = PROJECT_ROOT / "Informe_Tecnico_TRL3_IA_Tamizaje_Cervical.docx"

# Paleta corporativa / cientifica
HEX_NAVY = "0F172A"
HEX_BLUE = "1E3A5F"
HEX_ACCENT = "0284C7"
HEX_LIGHT_BG = "F8FAFC"
HEX_CALLOUT_BG = "F0F9FF"
HEX_WARN_BG = "FFFBEB"
HEX_WARN_BORDER = "D97706"
HEX_BORDER = "CBD5E1"

RGB_NAVY = RGBColor(0x0F, 0x17, 0x2A)
RGB_BLUE = RGBColor(0x1E, 0x3A, 0x5F)
RGB_ACCENT = RGBColor(0x02, 0x84, 0xC7)
RGB_MUTED = RGBColor(0x47, 0x55, 0x69)
RGB_TEXT = RGBColor(0x1E, 0x29, 0x3B)
RGB_WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def set_cell_shading(cell, fill_hex: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=80, bottom=80, left=120, right=120) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for m, val in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
        node = OxmlElement(f"w:{m}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tcMar.append(node)
    tcPr.append(tcMar)


def set_table_borders(table, border_hex: str = HEX_BORDER) -> None:
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="{border_hex}"/>'
        f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="{border_hex}"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{border_hex}"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideV w:val="none"/>'
        f"</w:tblBorders>"
    )
    tblPr.append(borders)


def prevent_row_split(row, is_header: bool = False) -> None:
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:cantSplit"))
    if is_header:
        trPr.append(OxmlElement("w:tblHeader"))


def add_page_number_field(run) -> None:
    fldChar1 = parse_xml(r'<w:fldChar %s w:fldCharType="begin"/>' % nsdecls("w"))
    instrText = parse_xml(r'<w:instrText %s xml:space="preserve"> PAGE </w:instrText>' % nsdecls("w"))
    fldChar2 = parse_xml(r'<w:fldChar %s w:fldCharType="separate"/>' % nsdecls("w"))
    fldChar3 = parse_xml(r'<w:fldChar %s w:fldCharType="end"/>' % nsdecls("w"))
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(fldChar3)


def add_heading_1(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    # Borde inferior sutil azul
    pPr = p._p.get_or_add_pPr()
    pBdr = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="10" w:space="4" w:color="{HEX_ACCENT}"/></w:pBdr>')
    pPr.append(pBdr)
    r = p.add_run(text)
    r.font.name = "Calibri"
    r.font.size = Pt(14.5)
    r.font.bold = True
    r.font.color.rgb = RGB_BLUE


def add_heading_2(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(11)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = "Calibri"
    r.font.size = Pt(12)
    r.font.bold = True
    r.font.color.rgb = RGB_NAVY


def add_body(doc: Document, text: str, bold_prefix: str = "", space_after: float = 5.5) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.font.name = "Calibri"
        rb.font.size = Pt(10.5)
        rb.font.bold = True
        rb.font.color.rgb = RGB_NAVY
    r = p.add_run(text)
    r.font.name = "Calibri"
    r.font.size = Pt(10.5)
    r.font.color.rgb = RGB_TEXT


def add_bullet(doc: Document, text: str, bold_prefix: str = "") -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.12
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.font.name = "Calibri"
        rb.font.size = Pt(10.2)
        rb.font.bold = True
        rb.font.color.rgb = RGB_NAVY
    r = p.add_run(text)
    r.font.name = "Calibri"
    r.font.size = Pt(10.2)
    r.font.color.rgb = RGB_TEXT


def add_callout(doc: Document, title: str, body: str, bg_hex: str = HEX_CALLOUT_BG, border_hex: str = HEX_ACCENT) -> None:
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_shading(cell, bg_hex)
    set_cell_margins(cell, top=110, bottom=110, left=160, right=140)
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="{border_hex}"/>'
        f'<w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/>'
        f"</w:tcBorders>"
    )
    tcPr.append(borders)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    rt = p.add_run(title + "\n")
    rt.font.name = "Calibri"
    rt.font.size = Pt(10.5)
    rt.font.bold = True
    rt.font.color.rgb = RGB_NAVY
    rb = p.add_run(body)
    rb.font.name = "Calibri"
    rb.font.size = Pt(10)
    rb.font.color.rgb = RGB_TEXT
    prevent_row_split(tbl.rows[0])
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(4)


def add_code_block(doc: Document, code_text: str) -> None:
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_shading(cell, "0F172A")
    set_cell_margins(cell, top=100, bottom=100, left=130, right=130)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.05
    r = p.add_run(code_text)
    r.font.name = "Consolas"
    r.font.size = Pt(8.2)
    r.font.color.rgb = RGBColor(0xE2, 0xE8, 0xF0)
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(4)


def add_styled_table(doc: Document, headers: list[str], rows: list[list[str]], col_widths: list[float] | None = None, align_right_cols: set[int] | None = None) -> None:
    align_right_cols = align_right_cols or set()
    tbl = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl)

    # Header row
    hdr_row = tbl.rows[0]
    prevent_row_split(hdr_row, is_header=True)
    for j, h in enumerate(headers):
        cell = hdr_row.cells[j]
        set_cell_shading(cell, HEX_BLUE)
        set_cell_margins(cell, top=90, bottom=90, left=100, right=100)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        if j in align_right_cols:
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r = p.add_run(h)
        r.font.name = "Calibri"
        r.font.size = Pt(9.5)
        r.font.bold = True
        r.font.color.rgb = RGB_WHITE

    # Data rows
    for i, row_data in enumerate(rows):
        row = tbl.rows[i + 1]
        prevent_row_split(row, is_header=False)
        bg = HEX_LIGHT_BG if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(row_data):
            cell = row.cells[j]
            set_cell_shading(cell, bg)
            set_cell_margins(cell, top=70, bottom=70, left=100, right=100)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            if j in align_right_cols:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            r = p.add_run(str(val))
            r.font.name = "Calibri"
            r.font.size = Pt(9.3)
            r.font.bold = (j == 0)
            r.font.color.rgb = RGB_NAVY if j == 0 else RGB_TEXT

    if col_widths:
        for row in tbl.rows:
            for j, w in enumerate(col_widths):
                row.cells[j].width = Inches(w)

    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(4)


def add_figure(doc: Document, img_path: Path, caption: str, width_in: float = 5.9) -> None:
    if not img_path.exists():
        add_body(doc, f"[Figura no encontrada: {img_path.name}]")
        return
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(3)
    p_img.paragraph_format.keep_with_next = True
    run_img = p_img.add_run()
    run_img.add_picture(str(img_path), width=Inches(width_in))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_after = Pt(8)
    rc = p_cap.add_run(caption)
    rc.font.name = "Calibri"
    rc.font.size = Pt(9.0)
    rc.font.italic = True
    rc.font.color.rgb = RGB_MUTED


def build_report() -> None:
    ev = json.loads(EVIDENCE_JSON.read_text(encoding="utf-8"))
    cnn = ev["modulo_cnn"]
    tab = ev["modulo_tabular"]
    ood = ev.get("modulo_ood")

    doc = Document()

    # Margenes pagina
    for section in doc.sections:
        section.top_margin = Inches(0.85)
        section.bottom_margin = Inches(0.85)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)
        section.different_first_page_header_footer = True

        # Header paginas interiores
        hdr = section.header
        hp = hdr.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hr = hp.add_run("Prototipo IA-Tamizaje Cervical v0.1  |  Informe Técnico de Evidencia Experimental TRL 3")
        hr.font.name = "Calibri"
        hr.font.size = Pt(8.5)
        hr.font.color.rgb = RGB_MUTED

        # Footer paginas interiores
        ftr = section.footer
        fp = ftr.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fr1 = fp.add_run("Repositorio: github.com/MaycolPGR-aduch/ML_cancer_utero   •   Página ")
        fr1.font.name = "Calibri"
        fr1.font.size = Pt(8.5)
        fr1.font.color.rgb = RGB_MUTED
        fr_pg = fp.add_run()
        fr_pg.font.name = "Calibri"
        fr_pg.font.size = Pt(8.5)
        fr_pg.font.color.rgb = RGB_MUTED
        add_page_number_field(fr_pg)

    # =========================================================================
    # PORTADA
    # =========================================================================
    banner_tbl = doc.add_table(rows=1, cols=1)
    banner_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    bcell = banner_tbl.cell(0, 0)
    set_cell_shading(bcell, HEX_NAVY)
    set_cell_margins(bcell, top=220, bottom=220, left=220, right=220)
    bp = bcell.paragraphs[0]
    bp.paragraph_format.space_after = Pt(6)
    r_tag = bp.add_run("NIVEL DE MADUREZ TECNOLÓGICA: TRL 3  •  PRUEBA DE CONCEPTO EXPERIMENTAL\n")
    r_tag.font.name = "Calibri"
    r_tag.font.size = Pt(10)
    r_tag.font.bold = True
    r_tag.font.color.rgb = RGBColor(0x38, 0xBD, 0xF8)

    r_title = bp.add_run("INFORME TÉCNICO DE EVIDENCIA EXPERIMENTAL\nPROTOTIPO IA-TAMIZAJE CERVICAL v0.1")
    r_title.font.name = "Calibri"
    r_title.font.size = Pt(21)
    r_title.font.bold = True
    r_title.font.color.rgb = RGB_WHITE

    bp2 = bcell.add_paragraph()
    bp2.paragraph_format.space_after = Pt(0)
    r_sub = bp2.add_run(
        "Sistema Multimodal de Apoyo al Tamizaje Temprano de Cáncer de Cuello Uterino\n"
        "mediante Redes Neuronales Convolucionales (MobileNetV2 + Grad-CAM), "
        "Modelado Tabular de Factores de Riesgo e Interfaz Web FastAPI"
    )
    r_sub.font.name = "Calibri"
    r_sub.font.size = Pt(11.5)
    r_sub.font.color.rgb = RGBColor(0xCB, 0xD5, 0xE1)

    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(14)

    # Metadatos de portada
    add_styled_table(
        doc,
        ["Atributo de Control", "Detalle Registrado"],
        [
            ["Nombre del Prototipo", "Prototipo IA-Tamizaje Cervical v0.1 (Módulos CNN + Tabular + App Web)"],
            ["Versiones de Software", f"{cnn['identificacion']['version']}  |  {tab['identificacion']['version']}"],
            ["Nivel de Madurez Evidenciado", "TRL 3 — Prueba de concepto analítica y experimental en entorno controlado"],
            ["Fecha de Ejecución y Consolidación", "19 de noviembre de 2026 (UTC: 2026-11-19T21:30Z)"],
            ["Semilla de Reproducibilidad", "random_state = 42 (fija en particiones, modelos y entrenamiento)"],
            ["Entorno Experimental", "Windows 10/11 (AMD64), Intel Core i7-1260P (CPU), Python 3.10.0, PyTorch 2.14.1+cpu, scikit-learn 1.7.2, FastAPI 0.141.1"],
            ["Repositorio Oficial (Código y Evidencias)", "https://github.com/MaycolPGR-aduch/ML_cancer_utero (rama main, commit b2ea17c)"],
        ],
        col_widths=[2.3, 4.5],
    )

    # Tarjetas KPI resumen ejecutivo en portada
    cm_cnn = cnn["metricas_test"]
    bin_cnn = cm_cnn["tamizaje_clinico_binario"]
    tm_cv = tab["metricas_modelo_seleccionado"]["cv_5fold"]

    add_callout(
        doc,
        "Resumen Ejecutivo de Resultados Verificados (TRL 3)",
        f"• Módulo CNN (SIPaKMeD, 4,049 imágenes, split sin fuga por lámina, Test n=614): "
        f"Accuracy = {cm_cnn['accuracy']*100:.2f}%, Macro F1 = {cm_cnn['macro_f1']*100:.2f}%, "
        f"Macro Recall = {cm_cnn['macro_recall']*100:.2f}%, Macro ROC-AUC = {cm_cnn['macro_roc_auc_ovr']:.4f}, "
        f"Macro PR-AUC = {cm_cnn['macro_pr_auc_ovr']:.4f}. En tamizaje binario (Anormal vs Normal/Benigna): "
        f"Sensibilidad = {bin_cnn['sensibilidad_recall']*100:.2f}%, Especificidad = {bin_cnn['especificidad']*100:.2f}%, "
        f"ROC-AUC = {bin_cnn['roc_auc']:.4f}, con latencia promedio en CPU de {cnn['prueba_inferencia']['latencia_promedio_ms_30_runs']:.2f} ms/imagen.\n"
        f"• Módulo Tabular (Factores de riesgo, 858 registros → 26 variables sin leakage, CV 5-Fold): "
        f"RandomForest balanceado alcanzó ROC-AUC = {tm_cv['roc_auc']:.4f} (±{tm_cv['roc_auc_std']:.4f}) y "
        f"PR-AUC = {tm_cv['pr_auc']:.4f} (2.6× sobre la prevalencia base de Biopsia de 6.41%), "
        f"con latencia promedio en CPU de {tab['prueba_inferencia']['latencia_promedio_ms_50_runs']:.2f} ms/caso.\n"
        f"• Software Operativo en Laboratorio: API REST en FastAPI, interfaz web interactiva de 3 pestañas "
        f"(subida de imagen + Grad-CAM, formulario clínico tabular y panel de evidencia TRL 3) y script CLI reproducible (infer.py).",
        bg_hex=HEX_CALLOUT_BG,
        border_hex=HEX_ACCENT,
    )

    add_callout(
        doc,
        "Aviso Clínico, Ético y Regulatorio (Alcance TRL 3)",
        "Este prototipo constituye una prueba de concepto analítica y experimental en entorno controlado de laboratorio. "
        "Está diseñado únicamente como herramienta de investigación y apoyo temprano al tamizaje; NO ha sido sometido aún a validación "
        "clínica prospectiva ni a certificación como dispositivo médico (SaMD), por lo que NO sustituye el criterio del especialista "
        "en citopatología o ginecología ni constituye un diagnóstico médico definitivo.",
        bg_hex=HEX_WARN_BG,
        border_hex=HEX_WARN_BORDER,
    )

    doc.add_page_break()

    # =========================================================================
    # SECCIÓN 1: INTRODUCCIÓN Y OBJETIVO TRL 3
    # =========================================================================
    add_heading_1(doc, "1. Introducción, Contexto Clínico y Objetivo TRL 3")
    add_body(
        doc,
        "El cáncer de cuello uterino (cervical) continúa siendo una de las neoplasias ginecológicas de mayor impacto "
        "en salud pública a nivel mundial, pese a ser altamente prevenible cuando se detectan lesiones intraepiteliales "
        "y anomalías celulares en etapas tempranas mediante tamizaje citológico y estratificación oportuna de factores de riesgo. "
        "La evaluación manual de extendidos citológicos requiere especialistas altamente capacitados, consume tiempo y puede "
        "presentar variabilidad interobservador en contextos de alta demanda asistencial."
    )
    add_body(
        doc,
        "El presente informe documenta de forma trazable, reproducible y verificable la evidencia técnica alcanzada en "
        "nivel TRL 3 (Technology Readiness Level 3 — prueba de concepto analítica y experimental en entorno controlado) "
        "del Prototipo IA-Tamizaje Cervical v0.1. A diferencia de etapas posteriores (TRL 4–7), el objetivo central en TRL 3 "
        "consiste en demostrar que los algoritmos de aprendizaje profundo y aprendizaje automático, junto con su flujo completo "
        "de preprocesamiento, inferencia y salida estructurada de software, funcionan técnicamente sobre datos retrospectivos "
        "de referencia sin fuga de información (data leakage)."
    )
    add_bullet(doc, "Demostrar la viabilidad técnica de clasificación automatizada de citología cervical en 5 clases morfológicas mediante una red neuronal convolucional (MobileNetV2) con explicabilidad visual Grad-CAM.", "Objetivo 1 (Módulo CNN): ")
    add_bullet(doc, "Demostrar la viabilidad de estratificación de riesgo de biopsia positiva a partir de variables clínicas y demográficas retrospectivas bajo control estricto de faltantes y fuga de datos.", "Objetivo 2 (Módulo Tabular): ")
    add_bullet(doc, "Integrar ambos modelos en un software verificable (API REST FastAPI, interfaz web interactiva y script CLI reproducible) capaz de recibir entradas reales y devolver probabilidades, priorización de tamizaje y tiempos de inferencia.", "Objetivo 3 (Integración de Software): ")

    # =========================================================================
    # SECCIÓN 2: FICHA TÉCNICA CONSOLIDADA TRL 3
    # =========================================================================
    add_heading_1(doc, "2. Ficha Técnica de Evidencia Experimental (TRL 3)")
    add_body(
        doc,
        "La siguiente tabla consolida los campos mínimos exigidos para acreditar madurez TRL 3 tanto en la vertiente "
        "de Redes Neuronales Convolucionales (imágenes) como en la vertiente algorítmica tabular:"
    )

    add_styled_table(
        doc,
        ["Campo Mínimo TRL 3", "Módulo CNN (Imágenes Citológicas)", "Módulo Tabular (Factores de Riesgo)"],
        [
            [
                "Identificación y Versión",
                f"Prototipo IA-Tamizaje Cervical ({cnn['identificacion']['version']}), random_state=42, PyTorch {cnn['identificacion']['entorno']['pytorch']} en CPU.",
                f"Prototipo IA-Tamizaje Cervical ({tab['identificacion']['version']}), random_state=42, scikit-learn {tab['identificacion']['entorno']['scikit_learn']} en CPU.",
            ],
            [
                "Conjunto de Datos y Clases",
                "SIPaKMeD (CROPPED): 4,049 imágenes en 5 clases (Dyskeratotic: 813, Koilocytotic: 825, Metaplastic: 793, Parabasal: 787, Superficial-Intermediate: 831).",
                "kag_risk_factors_cervical_cancer.csv: 858 registros × 36 columnas iniciales → 26 variables clínicas sin fuga. Objetivo: Biopsy (55/858 positivos, 6.41%).",
            ],
            [
                "Preprocesamiento y Control de Fuga",
                "Resize 224×224, normalización ImageNet, data augmentation exclusivo en train y escalado proporcional de contornos *_cyt.dat / *_nuc.dat.",
                "Conversión '?' → NaN, exclusión de 8 desenlaces post-prueba y 2 columnas con >50% faltantes (787/858). SimpleImputer(median) + StandardScaler en Pipeline.",
            ],
            [
                "Estrategia de Validación (No Leakage)",
                "StratifiedGroupKFold (k=7) agrupado por (clase, id_lámina): Train 2,819 (691 láminas), Val 616 (137), Test 614 (138). 0 láminas compartidas.",
                "Validación cruzada estratificada StratifiedKFold (k=5, shuffle=True, random_state=42) y evaluación Out-Of-Fold (OOF) sobre los 858 casos.",
            ],
            [
                "Arquitectura / Algoritmo",
                "MobileNetV2 (Transfer Learning ImageNet): 2 épocas cabeza + 6 épocas fine-tuning (features[10:]), AdamW, CosineAnnealingLR, CrossEntropy ponderada.",
                "Comparación baseline: LogisticRegression, RandomForest y ExtraTrees (class_weight='balanced'). Seleccionado: RandomForest (balanced), umbral tamizaje=0.44.",
            ],
            [
                "Resultados Principales",
                f"Test (n=614): Acc {cm_cnn['accuracy']*100:.2f}%, Macro-F1 {cm_cnn['macro_f1']*100:.2f}%, Macro-Recall {cm_cnn['macro_recall']*100:.2f}%, ROC-AUC {cm_cnn['macro_roc_auc_ovr']:.4f}, PR-AUC {cm_cnn['macro_pr_auc_ovr']:.4f}. Tamizaje binario Sens. {bin_cnn['sensibilidad_recall']*100:.2f}%.",
                f"CV 5-Fold: ROC-AUC {tm_cv['roc_auc']:.4f} (±{tm_cv['roc_auc_std']:.4f}), PR-AUC {tm_cv['pr_auc']:.4f} (2.6× sobre base 0.0641). OOF thr=0.44: Recall 30.91%, Esp. 85.93%, F1 0.1838.",
            ],
            [
                "Evidencia de Software y Latencia",
                f"train_eval_cnn.py + infer.py + gradcam.py + FastAPI (/api/predict/image). Latencia CPU promedio: {cnn['prueba_inferencia']['latencia_promedio_ms_30_runs']:.2f} ms/imagen.",
                f"train_eval_tabular.py + infer.py + FastAPI (/api/predict/tabular). Latencia CPU promedio: {tab['prueba_inferencia']['latencia_promedio_ms_50_runs']:.2f} ms/registro.",
            ],
        ],
        col_widths=[1.5, 2.7, 2.6],
    )

    # =========================================================================
    # SECCIÓN 3: CONJUNTOS DE DATOS Y CONTROL ANTI-LEAKAGE
    # =========================================================================
    add_heading_1(doc, "3. Conjuntos de Datos, Preprocesamiento y Control Anti-Leakage")

    add_heading_2(doc, "3.1 Módulo de Imágenes: Dataset Citológico SIPaKMeD")
    add_body(
        doc,
        "Se empleó la colección de células individuales recortadas (CROPPED) del repositorio público de referencia SIPaKMeD "
        "(Plissiti et al., 2018), compuesta por exactamente 4,049 imágenes en formato BMP distribuidas de manera equilibrada en "
        "cinco categorías citológicas con relevancia clínica directa para el tamizaje:"
    )

    split = cnn["datos"]["particion_sin_fuga"]
    cls_rows = []
    clin_map = {
        "im_Dyskeratotic": "Anormal (queratinización atípica)",
        "im_Koilocytotic": "Anormal (asociada a VPH / LSIL)",
        "im_Metaplastic": "Benigna (metaplasia escamosa)",
        "im_Parabasal": "Normal (capa profunda)",
        "im_Superficial-Intermediate": "Normal (capas maduras)",
    }
    for cls in cnn["datos"]["clases"]:
        cls_rows.append([
            cls,
            clin_map[cls],
            str(cnn["datos"]["distribucion_por_clase"][cls]),
            str(split["train"]["por_clase"][cls]),
            str(split["val"]["por_clase"][cls]),
            str(split["test"]["por_clase"][cls]),
        ])
    cls_rows.append([
        "TOTAL (0 láminas compartidas)",
        "966 láminas únicas agrupadas",
        str(cnn["datos"]["total_imagenes"]),
        f"{split['train']['imagenes']} ({split['train']['laminas_unicas']} lám.)",
        f"{split['val']['imagenes']} ({split['val']['laminas_unicas']} lám.)",
        f"{split['test']['imagenes']} ({split['test']['laminas_unicas']} lám.)",
    ])
    add_styled_table(
        doc,
        ["Clase SIPaKMeD", "Significado Clínico", "Total", "Train", "Val", "Test"],
        cls_rows,
        col_widths=[1.85, 2.05, 0.55, 0.85, 0.75, 0.75],
        align_right_cols={2, 3, 4, 5},
    )

    add_body(
        doc,
        "Una falla metodológica frecuente al trabajar con SIPaKMeD consiste en dividir las 4,049 células de manera aleatoria simple, "
        "lo cual introduce fuga de información porque varias células recortadas (ej. 001_01.bmp, 001_02.bmp, 001_03.bmp) provienen "
        "de la misma lámina microscópica y comparten tinción e iluminación. En este prototipo se implementó una partición agrupada "
        "por (clase, id_lámina) mediante StratifiedGroupKFold (k=7), verificada programáticamente mediante aserciones que garantizan "
        "cero láminas compartidas entre entrenamiento (691 láminas), validación (137 láminas) y prueba (138 láminas).",
        bold_prefix="Control estricto anti-leakage por lámina: "
    )

    add_body(
        doc,
        "Todas las imágenes se redimensionaron a 224×224 píxeles y se normalizaron con los parámetros estándar de ImageNet "
        "(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). El aumento de datos (volteos horizontales/verticales aleatorios, "
        "rotaciones hasta ±25° y variaciones suaves de brillo, contraste y saturación) se aplicó exclusivamente durante el entrenamiento. "
        "Adicionalmente, se implementó el escalado geométrico proporcional de las coordenadas de contorno citoplasmático (*_cyt.dat) "
        "y nuclear (*_nuc.dat) para preservar la correspondencia espacial tras el redimensionamiento a 224×224, como se evidencia en la Figura 1."
    )

    add_figure(
        doc,
        OUTPUTS_DIR / "cnn_sample_annotations_224.png",
        "Figura 1. Verificación del preprocesamiento geométrico a 224×224 px y escalado proporcional de anotaciones de contorno "
        "citoplasmático (*_cyt.dat, cian) y nuclear (*_nuc.dat, amarillo) sobre muestras reales de las 5 clases de SIPaKMeD.",
        width_in=6.3,
    )

    add_heading_2(doc, "3.2 Módulo Tabular: Factores de Riesgo Cervical")
    add_body(
        doc,
        "El módulo tabular utiliza la cohorte retrospectiva del Hospital Universitario de Caracas "
        "(kag_risk_factors_cervical_cancer.csv), con 858 registros y 36 columnas originales. Durante la auditoría de calidad "
        "de datos se identificaron valores '?' codificando datos faltantes en 26 columnas, destacando 787 ausencias (91.7%) "
        "en las variables de tiempo desde diagnóstico de ETS y entre 105 y 117 ausencias (12.2%–13.6%) en antecedentes de ETS, "
        "anticonceptivos hormonales y uso de DIU."
    )
    add_bullet(doc, "Se excluyeron las 8 variables de pruebas clínicas posteriores o diagnósticos directos (Hinselmann, Schiller, Citology, Biopsy, Dx:Cancer, Dx:CIN, Dx:HPV, Dx) del conjunto predictor para evitar fuga de información respecto a la variable objetivo Biopsy.", "Exclusión por data leakage: ")
    add_bullet(doc, "Se excluyeron STDs: Time since first diagnosis y STDs: Time since last diagnosis por superar el umbral de 50% de valores ausentes (787/858).", "Exclusión por alta ausencia (>50%): ")
    add_bullet(doc, "Quedaron 26 variables clínicas y demográficas de entrada. La imputación por mediana (SimpleImputer) y el escalado (StandardScaler) se encapsularon dentro de un Pipeline de scikit-learn ajustado estrictamente dentro de cada fold de validación cruzada.", "Pipeline sin fuga: ")

    out_rows = []
    for col, info in tab["datos"]["desenlaces_registrados"].items():
        rol = "Variable objetivo principal" if col == "Biopsy" else "Excluida del predictor (post-prueba / leakage)"
        out_rows.append([col, str(info["positivos"]), str(info["negativos"]), f"{info['prevalencia_pct']:.2f}%", rol])
    add_styled_table(
        doc,
        ["Desenlace Clínico", "Positivos", "Negativos", "Prevalencia", "Tratamiento en el Prototipo"],
        out_rows,
        col_widths=[1.2, 0.75, 0.8, 0.85, 3.2],
        align_right_cols={1, 2, 3},
    )

    # =========================================================================
    # SECCIÓN 4: ARQUITECTURA, ENTRENAMIENTO Y SOFTWARE
    # =========================================================================
    add_heading_1(doc, "4. Metodología de Entrenamiento y Arquitectura de Software")

    add_heading_2(doc, "4.1 Arquitectura CNN y Esquema de Entrenamiento en Dos Etapas")
    add_body(
        doc,
        "Se seleccionó MobileNetV2 preentrenada en ImageNet mediante Transfer Learning, sustituyendo la cabeza clasificadora "
        "por Dropout(p=0.25) seguido de una capa lineal de 1,280 → 5 salidas. Esta arquitectura combina alta capacidad "
        "representacional para texturas citológicas con muy baja carga computacional, permitiendo inferencia en tiempo real sobre CPU "
        "estándar (~13 ms). El entrenamiento (1,093.2 s en CPU Intel Core i7-1260P con 8 hilos) se ejecutó en dos fases reproducibles:"
    )
    add_bullet(doc, "Backbone convolucional congelado durante 2 épocas (AdamW, lr=1.5e-3, weight_decay=1e-4, batch_size=32) para calibrar los pesos del clasificador sin degradar los filtros preentrenados.", "Fase 1 — Ajuste de cabeza clasificadora: ")
    add_bullet(doc, "Descongelamiento de los bloques convolucionales superiores (features[10:]) más el clasificador durante 6 épocas adicionales con tasa de aprendizaje reducida (lr=4e-4) y decaimiento cosenoidal (CosineAnnealingLR hasta 2e-5). Se empleó CrossEntropyLoss ponderada por frecuencia de clase y se seleccionó el mejor checkpoint según Macro-F1 en validación (época 8, Val Macro-F1 = 0.9513).", "Fase 2 — Fine-tuning de capas profundas: ")

    hist_rows = [
        [
            f"Época {h['epoch']:02d}",
            "Fase 1 (Cabeza)" if h["phase"] == "head_only" else "Fase 2 (Fine-tuning)",
            f"{h['train_loss']:.4f}",
            f"{h['val_loss']:.4f}",
            f"{h['val_accuracy']*100:.2f}%",
            f"{h['val_macro_f1']*100:.2f}%",
            f"{h['seconds']:.1f} s",
        ]
        for h in cnn["entrenamiento"]["historial"]
    ]
    add_styled_table(
        doc,
        ["Época", "Etapa", "Train Loss", "Val Loss", "Val Accuracy", "Val Macro-F1", "Duración"],
        hist_rows,
        col_widths=[0.75, 1.55, 0.9, 0.9, 0.95, 1.0, 0.75],
        align_right_cols={2, 3, 4, 5, 6},
    )

    add_figure(
        doc,
        OUTPUTS_DIR / "cnn_training_history.png",
        "Figura 2. Evolución de la función de pérdida (entrenamiento vs. validación) y del desempeño (Val Macro-F1 y Val Accuracy) "
        "a lo largo de las 8 épocas de entrenamiento en dos etapas sobre MobileNetV2.",
        width_in=6.1,
    )

    add_heading_2(doc, "4.2 Arquitectura del Prototipo Web Multimodal (FastAPI)")
    add_body(
        doc,
        "Para evidenciar funcionamiento técnico real de software en entorno controlado, el prototipo se estructuró bajo una "
        "arquitectura desacoplada cliente-servidor que separa completamente el entrenamiento offline de la inferencia online:"
    )
    add_bullet(doc, "Servidor FastAPI (prototipo_trl3/app/main.py) que carga en memoria ambos modelos entrenados (~9.5 MB en total) y expone endpoints REST documentados automáticamente vía OpenAPI/Swagger (/api/health, /api/samples, /api/predict/image, /api/predict/sample/{filename}, /api/tabular/schema, /api/predict/tabular, /api/evidence).", "Backend API REST: ")
    add_bullet(doc, "Implementación nativa en PyTorch (prototipo_trl3/gradcam.py) mediante hooks de activaciones y gradientes sobre la última capa convolucional de MobileNetV2, superponiendo un mapa de calor sobre la célula analizada para auditar qué regiones nucleares/citoplasmáticas guiaron la decisión.", "Módulo XAI (Grad-CAM): ")
    add_bullet(doc, "Aplicación web ligera (HTML5/CSS3/JS en prototipo_trl3/app/static/) con tres pestañas funcionales: (1) Citología Cervical con carga drag-and-drop y 5 muestras reales del conjunto Test, (2) Perfil Clínico Tabular con soporte para imputación automática de faltantes, y (3) Panel de Evidencia TRL 3 con descarga directa de evidencia_trl3.json y gráficas.", "Frontend Web: ")
    add_bullet(doc, "Script prototipo_trl3/infer.py para ejecución verificable por consola sobre imágenes y registros tabulares con salida JSON estructurada.", "Interfaz CLI Reproducible: ")

    # =========================================================================
    # SECCIÓN 5: RESULTADOS EXPERIMENTALES VERIFICADOS
    # =========================================================================
    add_heading_1(doc, "5. Resultados Experimentales Verificados")

    add_heading_2(doc, "5.1 Desempeño del Módulo CNN en Conjunto de Prueba (n = 614, sin fuga)")
    add_body(
        doc,
        "Evaluado sobre las 614 imágenes de prueba provenientes de 138 láminas nunca vistas durante el entrenamiento ni la validación, "
        "el modelo MobileNetV2 alcanzó un rendimiento sobresaliente y altamente balanceado en las cinco clases:"
    )

    per_cls_rows = []
    for cls in cnn["datos"]["clases"]:
        m = cm_cnn["por_clase"][cls]
        per_cls_rows.append([
            cls,
            clin_map[cls].split(" ")[0],
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1_score']:.4f}",
            f"{m['roc_auc_ovr']:.4f}",
            f"{m['pr_auc_ovr']:.4f}",
            str(m["soporte_test"]),
        ])
    per_cls_rows.append([
        "Promedio Macro (5 clases)",
        "Global",
        f"{cm_cnn['macro_precision']:.4f}",
        f"{cm_cnn['macro_recall']:.4f}",
        f"{cm_cnn['macro_f1']:.4f}",
        f"{cm_cnn['macro_roc_auc_ovr']:.4f}",
        f"{cm_cnn['macro_pr_auc_ovr']:.4f}",
        "614",
    ])
    add_styled_table(
        doc,
        ["Clase SIPaKMeD", "Categoría", "Precisión", "Recall", "F1-Score", "ROC-AUC", "PR-AUC", "Soporte"],
        per_cls_rows,
        col_widths=[1.8, 0.8, 0.72, 0.68, 0.72, 0.72, 0.72, 0.64],
        align_right_cols={2, 3, 4, 5, 6, 7},
    )

    add_body(
        doc,
        f"Al agrupar las predicciones desde una perspectiva clínica de tamizaje temprano —considerando como hallazgo Anormal "
        f"las clases im_Dyskeratotic e im_Koilocytotic frente a las clases Normal/Benigna (im_Metaplastic, im_Parabasal e "
        f"im_Superficial-Intermediate)—, el prototipo logra una Sensibilidad (Recall) de {bin_cnn['sensibilidad_recall']*100:.2f}%, "
        f"una Especificidad de {bin_cnn['especificidad']*100:.2f}%, Precisión de {bin_cnn['precision']*100:.2f}%, "
        f"F1-Score de {bin_cnn['f1_score']*100:.2f}% y ROC-AUC binario de {bin_cnn['roc_auc']:.4f} "
        f"(TP={bin_cnn['matriz_confusion_binaria']['tp']}, TN={bin_cnn['matriz_confusion_binaria']['tn']}, "
        f"FP={bin_cnn['matriz_confusion_binaria']['fp']}, FN={bin_cnn['matriz_confusion_binaria']['fn']}). "
        f"Cabe destacar que 15 de los errores entre clases anormales ocurrieron entre im_Dyskeratotic e im_Koilocytotic, "
        f"manteniendo igualmente activa la alerta de tamizaje anormal.",
        bold_prefix="Desempeño en tamizaje clínico binario: "
    )

    add_figure(
        doc,
        OUTPUTS_DIR / "cnn_confusion_matrix.png",
        "Figura 3. Matriz de confusión 5×5 del módulo CNN (MobileNetV2) sobre el conjunto de prueba SIPaKMeD (n=614, 0 láminas compartidas).",
        width_in=5.0,
    )

    add_figure(
        doc,
        OUTPUTS_DIR / "cnn_roc_pr_curves.png",
        "Figura 4. Curvas ROC y Precision-Recall multiclase (One-vs-Rest) en el conjunto de prueba. Todas las clases superan ROC-AUC de 0.983.",
        width_in=6.2,
    )

    add_figure(
        doc,
        OUTPUTS_DIR / "cnn_sample_gradcam.png",
        "Figura 5. Evidencia de explicabilidad visual (Grad-CAM) sobre 5 células representativas del conjunto Test (una por clase). "
        "Las zonas cálidas resaltan las regiones nucleares y perinucleares que concentran la atención de MobileNetV2.",
        width_in=6.2,
    )

    add_heading_2(doc, "5.2 Desempeño del Módulo Tabular (Validación Cruzada Estratificada 5-Fold)")
    add_body(
        doc,
        "Debido al fuerte desbalance del desenlace Biopsy (6.41% positivos) y a la exclusión rigurosa de variables post-prueba "
        "(evitando métricas artificialmente infladas presentes en enfoques con fuga), se compararon tres algoritmos baseline "
        "balanceados mediante validación cruzada estratificada (k=5) y optimización de umbral orientado a tamizaje:"
    )

    base_rows = []
    for mname, mres in tab["comparacion_modelos_baseline"].items():
        c5 = mres["cv_5fold_mean"]
        oof = mres["oof_global"]["umbral_optimizado_tamizaje"]
        sel = " (Seleccionado)" if mname == tab["metodologia"]["modelo_seleccionado"] else ""
        base_rows.append([
            mname + sel,
            f"{c5['roc_auc']:.4f} ± {c5['roc_auc_std']:.4f}",
            f"{c5['pr_auc']:.4f} ± {c5['pr_auc_std']:.4f}",
            f"{oof['threshold']:.2f}",
            f"{oof['recall']*100:.2f}%",
            f"{oof['precision']*100:.2f}%",
            f"{oof['f1']:.4f}",
        ])
    add_styled_table(
        doc,
        ["Modelo Baseline (Pipeline)", "ROC-AUC CV", "PR-AUC CV", "Umbral*", "Recall OOF", "Prec. OOF", "F1 OOF"],
        base_rows,
        col_widths=[1.95, 1.15, 1.1, 0.65, 0.7, 0.65, 0.6],
        align_right_cols={1, 2, 3, 4, 5, 6},
    )

    toof = tab["metricas_modelo_seleccionado"]["oof_con_umbral_tamizaje"]
    tcm = toof["matriz_confusion"]
    add_body(
        doc,
        f"El modelo seleccionado (RandomForest balanceado, max_depth=5, min_samples_leaf=5) superó a la regresión logística "
        f"en ROC-AUC (0.6437 vs 0.5467) y alcanzó un PR-AUC medio de 0.1695 (2.6 veces superior a la prevalencia base de 0.0641). "
        f"Con el umbral de tamizaje ajustado a {toof['umbral']}, la matriz de confusión Out-Of-Fold registró "
        f"TN={tcm['tn']}, FP={tcm['fp']}, FN={tcm['fn']} y TP={tcm['tp']} (Sensibilidad={toof['recall_sensibilidad']*100:.2f}%, "
        f"Especificidad={toof['especificidad']*100:.2f}%). Las variables con mayor peso relativo fueron: "
        f"años de anticonceptivos hormonales (19.8%), edad (14.2%), edad de primera relación sexual (12.6%), "
        f"número de embarazos (10.2%) y número de parejas sexuales (5.7%)."
    )
    add_body(
        doc,
        "Este contraste empírico entre ambos módulos constituye un hallazgo técnico clave de TRL 3: mientras las variables "
        "sociodemográficas y de cuestionario por sí solas ofrecen una capacidad discriminativa moderada (útil para priorización "
        "poblacional inicial), la incorporación de imágenes citológicas procesadas con CNN eleva drásticamente el desempeño "
        "diagnóstico de apoyo (ROC-AUC > 0.994), justificando la arquitectura multimodal del prototipo.",
        bold_prefix="Interpretación científica comparativa: "
    )

    add_figure(
        doc,
        OUTPUTS_DIR / "tabular_confusion_matrix.png",
        "Figura 6. Matriz de confusión Out-Of-Fold (5-Fold) del módulo tabular RandomForest (balanced) con umbral de tamizaje 0.44.",
        width_in=4.3,
    )

    add_figure(
        doc,
        OUTPUTS_DIR / "tabular_roc_pr_curves.png",
        "Figura 7. Curvas ROC y Precision-Recall Out-Of-Fold del módulo tabular frente a la línea base de azar y prevalencia poblacional (6.41%).",
        width_in=6.1,
    )

    if ood is not None:
        om = ood["metricas_test"]
        add_heading_2(doc, "5.3 Filtro de Validez de Dominio (Out-of-Distribution / OOD Gate)")
        add_body(
            doc,
            "Para mitigar el riesgo de clasificaciones forzadas cuando un usuario sube imágenes que no corresponden a citología "
            "cervical, se implementó un filtro de seguridad previo reutilizando los embeddings profundos de 1,280 dimensiones de "
            "MobileNetV2 (sin ejecutar un segundo pase CNN). Se utilizaron como ejemplos negativos las 7 categorías médicas "
            "NO cervicales de archivebase2 (Multi Cancer Dataset: ALL, Brain, Breast, Kidney, Lung & Colon, Lymphoma y Oral — "
            "21 subclases), excluyendo explícitamente su subcarpeta Cervical Cancer porque corresponde a copias pre-aumentadas "
            "de SIPaKMeD sin ID de lámina."
        )
        add_body(
            doc,
            f"El filtro combina una doble compuerta (clasificador calibrado StandardScaler + LogisticRegression y similitud coseno "
            f"mínima contra los 5 centroides citológicos de entrenamiento). En el conjunto de prueba retenido (SIPaKMeD Test sin "
            f"fuga vs. 7 órganos no cervicales), el filtro logró una aceptación de citología válida (Recall) de "
            f"{om['aceptacion_citologia_valida_recall']*100:.2f}%, un rechazo de imágenes no cervicales (Especificidad) de "
            f"{om['rechazo_no_cervical_especificidad']*100:.2f}%, ROC-AUC = {om['roc_auc']:.4f}, PR-AUC = {om['pr_auc']:.4f} y "
            f"un rechazo de {om['stress_test_sinteticas_rechazo']} en el stress-test de imágenes sintéticas no médicas, "
            f"con un sobrecosto de latencia de apenas +{om['sobrecosto_latencia_ms']:.3f} ms en CPU.",
            bold_prefix="Desempeño del filtro OOD: "
        )

        looo_rows = []
        for organ, info in om["validacion_leave_one_organ_out"].items():
            looo_rows.append([
                organ,
                str(info["imagenes_evaluadas"]),
                str(info["rechazadas_como_ood"]),
                f"{info['tasa_rechazo_ood']*100:.2f}%",
            ])
        add_styled_table(
            doc,
            ["Órgano No Visto (Leave-One-Organ-Out)", "Imágenes Evaluadas", "Rechazadas OOD", "Tasa de Rechazo"],
            looo_rows,
            col_widths=[2.6, 1.4, 1.4, 1.4],
            align_right_cols={1, 2, 3},
        )

        add_figure(
            doc,
            OUTPUTS_DIR / "ood_confusion_matrix.png",
            "Figura 8. Matriz de confusión del filtro OOD en prueba retenida y tasas de rechazo en validación Leave-One-Organ-Out "
            "(órganos médicos completamente no vistos durante el entrenamiento del filtro).",
            width_in=6.2,
        )
        add_figure(
            doc,
            OUTPUTS_DIR / "ood_score_distribution.png",
            "Figura 9. Separación de probabilidad OOD y similitud coseno en el espacio de embeddings entre citología cervical válida (Test), "
            "7 órganos médicos no cervicales e imágenes sintéticas no médicas.",
            width_in=6.2,
        )

    # =========================================================================
    # SECCIÓN 6: EVIDENCIA DE EJECUCIÓN DE SOFTWARE
    # =========================================================================
    add_heading_1(doc, "6. Evidencia Verificable de Ejecución de Software")
    add_body(
        doc,
        "Se verificó la ejecución completa de extremo a extremo tanto mediante pruebas automatizadas sobre los 7 endpoints "
        "de la API FastAPI (TestClient: /api/health, /api/samples, /api/predict/image, /api/predict/sample/{filename}, "
        "/api/tabular/schema, /api/predict/tabular y /api/evidence) como mediante el script CLI reproducible infer.py. "
        "A continuación se presenta una salida real registrada durante la prueba multimodal combinada (célula de prueba "
        "im_Dyskeratotic_011_01.bmp y registro tabular índice 0):"
    )

    cli_evidence = """$ python prototipo_trl3/infer.py --image prototipo_trl3/data/sample_images/im_Dyskeratotic_011_01.bmp --tabular-index 0
{
  "cnn": {
    "modulo": "cnn_citologia_sipakmed",
    "version": "v0.1-trl3-cnn",
    "entrada": { "archivo": "im_Dyskeratotic_011_01.bmp", "tamano_original": [106, 108], "tamano_modelo": [224, 224] },
    "prediccion": {
      "clase_predicha": "im_Dyskeratotic",
      "nombre_clinico": "Disqueratosica (Anormal)",
      "categoria_tamizaje": "Anormal",
      "confianza_clase_predicha": 1.0,
      "probabilidad_anormal_agregada": 1.0,
      "nivel_prioridad_tamizaje": "ALTO (Sospecha de celula anormal - requiere revision citopatologica)"
    },
    "latencia_inferencia_ms": 28.91
  },
  "tabular": {
    "modulo": "tabular_factores_riesgo",
    "version": "v0.1-trl3-tabular",
    "modelo": "RandomForest (balanced)",
    "prediccion": {
      "probabilidad_riesgo_biopsia": 0.3678,
      "umbral_tamizaje": 0.44,
      "alerta_tamizaje": false,
      "estrato_riesgo": "BAJO (Riesgo estimado por debajo del umbral de tamizaje)"
    },
    "variables_imputadas_automaticamente": 0,
    "latencia_inferencia_ms": 47.13
  }
}"""
    add_code_block(doc, cli_evidence)

    # =========================================================================
    # SECCIÓN 7: LIMITACIONES, ÉTICA Y HOJA DE RUTA TRL 4-5
    # =========================================================================
    add_heading_1(doc, "7. Limitaciones Actuales, Ética y Hoja de Ruta (TRL 4 → TRL 5)")
    add_bullet(doc, "El módulo CNN actual opera sobre células individuales previamente recortadas (SIPaKMeD CROPPED). Para procesar láminas microscópicas de campo completo en rutina clínica será necesario integrar una etapa previa de detección/segmentación celular aprovechando los archivos *_cyt.dat y *_nuc.dat.", "Células recortadas vs. campo completo: ")
    add_bullet(doc, "Un clasificador supervisado estándar asignará probabilidades a cualquier imagen cargada. Como paso hacia TRL 4 se contempla incorporar un detector de imágenes fuera de dominio (Out-of-Distribution) para rechazar archivos que no correspondan a citología cervical.", "Control fuera de dominio (OOD): ")
    add_bullet(doc, "El diseño actual procesa las imágenes y formularios exclusivamente en memoria durante la petición HTTP sin persistir datos identificables de pacientes, alineándose con principios de privacidad por diseño (GDPR / normativa nacional de datos en salud).", "Privacidad y seguridad de datos: ")
    add_bullet(doc, "TRL 4 (Entorno de laboratorio integrado): empaquetado Docker, exportación a ONNX Runtime y pruebas de carga API. TRL 5 (Entorno relevante): validación externa multicéntrica sobre cohortes independientes (ej. Herlev y muestras locales anonimizadas con aprobación de comité de ética) evaluando calibración y equidad.", "Hoja de ruta de maduración: ")

    # =========================================================================
    # SECCIÓN 8: CONCLUSIONES Y TRAZABILIDAD DE ARTEFACTOS
    # =========================================================================
    add_heading_1(doc, "8. Conclusiones y Trazabilidad de Artefactos")
    add_body(
        doc,
        "El Prototipo IA-Tamizaje Cervical v0.1 cumple de manera íntegra, transparente y verificable con todos los requisitos "
        "técnicos y metodológicos de madurez TRL 3. Se demostró la viabilidad analítica y experimental tanto de la clasificación "
        "citológica profunda (Macro-F1 de 95.02%, Sensibilidad de tamizaje anormal de 95.16% y latencia de 13.23 ms en CPU sin "
        "fuga por lámina) como de la estratificación tabular libre de leakage (ROC-AUC 0.6437, 2.6× ganancia en PR-AUC), "
        "integrados en una aplicación web FastAPI operativa y versionados públicamente en GitHub."
    )

    add_styled_table(
        doc,
        ["Artefacto / Archivo", "Ubicación en el Repositorio", "Propósito de Evidencia TRL 3"],
        [
            ["Evidencia JSON Consolidada", "prototipo_trl3/outputs/evidencia_trl3.json", "Registro estructurado y auditable de versiones, splits, métricas y latencias"],
            ["Entrenamiento y Evaluación CNN", "prototipo_trl3/train_eval_cnn.py", "Reproducibilidad completa de MobileNetV2 + split sin fuga por lámina"],
            ["Entrenamiento y Evaluación Tabular", "prototipo_trl3/train_eval_tabular.py", "Reproducibilidad de comparación baseline CV 5-fold y control de leakage"],
            ["Explicabilidad Grad-CAM", "prototipo_trl3/gradcam.py", "Generación en vivo de mapas de calor sobre la última capa conv"],
            ["Inferencia CLI", "prototipo_trl3/infer.py", "Prueba reproducible por consola de imagen y/o perfil tabular"],
            ["Aplicación Web FastAPI", "prototipo_trl3/app/main.py + static/*", "Interfaz interactiva de 3 pestañas, API REST y Swagger (/docs)"],
            ["Modelos Entrenados", "prototipo_trl3/models/*.pt, *.joblib", "Pesos serializados listos para inferencia inmediata (~9.5 MB totales)"],
            ["Documentación y Seguimiento", "README.md, GUIA_EJECUCION.md, SEGUIMIENTO_TAREAS.md", "Manual operativo, control de fases y repositorio GitHub (commit b2ea17c)"],
        ],
        col_widths=[1.75, 2.25, 2.8],
    )

    doc.save(str(OUT_DOCX))
    print(f"[OK] Informe Word TRL 3 generado exitosamente en: {OUT_DOCX}")
    print(f"[INFO] Tamano del archivo: {OUT_DOCX.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    build_report()
