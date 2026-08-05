#!/usr/bin/env python3
"""Convert anna_admet_report.md -> anna_admet_report.pdf using reportlab.

Follows the pattern in ~/python_mac/moltui/charges/make_pdf.py (reportlab
Platypus flowables, explicit column widths). Generalised for this report:
landscape A4 (it has an 18-column summary table), content-adaptive column
widths, emoji -> ASCII (Helvetica has no emoji glyphs), and ####/italic support.

Run: /Users/cafierom/python_mac/moltui/mt-env/bin/python code/make_pdf.py
"""
import pathlib, re
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Table, TableStyle)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

ROOT = pathlib.Path(__file__).resolve().parent.parent  # repo root (code/ is one level down)
MD_IN = ROOT / "anna_admet_report.md"
PDF_OUT = ROOT / "anna_admet_report.pdf"
md = MD_IN.read_text().splitlines()

styles = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=18, spaceAfter=6, spaceBefore=10)
H2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=14, spaceAfter=4, spaceBefore=14)
H3 = ParagraphStyle('H3', parent=styles['Heading3'], fontSize=11, spaceAfter=3, spaceBefore=10,
                    textColor=colors.HexColor('#444444'))
H4 = ParagraphStyle('H4', parent=styles['Heading4'], fontSize=10, spaceAfter=2, spaceBefore=8,
                    textColor=colors.HexColor('#333333'), fontName='Helvetica-Bold')
BODY = ParagraphStyle('BODY', parent=styles['BodyText'], fontSize=10, leading=13, spaceAfter=4)
BQ = ParagraphStyle('BQ', parent=BODY, fontSize=8.5, leading=11,
                    textColor=colors.HexColor('#444444'),
                    backColor=colors.HexColor('#f5f5f5'),
                    leftIndent=8, borderPadding=4, spaceBefore=4, spaceAfter=6)

# cell font scales down for very wide tables
def cell_style(ncol):
    fs = 7 if ncol >= 12 else (8 if ncol >= 8 else 8.5)
    return ParagraphStyle(f'CELL{ncol}', parent=styles['BodyText'], fontSize=fs,
                          leading=fs + 1, spaceAfter=0, spaceBefore=0)
def cellh_style(ncol):
    s = cell_style(ncol); s.fontName = 'Helvetica-Bold'; return s

MARGIN = 1.0 * cm
PAGE_W = landscape(A4)[0] - 2 * MARGIN  # usable width (landscape ~774pt)

EMOJI = {'🟢': '[+]', '🔴': '[!]', '✅': '[+]', '⚠️': '[!]', '⚠': '[!]',
         '⭐': '*', '➡': '->'}

def esc(s):
    # Helvetica (built-in) is WinAnsi only: drop emoji, escape HTML. Em dash ok.
    for k, v in EMOJI.items():
        s = s.replace(k, v)
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))

def inline(s):
    s = esc(s)
    s = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'(?<![\w_])_(.+?)_(?![\w_])', r'<i>\1</i>', s)  # italics, not inside identifiers
    return s

def parse_row(line):
    return [c.strip() for c in line.strip().strip('|').split('|')]

def col_widths(header, rows):
    # content-adaptive: weight per col = min(max_chars_in_col + 1, 40)
    weights = []
    for c in range(len(header)):
        w = len(str(header[c])) + 1
        for r in rows:
            if c < len(r):
                w = max(w, len(str(r[c])) + 1)
        weights.append(max(min(w, 40), 3))
    total = sum(weights)
    return [PAGE_W * (w / total) for w in weights]

SEP_RE = re.compile(r'^\s*\|?[\s:|\-]+\|?\s*$')

flow, i, n = [], 0, len(md)
while i < n:
    line = md[i]
    if not line.strip():
        i += 1; continue
    if line.startswith('#### '):
        flow.append(Paragraph(inline(line[5:]), H4)); i += 1; continue
    if line.startswith('### '):
        flow.append(Paragraph(inline(line[4:]), H3)); i += 1; continue
    if line.startswith('## '):
        flow.append(Paragraph(inline(line[3:]), H2)); i += 1; continue
    if line.startswith('# '):
        flow.append(Paragraph(inline(line[2:]), H1)); i += 1; continue
    if line.startswith('> '):
        bq = []
        while i < n and md[i].startswith('> '):
            bq.append(md[i][2:]); i += 1
        flow.append(Paragraph(inline(' '.join(bq)), BQ)); continue
    if '|' in line and i + 1 < n and SEP_RE.match(md[i + 1]) and '-' in md[i + 1]:
        header = parse_row(line); i += 2
        rows = []
        while i < n and '|' in md[i] and md[i].strip():
            r = parse_row(md[i])
            while len(r) < len(header):
                r.append('')
            rows.append(r[:len(header)]); i += 1
        ncol = len(header)
        cw = col_widths(header, rows)
        cs, chs = cell_style(ncol), cellh_style(ncol)
        data = [[Paragraph(inline(h), chs) for h in header]] + \
               [[Paragraph(inline(c), cs) for c in r] for r in rows]
        t = Table(data, colWidths=cw, repeatRows=1)
        t.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bbbbbb')),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eeeeee')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 3),
            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        flow.append(Spacer(1, 4)); flow.append(t); flow.append(Spacer(1, 6)); continue
    para = []
    while i < n and md[i].strip() and not md[i].startswith(('#', '>', '|')):
        para.append(md[i]); i += 1
    flow.append(Paragraph(inline(' '.join(para)), BODY))

doc = SimpleDocTemplate(str(PDF_OUT), pagesize=landscape(A4),
                        leftMargin=MARGIN, rightMargin=MARGIN,
                        topMargin=MARGIN, bottomMargin=MARGIN,
                        title='ADMET-AI predictions — Anna top 15')
doc.build(flow)
print(f"Wrote {PDF_OUT}  ({PDF_OUT.stat().st_size} bytes)")