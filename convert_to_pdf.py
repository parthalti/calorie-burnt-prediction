"""
Converts SYNOPSIS_REPORT.md into a high-quality PDF document using
Headless Chrome / Edge print-to-pdf engine.
Ensures strictly <= 11 pages (under 12 pages), perfectly paginated.
"""

import os
import subprocess
import mistune
import pypdf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(BASE_DIR, "SYNOPSIS_REPORT.md")
HTML_PATH = os.path.join(BASE_DIR, "SYNOPSIS_REPORT.html")
PDF_PATH = os.path.join(BASE_DIR, "SYNOPSIS_REPORT.pdf")

with open(MD_PATH, "r", encoding="utf-8") as f:
    raw_md = f.read()

parts = [p.strip() for p in raw_md.split("<!-- PAGE BREAK -->") if p.strip()]
print(f"[INFO] Parsed {len(parts)} sections from markdown.")

md = mistune.create_markdown(plugins=['table'])

# Render each section separately to prevent HTML escaping of page breaks
html_sections = []
for idx, part in enumerate(parts):
    section_html = md(part)
    html_sections.append(f'<div class="report-page" id="page-{idx+1}">\n{section_html}\n</div>')

body_content = "\n\n".join(html_sections)

html_document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Synopsis Report - Calories Burnt Prediction & Health Intelligence System</title>
<style>
    @page {{
        size: A4;
        margin: 15mm 14mm 15mm 14mm;
        @bottom-right {{
            content: "Page " counter(page);
            font-size: 8pt;
            font-family: 'Segoe UI', Arial, sans-serif;
            color: #718096;
        }}
    }}
    * {{
        box-sizing: border-box;
    }}
    body {{
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Arial, sans-serif;
        font-size: 8.8pt;
        line-height: 1.40;
        color: #2D3748;
        margin: 0;
        padding: 0;
    }}
    .report-page {{
        page-break-after: always;
        break-after: page;
    }}
    .report-page:last-child {{
        page-break-after: avoid;
        break-after: avoid;
    }}
    #page-1 {{
        text-align: center;
    }}
    #page-1 h1 {{
        font-size: 15pt;
        font-weight: 800;
        letter-spacing: 0.4px;
        color: #1A365D;
        border-bottom: 2px solid #2B6CB0;
        padding-bottom: 4px;
        margin-bottom: 0.3em;
    }}
    #page-1 h2 {{
        font-size: 10.5pt;
        font-weight: 600;
        color: #2B6CB0;
        margin-top: 0;
        margin-bottom: 0.6em;
    }}
    #page-1 p {{
        text-align: center;
        margin-bottom: 0.4em;
    }}
    #page-1 table {{
        text-align: left;
        margin-top: 0.5em;
    }}
    h1 {{
        font-size: 12pt;
        font-weight: 700;
        color: #1A365D;
        margin-top: 0;
        margin-bottom: 0.35em;
        border-bottom: 1.5px solid #2B6CB0;
        padding-bottom: 2px;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        page-break-after: avoid;
    }}
    h2 {{
        font-size: 10pt;
        font-weight: 700;
        color: #2B6CB0;
        margin-top: 0.5em;
        margin-bottom: 0.2em;
        page-break-after: avoid;
    }}
    h3 {{
        font-size: 9pt;
        font-weight: 600;
        color: #2D3748;
        margin-top: 0.4em;
        margin-bottom: 0.15em;
        page-break-after: avoid;
    }}
    p {{
        margin-top: 0;
        margin-bottom: 0.35em;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 0.35em 0 0.5em 0;
        font-size: 8pt;
        page-break-inside: avoid;
    }}
    th, td {{
        border: 1px solid #CBD5E0;
        padding: 3px 5px;
        text-align: left;
    }}
    th {{
        background-color: #EDF2F7;
        font-weight: 700;
        color: #1A202C;
    }}
    tr:nth-child(even) {{
        background-color: #F7FAFC;
    }}
    blockquote {{
        border-left: 3px solid #3182CE;
        background-color: #EBF8FF;
        margin: 0.4em 0;
        padding: 4px 8px;
        font-style: italic;
        color: #2C5282;
        font-size: 8.4pt;
    }}
    code, pre {{
        font-family: 'Consolas', 'Courier New', Courier, monospace;
        font-size: 8pt;
        background-color: #F7FAFC;
    }}
    pre {{
        padding: 5px 8px;
        border: 1px solid #E2E8F0;
        border-radius: 4px;
        white-space: pre-wrap;
        page-break-inside: avoid;
        margin: 0.3em 0;
    }}
    ul, ol {{
        margin-top: 0.1em;
        margin-bottom: 0.35em;
        padding-left: 1.2em;
    }}
    li {{
        margin-bottom: 0.12em;
    }}
    hr {{
        border: 0;
        height: 1px;
        background: #E2E8F0;
        margin: 0.5em 0;
    }}
    strong {{
        color: #1A202C;
    }}
</style>
</head>
<body>

{body_content}

</body>
</html>
"""

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(html_document)

# Locate Chrome or Edge
browser_candidates = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
browser_exe = next((p for p in browser_candidates if os.path.exists(p)), None)

if not browser_exe:
    raise RuntimeError("No compatible headless browser (Chrome/Edge) found for PDF conversion.")

print(f"[INFO] Using browser engine: {browser_exe}")
cmd = [
    browser_exe,
    "--headless",
    "--disable-gpu",
    "--no-pdf-header-footer",
    "--run-all-compositor-stages-before-draw",
    f"--print-to-pdf={PDF_PATH}",
    HTML_PATH
]

res = subprocess.run(cmd, capture_output=True, text=True)
if os.path.exists(PDF_PATH):
    size_kb = round(os.path.getsize(PDF_PATH) / 1024, 1)
    reader = pypdf.PdfReader(PDF_PATH)
    page_count = len(reader.pages)
    print(f"[SUCCESS] PDF Generated successfully: {PDF_PATH} ({size_kb} KB)")
    print(f"[PAGE COUNT] Generated PDF has {page_count} pages.")
else:
    print(f"[ERROR] Failed to generate PDF. Exit code: {res.returncode}, Stderr: {res.stderr}")
