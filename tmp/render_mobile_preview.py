from pathlib import Path
import html
import re
from urllib.parse import quote


ROOT = Path.cwd()
MD_PATH = ROOT / "研究内容一_正文版.md"
OUT_PATH = ROOT / "output" / "mobile_preview" / "研究内容一_正文版.html"


CSS = """
body{font-family:Arial,"Microsoft YaHei",sans-serif;max-width:980px;margin:32px auto;line-height:1.72;color:#222;padding:0 18px;background:#fff}
h1,h2,h3,h4{line-height:1.28;margin-top:1.35em}
h1{font-size:30px}
h2{font-size:24px;border-bottom:1px solid #e8e8e8;padding-bottom:6px}
h3{font-size:19px}
p{margin:0 0 12px}
img{max-width:100%;height:auto;border:1px solid #eee;background:#fafafa}
figure{margin:24px 0}
figcaption{font-size:14px;color:#555;text-align:center;margin-top:6px}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:14px}
th,td{border:1px solid #ddd;padding:6px 8px;vertical-align:top}
th{background:#f7f7f7}
code{background:#f5f5f5;padding:1px 4px;border-radius:3px}
pre{background:#f7f7f7;padding:12px;overflow:auto}
.math{font-family:"Times New Roman",serif;color:#333}
"""


def inline(text: str) -> str:
    placeholders = []

    def stash(match):
        placeholders.append(match.group(0))
        return f"@@MATH{len(placeholders)-1}@@"

    text = re.sub(r"\$\$.*?\$\$", stash, text)
    text = re.sub(r"\$[^$]+\$", stash, text)
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    for i, raw in enumerate(placeholders):
        text = text.replace(f"@@MATH{i}@@", f'<span class="math">{html.escape(raw)}</span>')
    return text


def flush_paragraph(out, paragraph):
    if paragraph:
        out.append("<p>" + inline(" ".join(paragraph).strip()) + "</p>")
        paragraph.clear()


def image_html(line: str) -> str:
    match = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
    if not match:
        return f"<p>{inline(line)}</p>"
    alt, src = match.group(1), match.group(2)
    src_path = (ROOT / src).resolve() if not re.match(r"^[a-zA-Z]+:", src) else src
    if isinstance(src_path, Path):
        uri = "file:///" + quote(str(src_path).replace("\\", "/"), safe="/:")
    else:
        uri = src_path
    return f'<figure><img src="{uri}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>'


def parse_table(lines, index, out):
    rows = []
    while index < len(lines) and lines[index].strip().startswith("|"):
        raw = lines[index].strip().strip("|")
        rows.append([cell.strip() for cell in raw.split("|")])
        index += 1

    if len(rows) >= 2 and all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in rows[1]):
        out.append("<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in rows[0]) + "</tr></thead><tbody>")
        for row in rows[2:]:
            out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>")
        out.append("</tbody></table>")
    else:
        out.extend("<p>" + inline(" | ".join(row)) + "</p>" for row in rows)
    return index


def render():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = MD_PATH.read_text(encoding="utf-8-sig").splitlines()
    out = [
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(MD_PATH.stem)}</title><style>{CSS}</style></head><body>"
    ]
    paragraph = []
    in_code = False
    code_buf = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_code:
                flush_paragraph(out, paragraph)
                in_code = True
                code_buf = []
            else:
                out.append("<pre><code>" + html.escape("\n".join(code_buf)) + "</code></pre>")
                in_code = False
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue
        if not stripped:
            flush_paragraph(out, paragraph)
            i += 1
            continue
        if stripped.startswith("|"):
            flush_paragraph(out, paragraph)
            i = parse_table(lines, i, out)
            continue
        if stripped.startswith("!["):
            flush_paragraph(out, paragraph)
            out.append(image_html(stripped))
            i += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            flush_paragraph(out, paragraph)
            level = len(heading.group(1))
            out.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
            i += 1
            continue
        paragraph.append(stripped)
        i += 1
    flush_paragraph(out, paragraph)
    out.append("</body></html>")
    OUT_PATH.write_text("\n".join(out), encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    render()
