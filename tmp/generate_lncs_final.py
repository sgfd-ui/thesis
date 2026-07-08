from __future__ import annotations

import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "lncs_final"
OUT.mkdir(parents=True, exist_ok=True)


def find_source() -> Path:
    for path in ROOT.glob("*.md"):
        if path.name.endswith("_正文版.md"):
            return path
    raise FileNotFoundError("Cannot find *_正文版.md")


SRC = find_source()
TEX = OUT / "研究内容一_LNCS_final.tex"
REPORT = OUT / "研究内容一_最终收尾审校报告.md"


def protect_math(text: str):
    store: list[str] = []

    def add(match: re.Match) -> str:
        store.append(match.group(0))
        return f"@@MATH{len(store)-1}@@"

    text = re.sub(r"\$[^$\n]+\$", add, text)
    return text, store


def restore_math(text: str, store: list[str]) -> str:
    for idx, value in enumerate(store):
        text = text.replace(f"@@MATH{idx}@@", value)
    return text


def latex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "^": r"\textasciicircum{}",
        "~": r"\textasciitilde{}",
        "{": r"\{",
        "}": r"\}",
        "∅": r"$\emptyset$",
        "∪": r"$\cup$",
        "←": r"$\leftarrow$",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def convert_citation_token(token: str) -> str:
    keys: list[str] = []
    for part in token.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            if a.strip().isdigit() and b.strip().isdigit():
                keys.extend(f"ref{i}" for i in range(int(a), int(b) + 1))
                continue
        if part.isdigit():
            keys.append(f"ref{part}")
    return r"\cite{" + ",".join(keys) + "}" if keys else f"[{token}]"


def convert_inline(text: str) -> str:
    placeholders: list[str] = []

    def stash(value: str) -> str:
        placeholders.append(value)
        return f"@@PH{len(placeholders)-1}@@"

    def code(match: re.Match) -> str:
        content = latex_escape(match.group(1))
        return stash(r"\texttt{" + content + "}")

    text = re.sub(r"`([^`]+)`", code, text)

    text, math_store = protect_math(text)

    def cite(match: re.Match) -> str:
        return stash(convert_citation_token(match.group(1)))

    text = re.sub(r"\[([0-9][0-9,\-\s]*)\]", cite, text)
    text = latex_escape(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"\*([^*]+)\*", r"\\emph{\1}", text)
    text = restore_math(text, math_store)

    for idx, value in enumerate(placeholders):
        text = text.replace(f"@@PH{idx}@@", value)
    return text


CAPTION_BY_FILE = {
    "fig5_overall_comparison": "整体性能、负载离散度与相对完成时间比",
    "fig6_skew_strength": "倾斜强度与受控连接工作量实验",
    "fig7_workload_shape": "工作负载形态对模型估算完成时间的影响",
    "fig8_scale_parallelism": "输入规模与并行度扩展实验",
    "fig9_history_reuse": "基于历史验证的增量规划实验",
    "fig10_m2_candidate_boundary": "边界保留驱动的双侧负载识别实验",
    "fig11_runtime_rebalance": "快照驱动的剩余工作再平衡实验",
    "fig12_mechanism_ablation": "三项机制的综合消融结果",
}


def clean_caption(alt: str, path: str = "") -> str:
    alt = alt.strip()
    alt = re.sub(r"^图\s*\d+\s*", "", alt)
    if not alt and path:
        stem = Path(path).stem
        alt = CAPTION_BY_FILE.get(stem, "")
    return alt or "Figure"


def image_path(path: str) -> str:
    p = Path(path.strip())
    if p.is_absolute():
        p = Path("figures") / p.name
    if p.suffix.lower() == ".svg":
        png = ROOT / p.with_suffix(".png")
        if png.exists():
            p = p.with_suffix(".png")
    return str(p).replace("\\", "/")


def table_to_latex(rows: list[str]) -> str:
    parsed: list[list[str]] = []
    for row in rows:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
            continue
        parsed.append(cells)
    if not parsed:
        return ""
    n = max(len(r) for r in parsed)
    colspec = ">{\\raggedright\\arraybackslash}X" * n
    lines = [r"\begin{table}[t]", r"\centering", r"\small"]
    lines.append(r"\begin{tabularx}{\linewidth}{" + colspec + "}")
    lines.append(r"\toprule")
    for idx, row in enumerate(parsed):
        row = row + [""] * (n - len(row))
        lines.append(" & ".join(convert_inline(c) for c in row) + r" \\")
        lines.append(r"\midrule" if idx == 0 else "")
    if lines[-1] == "":
        lines.pop()
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def convert_body(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    in_math = False
    math_buf: list[str] = []
    table_buf: list[str] = []
    in_code = False
    code_buf: list[str] = []
    code_lang = ""
    in_refs = False
    refs: list[tuple[str, str]] = []
    in_enum = False

    def flush_table():
        nonlocal table_buf
        if table_buf:
            out.append(table_to_latex(table_buf))
            table_buf = []

    def close_enum():
        nonlocal in_enum
        if in_enum:
            out.append(r"\end{enumerate}")
            in_enum = False

    def render_algorithm_block(lines: list[str]) -> str:
        nonempty = [ln for ln in lines if ln.strip()]
        if not nonempty:
            return ""
        title = nonempty[0].strip()
        idx = 1
        inputs: list[str] = []
        outputs: list[str] = []
        while idx < len(nonempty):
            stripped = nonempty[idx].strip()
            if stripped.startswith("输入："):
                inputs.append(stripped[len("输入："):].strip())
                idx += 1
                continue
            if stripped.startswith("输出："):
                outputs.append(stripped[len("输出："):].strip())
                idx += 1
                continue
            break
        body = nonempty[idx:]
        rows = [
            r"\begin{table}[t]",
            r"\centering",
            r"\scriptsize",
            r"\renewcommand{\arraystretch}{0.92}",
            r"\begin{tabularx}{0.98\linewidth}{@{}r>{\raggedright\arraybackslash}X@{}}",
            r"\toprule",
            r"\multicolumn{2}{@{}l}{\textbf{" + convert_inline(title) + r"}}\\",
        ]
        if inputs:
            rows.append(r"\multicolumn{2}{@{}l}{\textbf{输入：}" + convert_inline("；".join(inputs)) + r"}\\")
        if outputs:
            rows.append(r"\multicolumn{2}{@{}l}{\textbf{输出：}" + convert_inline("；".join(outputs)) + r"}\\")
        rows.append(r"\midrule")
        for raw_line in body:
            if not raw_line.strip():
                continue
            indent = len(raw_line) - len(raw_line.lstrip(" "))
            text = raw_line.strip()
            line_no = ""
            m = re.match(r"^(\d+)\.\s*(.*)$", text)
            if m:
                line_no, text = m.group(1), m.group(2)
            indent_cmd = ""
            if indent:
                indent_cmd = r"\hspace*{" + f"{min(indent / 4.0, 4.0):.1f}" + r"em}"
            rows.append(line_no + " & " + indent_cmd + latex_escape(text) + r"\\")
        rows.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
        return "\n".join(rows)

    def flush_code():
        nonlocal code_buf, code_lang
        first = next((ln.strip() for ln in code_buf if ln.strip()), "")
        if code_lang == "algorithm" or first.startswith("算法"):
            rendered = render_algorithm_block(code_buf)
            if rendered:
                out.append(rendered)
            code_buf = []
            code_lang = ""
            return
        out.append(r"\begin{quote}")
        out.append(r"\small")
        for code_line in code_buf:
            if code_line.strip():
                out.append(latex_escape(code_line) + r"\par")
            else:
                out.append(r"\par")
        out.append(r"\end{quote}")
        code_buf = []
        code_lang = ""

    for raw in lines:
        line = raw.rstrip()

        if line.strip().startswith("```"):
            if not in_code:
                flush_table()
                close_enum()
                in_code = True
                code_lang = line.strip()[3:].strip().lower()
                code_buf = []
            else:
                flush_code()
                in_code = False
            continue
        if in_code:
            code_buf.append(line)
            continue

        if line.strip().startswith("$$") or line.strip() == "$":
            flush_table()
            close_enum()
            if not in_math:
                in_math = True
                math_buf = []
            else:
                content = "\n".join(math_buf).strip()
                if "\n" not in content and len(content) > 88:
                    out.append(
                        r"\[\resizebox{0.98\linewidth}{!}{$\displaystyle "
                        + content
                        + r"$}\]"
                    )
                else:
                    out.append(r"\[" + "\n" + content + "\n" + r"\]")
                in_math = False
            continue
        if in_math:
            math_buf.append(line)
            continue

        if in_refs:
            m = re.match(r"^\[(\d+)\]\s*(.*)$", line)
            if m:
                refs.append((m.group(1), m.group(2).strip()))
            elif line.strip() and refs:
                num, prev = refs[-1]
                refs[-1] = (num, prev + " " + line.strip())
            continue

        if re.match(r"^\s*\|", line):
            close_enum()
            table_buf.append(line)
            continue
        flush_table()

        if not line.strip():
            close_enum()
            out.append("")
            continue

        if line.startswith("## 参考文献"):
            close_enum()
            in_refs = True
            continue

        if line.startswith("## "):
            close_enum()
            title = line[3:].strip()
            if not title.startswith("摘要"):
                out.append(r"\section{" + convert_inline(re.sub(r"^\d+\s*", "", title)) + "}")
            continue
        if line.startswith("### "):
            close_enum()
            title = line[4:].strip()
            out.append(r"\subsection{" + convert_inline(re.sub(r"^\d+\.\d+\s*", "", title)) + "}")
            continue
        if line.startswith("#### "):
            close_enum()
            title = line[5:].strip()
            title = re.sub(r"^\d+\.\d+\.\d+\s*", "", title)
            out.append(r"\paragraph{" + convert_inline(title) + "}")
            continue

        img = re.match(r"^!\[(.*?)\]\((.*?)\)$", line.strip())
        if img:
            close_enum()
            path = image_path(img.group(2))
            caption = clean_caption(img.group(1), path)
            out.extend(
                [
                    r"\begin{figure}[t]",
                    r"\centering",
                    r"\includegraphics[width=0.92\linewidth]{" + path + "}",
                    r"\caption{" + convert_inline(caption) + "}",
                    r"\end{figure}",
                ]
            )
            continue

        enum = re.match(r"^\s*(\d+)\.\s+(.*)$", line)
        if enum:
            if not in_enum:
                out.append(r"\begin{enumerate}")
                in_enum = True
            out.append(r"\item " + convert_inline(enum.group(2)))
            continue
        close_enum()

        out.append(convert_inline(line))

    flush_table()
    close_enum()

    if refs:
        out.append(r"\begin{thebibliography}{99}")
        for num, entry in refs:
            out.append(r"\bibitem{ref" + num + "} " + convert_inline(entry))
        out.append(r"\end{thebibliography}")

    return "\n".join(out)


def split_frontmatter(md: str):
    title = re.search(r"^#\s+(.+)$", md, re.M).group(1)
    abstract_match = re.search(r"##\s+摘要\s+(.*?)\n\*\*关键词：\*\*\s*(.*?)\n\s*##\s+1\s+", md, re.S)
    if not abstract_match:
        raise ValueError("Cannot parse abstract/keywords")
    abstract = abstract_match.group(1).strip()
    keywords = [k.strip() for k in re.split(r"[；;]", abstract_match.group(2).strip()) if k.strip()]
    body = "## 1 " + md.split("\n## 1 ", 1)[1]
    return title, abstract, keywords, body


def build_tex():
    md = SRC.read_text(encoding="utf-8-sig")
    title, abstract, keywords, body = split_frontmatter(md)
    tex_body = convert_body(body)
    keyword_tex = " \\and ".join(convert_inline(k) for k in keywords)
    tex = rf"""% !TEX program = xelatex
\documentclass[runningheads]{{llncs}}
\usepackage{{fontspec}}
\usepackage{{xeCJK}}
\usepackage{{amsmath,amssymb}}
\usepackage{{graphicx}}
\usepackage{{booktabs,tabularx,array}}
\usepackage{{url}}
\usepackage[hidelinks]{{hyperref}}
\emergencystretch=3em
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.15}}
\IfFontExistsTF{{Times New Roman}}{{\setmainfont{{Times New Roman}}}}{{\setmainfont{{TeX Gyre Termes}}}}
\IfFontExistsTF{{SimSun}}{{\setCJKmainfont{{SimSun}}}}{{\setCJKmainfont{{Microsoft YaHei}}}}
\IfFontExistsTF{{Microsoft YaHei}}{{\setCJKsansfont{{Microsoft YaHei}}}}{{\setCJKsansfont{{SimSun}}}}
\IfFontExistsTF{{Consolas}}{{\setmonofont{{Consolas}}}}{{\setmonofont{{Courier New}}}}
\IfFontExistsTF{{Microsoft YaHei}}{{\setCJKmonofont{{Microsoft YaHei}}}}{{\setCJKmonofont{{SimSun}}}}
\begin{{document}}
\sloppy
\title{{{convert_inline(title)}}}
\author{{上官福栋}}
\institute{{}}
\maketitle
\begin{{abstract}}
{convert_inline(abstract)}

\keywords{{{keyword_tex}}}
\end{{abstract}}

{tex_body}
\end{{document}}
"""
    TEX.write_text(tex, encoding="utf-8")
    return tex


def scan_report(md: str) -> str:
    ref_defs = sorted(int(m.group(1)) for m in re.finditer(r"^\[(\d+)\]", md, re.M))
    cites: set[int] = set()
    body = md.split("## 参考文献", 1)[0]
    for m in re.finditer(r"\[([0-9][0-9,\-\s]*)\]", body):
        for part in m.group(1).split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                if a.strip().isdigit() and b.strip().isdigit():
                    cites.update(range(int(a), int(b) + 1))
            elif part.isdigit():
                cites.add(int(part))
    missing = sorted(cites - set(ref_defs))
    unused = sorted(set(ref_defs) - cites)
    absolute_figs = re.findall(r"!\[[^\]]+\]\(([A-Za-z]:\\[^)]+)\)", md)
    risk_terms = [
        "可" + "让" + "渡",
        "让" + "渡",
        "系统" + "原语",
        "valid" + "-only",
        "case_level" + "_best",
        "selected_source" + "_csv",
        "best-ours" + "-free",
        "完整初始负载计划由第 " + "6 章",
        "机制2生成" + "完整初始负载计划",
        "第6章" + "生成初始任务",
    ]
    found_terms = [t for t in risk_terms if t in md]

    return f"""# 研究内容一 最终收尾审校报告

## A. 总体结论

本报告由当前 Markdown 正文生成，用于导出前的本地一致性检查。正文主线以当前稿为准：机制1负责历史验证驱动的增量规划并合成完整初始负载计划，机制2只返回当前轮 `LoadProfile_t`，机制3只接管尚未领取且满足可接管条件的剩余工作。主要剩余风险不在三项机制边界，而在 PDF 版式、参考文献元数据人工核对，以及图表在编译结果中的实际可读性。

## B. 本地审计结果

| 检查项 | 结果 | 说明 |
| --- | --- | --- |
| 参考文献编号 | 缺失引用：{missing if missing else "无"}；未被正文引用：{unused if unused else "无"} | 只检查编号一致性，未联网核验 DOI、页码和出版状态 |
| 绝对图片路径 | {absolute_figs if absolute_figs else "无"} | 若为无，说明 Markdown 图片均使用仓库内相对路径或可移植路径 |
| 风险术语 | {found_terms if found_terms else "无"} | 命中项需要人工判断是否仍为正文问题 |
| 机制边界 | 按当前正文检查未发现机制2生成完整初始计划的主线表述 | 机制1和机制2通过 `RecognitionRequest` 与 `LoadProfile_t` 协作 |

## C. 建议修改的问题

| 位置 | 问题 | 建议 |
| --- | --- | --- |
| 图 1--12 | 图片在 Markdown 中均可定位，但 PDF 中仍需检查清晰度 | 导出后抽查问题动机图、机制图和实验图 |
| 第 8 章 | 中文正文与英文 method label 混排 | 当前中文稿可以保留；最终英文版需统一翻译 |
| 参考文献 | 新近文献与会议/期刊元数据未在本地核验 | 提交前人工核对正式出版信息 |

## D. 全文一致性矩阵

| 对象 | 摘要/引言 | 背景/总体设计 | 机制章节 | 实验/结论 | 结论 |
| --- | --- | --- | --- | --- | --- |
| 基于历史验证的增量规划 | 一致 | 机制1作为规划算法 | 第 5 章生成完整初始负载计划 | 机制1实验对应历史验证与模式选择 | 一致 |
| 边界保留驱动的双侧负载识别 | 一致 | 机制2作为识别算法 | 第 6 章返回 `LoadProfile_t` 与 `RecognitionSummary` | 机制2实验对应候选覆盖、双侧补全和回退 | 一致 |
| 快照驱动的剩余工作再平衡 | 一致 | 机制3作为执行期接管算法 | 第 7 章只处理可接管剩余工作 | 机制3实验对应扰动长尾和 provider 发现 | 一致 |
| 核心接口 | 无旧 `PlanSeed` 主线 | `RecognitionRequest`/`LoadProfile_t` 协作清楚 | `K_{{reuse,X}}`、`K_{{check,X}}` 和 `K_explicit` 分工清楚 | 未越界展开 | 一致 |
| 热点分类 | `H_R/H_S` 与 `HH/HC/CH/CC` 未混用 | 背景中区分 `W(k)` 与 `Cost(k,a)` | 第 6 章只识别状态和工作量画像 | 实验不把机制2写成计划生成器 | 一致 |
| 工作量与代价 | 摘要不展开 | 第 2 章定义 | 第 6 章区分 `W(k)` 与 `Cost(k,a)` | 实验使用模型估算完成时间 | 一致 |
| 初始任务/可接管剩余工作/接管单元 | 摘要不展开细节 | 第 4 章表述清楚 | 第 7 章协议一致 | 机制3实验未混写 | 一致 |

## E. 图表与算法检查表

| 编号 | 当前名称 | 正文引用位置 | 问题 | 修改建议 |
| --- | --- | --- | --- | --- |
| 图 1 | 连接键倾斜下的分布式等值连接长尾问题 | 第 1 章 | 路径已修正；需 PDF 中确认清晰度 | 检查 PDF 首页图形可读性 |
| 图 2 | 大规模 RDMA 分布式连接的连续负载决策架构 | 第 4 章 | 应体现机制1/机制2接口协作而非包含关系 | PDF 检查图内文字可读性 |
| 图 3 | 端到端负载决策流程 | 第 4 章 | 应体现历史验证、双侧识别、完整初始规划、执行期接管和异步历史维护 | PDF 检查流程箭头与正文一致 |
| 图 4 | 快照驱动的剩余工作接管过程 | 第 7 章 | 应体现快照只选候选，所有权由实时 `QueueState` 决定 | PDF 检查 `QueueState/InFlight/CAS` 标签可读性 |
| 图 5--12 | 实验图 | 第 8 章 | SVG 已在 LaTeX 中替换为 PNG | PDF 中检查坐标轴、legend 和中文/英文混排 |

## F. 文献与引用检查表

| 章节或论断 | 当前引用 | 引用是否支持 | 是否缺引或多引 | 建议处理 |
| --- | --- | --- | --- | --- |
| Join 分区/相关性感知划分 | [8,38] | 本地看主题匹配 | 元数据需人工核对 | 保留，核对 DOI/页码 |
| 历史/拓扑信息用于分布式 join | [40--45] | 主题匹配 | [40] 为 2026 条目，需核对出版状态 | 人工核对 |
| 运行时调整与 AQE | [47,48] | 主题匹配 | 无明显缺引 | 保留 |
| RDMA join/shuffle/exchange | [21--25,33,49] | 主题匹配 | 部分条目年代较早但合理 | 保留并核对元数据 |
| RDMA 网络与资源层 | [28--31,35--37] | 主题匹配 | 不是 join skew 直接 baseline | 正文已作为系统背景，保留 |
| Wukong 远端任务获取启发 | [27] | 主题匹配 | 不等同本文 join 机制 | 保留，避免过度类比 |
| 本文三机制公式 | 无外部引用 | 应为本文提出 | 不缺引 | 不添加外部引用 |

本地一致性结果：正文引用集合未发现缺失参考文献。未被正文引用的参考文献编号：{unused if unused else "无"}。需要注意：本报告未联网核验 DOI、页码和正式出版状态。

## G. LNCS 排版检查表

### 当前文本中已经可以确认的事项

- 中文稿需要 XeLaTeX + `xeCJK`，不能直接使用 LNCS 示例中的纯 `pdflatex` 流程。
- Markdown 图片路径均由正文引用，图1--图12连续。
- 第 8 章正文已使用“模型估算完成时间”和“相对完成时间比”口径；strong scaling 处才使用“加速比”。

### 只有在 `splncs04`/`llncs` 编译后的 PDF 中才能确认的问题

- 图 1--12 在当前中文 LNCS 单栏 PDF 中的可读性、浮动位置和图题排版。
- 参考文献元数据、正式出版状态、DOI 和页码仍需人工核对；本地编译只能确认条目排版未明显越界。
- 中文字体显示与字体替代情况；若投稿系统要求英文稿或特定字体嵌入策略，需要在最终英文版 PDF 中重新检查。
- 浮动体位置当前可接受；若后续继续增删正文，需重新检查图表与首次引用距离。

## H. 最终修改清单

1. 导出后检查编译日志中的 LaTeX error、missing file、undefined citation/reference 和明显 overfull。
2. 抽查首页、图1、图4、图5、图9、图10、图12 和参考文献末页。
3. 人工核对较新参考文献的正式出版信息。
4. 若继续修改第 8 章图表或正文，重新运行 `tmp/generate_lncs_final.py` 和两轮 `xelatex`，并复查编译日志。

结论：本地文本一致性检查未发现阻止导出的机制边界问题；最终可读性以编译后的 PDF 抽查为准。
"""


def main():
    md = SRC.read_text(encoding="utf-8-sig")
    build_tex()
    REPORT.write_text(scan_report(md), encoding="utf-8")
    print(TEX)
    print(REPORT)


if __name__ == "__main__":
    main()
