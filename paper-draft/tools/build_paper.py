#!/usr/bin/env python3
"""Build paper.tex, references.bib and paper_numbered.md from paper.md + references.json.

paper.md is the single source of truth for the text; references.json for the bibliography.
Only references actually cited in the text are emitted, in order of first use (IEEE numeric).
This script does NOT compile anything (no TeX toolchain was available when it was written).

    python paper-draft/tools/build_paper.py
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "paper.md").read_text()
REFS = {r["key"]: r for r in json.loads((ROOT / "references.json").read_text())}

UNI = {"–": "--", "—": "---", "µ": r"$\mu$", "μ": r"$\mu$", "≤": r"$\le$", "≥": r"$\ge$", "±": r"$\pm$", "×": r"$\times$",
       "−": r"$-$", "≈": r"$\approx$", "Δ": r"$\Delta$", "α": r"$\alpha$", "’": "'", "‘": "`", "“": "``", "”": "''", "é": r"\'e", "ü": r'\"u'}
CITE = re.compile(r"\[@([^\]]+)\]")


def cite_keys(text: str) -> list[str]:
    out: list[str] = []
    for m in CITE.finditer(text):
        for k in re.split(r"\s*;\s*", m.group(1)):
            k = k.strip().lstrip("@")
            if k not in out:
                out.append(k)
    return out


ORDER = cite_keys(SRC)
missing = [k for k in ORDER if k not in REFS]
assert not missing, f"cited but not in references.json: {missing}"
NUM = {k: i + 1 for i, k in enumerate(ORDER)}


# ------------------------------------------------------------------ LaTeX -----------------
def esc(s: str) -> str:
    s = re.sub(r'"([^"]*)"', r"``\1''", s)
    for a, b in UNI.items():
        s = s.replace(a, b)
    s = s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace("_", r"\_").replace("~", r"\textasciitilde{}")
    s = re.sub(r"(?<!\\)<", r"$<$", s)
    s = re.sub(r"(?<!\\)>", r"$>$", s)
    return s


TOK = re.compile(r"(\$[^$]+\$)|(`[^`]+`)|(\[@[^\]]+\])|(\{\{PENDING[^}]*\}\})|(\*\*.+?\*\*)|(\*[^*\s][^*]*?\*)")


def inline(s: str) -> str:
    out, pos = [], 0
    for m in TOK.finditer(s):
        out.append(esc(s[pos:m.start()]))
        t = m.group(0)
        if m.group(1):
            out.append(t)
        elif m.group(2):
            out.append(r"\texttt{" + esc(t[1:-1]) + "}")
        elif m.group(3):
            out.append(r"\cite{" + ",".join(k.strip().lstrip("@") for k in re.split(r"\s*;\s*", t[1:-1])) + "}")
        elif m.group(4):
            inner = t[2:-2][len("PENDING"):]
            out.append(r"\pending{" + esc(inner) + "}")
        elif m.group(5):
            out.append(r"\textbf{" + inline(t[2:-2]) + "}")
        else:
            out.append(r"\emph{" + inline(t[1:-1]) + "}")
        pos = m.end()
    out.append(esc(s[pos:]))
    return "".join(out)


WIDE = {"tab:verdict", "tab:novelty", "tab:preproc", "tab:shap"}


def table_tex(caption: str, label: str, rows: list[list[str]]) -> str:
    n = len(rows[0])
    head, body = rows[0], rows[1:]
    wide = label in WIDE
    if label == "tab:verdict":
        spec, env = r">{\raggedright\arraybackslash}X" * n, "tabularx"
    elif wide:
        spec, env = ">{\\raggedright\\arraybackslash}X" + "c" * (n - 1), "tabularx"
    else:
        spec, env = "l" + "c" * (n - 1), "tabular"
    lines = [rf"\begin{{table{'*' if wide else ''}}}[!t]", rf"\caption{{{inline(caption)}}}", rf"\label{{{label}}}", r"\centering\footnotesize"]
    if env == "tabular":
        lines.append(r"\resizebox{\columnwidth}{!}{%")
        lines.append(rf"\begin{{tabular}}{{{spec}}}")
    else:
        lines.append(rf"\begin{{tabularx}}{{\textwidth}}{{{spec}}}")
    lines += [r"\toprule", " & ".join(r"\textbf{" + inline(c) + "}" for c in head) + r" \\", r"\midrule"]
    lines += [" & ".join(inline(c) for c in r) + r" \\" for r in body]
    lines += [r"\bottomrule", r"\end{tabular}}" if env == "tabular" else r"\end{tabularx}", rf"\end{{table{'*' if wide else ''}}}"]
    return "\n".join(lines)


def build_authors() -> str:
    """Parse the `AUTHORS:` block (one `- Name | role/id line | affiliation` per person) into an
    IEEEtran multi-author \\author{} block, each person getting their own name/affiliation pair
    joined by \\and, matching how IEEEtran lays out author columns."""
    m = re.search(r"^AUTHORS:\n((?:- .*\n?)+)", SRC, re.M)
    people = []
    for line in m.group(1).strip().splitlines():
        parts = [p.strip() for p in line.lstrip("- ").split("|")]
        name, role, affil = (parts + ["", ""])[:3]
        people.append((name, role, affil))
    blocks = []
    for name, role, affil in people:
        role_line = rf"\textit{{{inline(role)}}} \\ " if role else ""
        blocks.append(
            rf"\IEEEauthorblockN{{{inline(name)}}}"
            rf"\IEEEauthorblockA{{{role_line}{inline(affil)}}}"
        )
    return "\\and\n".join(blocks)


def build_tex() -> str:
    head_re = lambda k: re.search(rf"^{k}:\s*(.*)$", SRC, re.M).group(1).strip()  # noqa: E731
    title = head_re("TITLE")
    author_tex = build_authors()
    abstract = re.search(r"^ABSTRACT:\n(.*?)\n\nKEYWORDS:", SRC, re.M | re.S).group(1).strip()
    keywords = head_re("KEYWORDS")
    body = SRC[SRC.index("## I."):]
    out = [
        r"\documentclass[conference]{IEEEtran}",
        r"\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}",
        r"\usepackage{amsmath,amssymb,graphicx,booktabs,tabularx,array,xcolor,url,cite}",
        r"\newcommand{\pending}[1]{\textbf{\textcolor{red}{[PENDING#1]}}}",
        rf"\title{{{inline(title)}}}",
        f"\\author{{{author_tex}}}",
        r"\begin{document}", r"\maketitle",
        r"\begin{abstract}", inline(abstract), r"\end{abstract}",
        r"\begin{IEEEkeywords}", inline(keywords), r"\end{IEEEkeywords}",
    ]
    blocks = re.split(r"\n\s*\n", body.strip())
    i = 0
    while i < len(blocks):
        b = blocks[i].strip()
        if b.startswith("## "):
            out.append(r"\section{" + inline(re.sub(r"^[IVX]+\.\s*", "", b[3:])) + "}")
        elif b.startswith("### "):
            out.append(r"\subsection{" + inline(re.sub(r"^[A-Z]\.\s*", "", b[4:])) + "}")
        elif b.startswith("Table:"):
            m = re.match(r"Table:\s*(.*?)\s*\{#(tab:[\w-]+)\}\s*$", b, re.S)
            tbl = blocks[i + 1].strip().splitlines()
            rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in tbl if not re.match(r"^\|[-\s|:]+\|$", ln)]
            out.append(table_tex(m.group(1), m.group(2), rows))
            i += 1
        elif b.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)\{#(fig:[\w-]+)\}\s*$", b, re.S)
            out.append(rf"\begin{{figure}}[!t]\centering\includegraphics[width=\columnwidth]{{{m.group(2)}}}\caption{{{inline(m.group(1))}}}\label{{{m.group(3)}}}\end{{figure}}")
        elif re.match(r"^\d+\.\s", b):
            out.append(r"\begin{enumerate}")
            out += [r"\item " + inline(re.sub(r"^\d+\.\s*", "", ln)) for ln in b.splitlines()]
            out.append(r"\end{enumerate}")
        else:
            out.append(inline(" ".join(b.splitlines())))
        i += 1
    out += [r"\bibliographystyle{IEEEtran}", r"\bibliography{references}", r"\end{document}"]
    return "\n\n".join(out) + "\n"


# ------------------------------------------------------------------ BibTeX ----------------
# Classic BibTeX (what tectonic/plain LaTeX invokes for a .bst-based bibliography, unlike biber)
# does not reliably round-trip raw UTF-8 bytes through .bbl -> pdfTeX's 8-bit fonts; accented
# Latin letters must be given as LaTeX accent commands instead, or they surface as U+FFFD in the
# compiled PDF (found by actually compiling the paper, not assumed).
_ACCENTS = {
    "ä": '"a', "ë": '"e', "ï": '"i', "ö": '"o', "ü": '"u', "ÿ": '"y',
    "Ä": '"A', "Ë": '"E', "Ï": '"I', "Ö": '"O', "Ü": '"U',
    "á": "'a", "é": "'e", "í": "'i", "ó": "'o", "ú": "'u", "ý": "'y",
    "Á": "'A", "É": "'E", "Í": "'I", "Ó": "'O", "Ú": "'U",
    "à": "`a", "è": "`e", "ì": "`i", "ò": "`o", "ù": "`u",
    "À": "`A", "È": "`E", "Ì": "`I", "Ò": "`O", "Ù": "`U",
    "â": "^a", "ê": "^e", "î": "^i", "ô": "^o", "û": "^u",
    "Â": "^A", "Ê": "^E", "Î": "^I", "Ô": "^O", "Û": "^U",
    "ã": "~a", "ñ": "~n", "õ": "~o", "Ã": "~A", "Ñ": "~N", "Õ": "~O",
    "ç": "c{c}", "Ç": "c{C}", "ø": "o{o}", "Ø": "o{O}", "å": "r{a}", "Å": "r{A}",
    "ł": "l{}", "Ł": "L{}", "š": "v{s}", "Š": "v{S}", "č": "v{c}", "Č": "v{C}", "ž": "v{z}", "Ž": "v{Z}",
}


def bib_esc(s: str) -> str:
    s = html.unescape(s or "")
    for ch, cmd in _ACCENTS.items():
        s = s.replace(ch, f"\\{cmd}")
    return s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace("_", r"\_")


def bib_entry(k: str) -> str:
    r = REFS[k]
    venue = html.unescape(r.get("venue") or "")
    authors = " and ".join(bib_esc(a) for a in r["authors"])
    f = {"author": authors, "title": "{" + bib_esc(r["title"]) + "}", "year": str(r["year"])}
    conf = re.search(r"Proc\.|Conference|Symposium|Workshop|Neural Information|Summits|Communications in Computer", venue)
    if r.get("preprint"):
        typ = "misc"
        f["note"] = "Preprint, not peer reviewed"
        f["doi"] = r["doi"]
    elif "UCI Machine Learning Repository" in venue:
        typ, f["howpublished"], f["doi"] = "misc", "UCI Machine Learning Repository", r["doi"]
    elif venue == "arXiv":
        typ, f["howpublished"], f["note"] = "misc", "arXiv preprint", f"arXiv:{r['arxiv']}"
    elif r.get("url") and not r.get("doi") and "Summits" not in venue:
        typ, f["howpublished"], f["url"] = "misc", bib_esc(venue), r["url"]
    elif conf:
        typ, f["booktitle"] = "inproceedings", bib_esc(venue)
        for a in ("pages", "doi", "url"):
            if r.get(a):
                f[a] = r[a]
        if r.get("arxiv"):
            f["note"] = f"arXiv:{r['arxiv']}"
    else:
        typ, f["journal"] = "article", bib_esc(venue)
        for a in ("volume", "pages", "doi"):
            if r.get(a):
                f[a] = str(r[a])
        if r.get("issue"):
            f["number"] = str(r["issue"])
        if r.get("arxiv") and venue == "Journal of Machine Learning Research":
            f["note"] = f"arXiv:{r['arxiv']}"
    body = ",\n".join(f"  {a} = {{{v}}}" for a, v in f.items())
    return f"@{typ}{{{k},\n{body}\n}}"


def ieee_text(k: str) -> str:
    r = REFS[k]

    def fmt(name: str) -> str:
        fam, _, giv = name.partition(",")
        ini = " ".join(w[0] + "." for w in re.split(r"[\s.]+", giv.strip()) if w)
        return f"{ini} {fam.strip()}".strip()

    au = [fmt(a) for a in r["authors"]]
    a = ", ".join(au) if len(au) <= 6 else au[0] + " et al."
    venue = html.unescape(r.get("venue") or "")
    tail = venue
    if r.get("volume"):
        tail += f", vol. {r['volume']}"
    if r.get("issue"):
        tail += f", no. {r['issue']}"
    if r.get("pages"):
        tail += f", pp. {r['pages']}"
    ident = f", doi: {r['doi']}" if r.get("doi") else (f", arXiv:{r['arxiv']}" if r.get("arxiv") else (f", {r['url']}" if r.get("url") else ""))
    pre = " (preprint, not peer reviewed)" if r.get("preprint") else ""
    return f"{a}, \"{html.unescape(r['title'])},\" {tail}, {r['year']}{ident}{pre}."


def numbered_md() -> str:
    def sub(m: re.Match) -> str:
        return ", ".join(f"[{NUM[k]}]" for k in cite_keys(m.group(0)))

    text = CITE.sub(sub, SRC)
    text = re.sub(r"\s*\{#(?:tab|fig):[\w-]+\}", "", text)
    refs = "\n".join(f"[{NUM[k]}] {ieee_text(k)}" for k in ORDER)
    return text.rstrip() + "\n\n## References\n\n" + refs + "\n"


def body_words() -> int:
    body = SRC[SRC.index("## I."):]
    keep = []
    for ln in body.splitlines():
        if ln.startswith("|") or ln.startswith("Table:") or ln.startswith("!["):
            continue
        keep.append(ln)
    t = "\n".join(keep)
    t = CITE.sub("", t)
    t = re.sub(r"\{\{PENDING[^}]*\}\}", "", t)
    t = re.sub(r"\$[^$]+\$", "x", t)
    t = re.sub(r"^#+\s*", "", t, flags=re.M)
    return len(re.findall(r"\b[\w'’.-]+\b", t))


if __name__ == "__main__":
    (ROOT / "paper.tex").write_text(build_tex())
    (ROOT / "references.bib").write_text("\n\n".join(bib_entry(k) for k in ORDER) + "\n")
    (ROOT / "paper_numbered.md").write_text(numbered_md())
    unused = sorted(set(REFS) - set(ORDER))
    print(f"cited {len(ORDER)} references; retrieved but not cited: {unused}")
    print(f"body words (Sections I-IX, excluding tables/captions/pending markers/citations): {body_words()}")
