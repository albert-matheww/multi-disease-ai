#!/usr/bin/env python3
"""Save a work's ABSTRACT ONLY (from OpenAlex) to paper-draft/sources/<key>.txt.

Usage: python paper-draft/tools/openalex_abstract.py KEY DOI
Used when the full text is not legitimately retrievable; the source then stays at
read_depth "abstract" and the paper may claim only what the abstract states.
"""
import json, sys, urllib.parse, urllib.request
from pathlib import Path

key, doi = sys.argv[1], sys.argv[2]
url = "https://api.openalex.org/works/" + urllib.parse.quote("https://doi.org/" + doi, safe=":/")
req = urllib.request.Request(url, headers={"User-Agent": "paper-writer-skill/1.0"})
work = json.load(urllib.request.urlopen(req, timeout=30))
index = work.get("abstract_inverted_index")
if not index:
    print(f"[NONE] {key}: OpenAlex has no abstract for {doi}")
    sys.exit(1)
words = sorted(((pos, w) for w, positions in index.items() for pos in positions))
abstract = " ".join(w for _, w in words)
out = Path(__file__).resolve().parents[1] / "sources" / f"{key}.txt"
out.write_text(f"ABSTRACT-ONLY (OpenAlex) | {work['title']} | doi:{doi}\n\n{abstract}\n", encoding="utf-8")
print(f"[ABSTRACT] {key}: {len(abstract)} chars | {work['title'][:80]}")
