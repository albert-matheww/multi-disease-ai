#!/usr/bin/env python3
"""Save the text of a legitimately accessible source to paper-draft/sources/<key>.txt.

Usage:
    python paper-draft/tools/fetch_source.py KEY URL

PDFs (e.g. arxiv.org/pdf/...) are extracted with pypdf; HTML pages are stripped to
visible text. Nothing here bypasses a paywall: if a URL does not return the paper, the
script says so and the source stays at abstract/metadata read-depth.
"""

from __future__ import annotations

import io
import re
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

SOURCES = Path(__file__).resolve().parents[1] / "sources"


class _Visible(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "header", "footer", "noscript"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "header", "footer", "noscript") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip())


def fetch(url: str) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (paper-writer source fetch)"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read(), response.headers.get("Content-Type", "")


def to_text(body: bytes, content_type: str, url: str) -> str:
    if "pdf" in content_type.lower() or body[:5] == b"%PDF-":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(body))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    parser = _Visible()
    parser.feed(body.decode("utf-8", errors="ignore"))
    return "\n".join(parser.parts)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    key, url = sys.argv[1], sys.argv[2]
    try:
        body, content_type = fetch(url)
        text = to_text(body, content_type, url)
    except Exception as exc:  # noqa: BLE001 - report and move on
        print(f"[FAIL] {key}: {exc}")
        return 1
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < 1500:
        print(f"[THIN] {key}: only {len(text)} chars from {url} - probably not the full text; not saved")
        return 1
    SOURCES.mkdir(parents=True, exist_ok=True)
    (SOURCES / f"{key}.txt").write_text(f"SOURCE-URL: {url}\n\n{text}\n", encoding="utf-8")
    print(f"[OK]   {key}: {len(text):,} chars from {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
