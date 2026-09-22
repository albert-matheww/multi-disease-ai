#!/usr/bin/env python3
"""Build paper-draft/references.json from authoritative records (never from memory).

Each seed names an identifier (DOI / arXiv id / Europe PMC id) plus what *I* observed
about it (read depth, what it is used for, and - only where verified in the text I read or
in a registry - a venue). The script fetches title, authors, year, volume, pages from
Crossref / arXiv / Europe PMC, and writes the merged records. Run verify_refs.py on the
result afterwards.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

UA = {"User-Agent": "paper-writer-skill/1.0"}
OUT = Path(__file__).resolve().parents[1] / "references.json"
ATOM = {"a": "http://www.w3.org/2005/Atom"}


def get(url: str) -> bytes:
    for attempt in range(3):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40).read()
        except Exception:  # noqa: BLE001
            if attempt == 2:
                raise
            time.sleep(3)


def crossref(doi: str) -> dict:
    m = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/()")))["message"]
    printed = (m.get("published-print") or {}).get("date-parts", [[None]])[0][0]
    return {
        "title": " ".join((m.get("title") or [""])[0].replace("\xa0", " ").split()),
        "authors": [f"{a.get('family', '')}, {a.get('given', '')}".strip(", ") for a in m.get("author", [])],
        "year": printed or m["issued"]["date-parts"][0][0],
        "venue": (m.get("container-title") or [""])[0],
        "volume": m.get("volume"), "issue": m.get("issue"), "pages": m.get("page"), "doi": m["DOI"],
    }


def arxiv(arxiv_id: str) -> dict:
    root = ET.fromstring(get(f"https://export.arxiv.org/api/query?id_list={arxiv_id}"))
    e = root.find("a:entry", ATOM)
    names = [a.findtext("a:name", "", ATOM) for a in e.findall("a:author", ATOM)]
    return {
        "title": " ".join(e.findtext("a:title", "", ATOM).split()),
        "authors": [f"{n.split()[-1]}, {' '.join(n.split()[:-1])}" for n in names],
        "year": int(e.findtext("a:published", "", ATOM)[:4]),
        "venue": "arXiv", "arxiv": arxiv_id,
    }


# Seeds: what was observed. `venue`/`year` overrides are used only where the venue string
# was verified in the text that was read (see 02_literature_map.md) or in a registry.
SEEDS = [
    dict(key="hollmann2025", doi="10.1038/s41586-024-08328-6", read="full", used="TabPFN definition, size limits, native handling of missing values/categoricals, benchmark claim"),
    dict(key="hollmann2023", arxiv="2207.01848", venue="Proc. International Conference on Learning Representations (ICLR)", year=2023, read="full", used="original TabPFN: limits (1,000 rows, 100 numeric features, 10 classes)"),
    dict(key="muller2022", arxiv="2112.10510", venue="Proc. International Conference on Learning Representations (ICLR)", year=2022, read="full", used="prior-data fitted networks (PFNs) approximate Bayesian inference in one forward pass"),
    dict(key="grinsztajn2022", arxiv="2207.08815", read="full", used="trees vs deep learning on ~10K-sample tabular data (pre-TabPFN-v2 evidence)"),
    dict(key="closerlook2025", arxiv="2502.17361", read="abstract", used="analysis of TabPFN v2 strengths and limits (abstract only)"),
    dict(key="clinbench2026", doi="10.64898/2026.02.02.26345274", read="abstract", preprint=True, used="clinical benchmark: TabPFN competitive, rarely best (preprint, abstract only)"),
    dict(key="realtime2026", arxiv="2603.29946", venue="2nd FM4Science Workshop at ICLR", year=2026, read="full", used="KernelSHAP cost on tabular foundation models; ShapPFN"),
    dict(key="lundberg2017", arxiv="1705.07874", venue="Advances in Neural Information Processing Systems (NIPS)", year=2017, read="full", used="SHAP values"),
    dict(key="rundel2024", doi="10.1007/978-3-031-63797-1_23", read="full", used="TabPFN-specific Shapley/LOCO via in-context learning"),
    dict(key="kumar2020", arxiv="2002.11097", read="full", used="limits of Shapley-value feature attributions"),
    dict(key="vovk2015", doi="10.1007/s10472-013-9368-4", verified_note="Crossref: online 2013, print 2015 (vol. 74, issue 1-2, pp. 9-28); cited by print year", read="full", used="cross-conformal prediction (validity studied empirically)"),
    dict(key="barber2021", doi="10.1214/20-AOS1965", read="full", used="jackknife interval form (Eq. 7); jackknife+/CV+ guarantee"),
    dict(key="angelopoulos2021", arxiv="2107.07511", read="full", used="marginal coverage definition; score function determines usefulness"),
    dict(key="lee2018", arxiv="1807.03888", venue="Advances in Neural Information Processing Systems (NIPS)", year=2018, read="full", used="Mahalanobis-distance OOD detection on network features"),
    dict(key="guo2017", arxiv="1706.04599", read="full", used="probability calibration of modern classifiers"),
    dict(key="shan2015", doi="10.1371/journal.pone.0127272", read="full", used="Youden index as a cut-point criterion"),
    dict(key="ledoit2004", doi="10.1016/s0047-259x(03)00096-4", read="metadata", used="Ledoit-Wolf shrinkage; attribution taken from the scikit-learn documentation"),
    dict(key="kapoor2023", doi="10.1016/j.patter.2023.100804", read="full", used="taxonomy of data leakage; L1: no clean train/test separation in preprocessing"),
    dict(key="varoquaux2018", doi="10.1016/j.neuroimage.2017.06.061", read="full", used="small-sample error bars in cross-validation"),
    dict(key="nadeau2003", doi="10.1023/A:1024068626366", read="full", used="corrected resampled t-test (variance factor 1/J + n2/n1)"),
    dict(key="brown2001", doi="10.1214/ss/1009213286", read="abstract", used="recommends the Wilson interval for small n"),
    dict(key="diciccio1996", doi="10.1214/ss/1032280214", read="abstract", used="bootstrap confidence intervals"),
    dict(key="pedregosa2011", arxiv="1201.0490", venue="Journal of Machine Learning Research", year=2011, read="full", used="scikit-learn"),
    dict(key="detrano1989", doi="10.1016/0002-9149(89)90524-9", read="abstract", used="origin of the Cleveland heart data (303 patients) and its multi-site external testing"),
    dict(key="straw2022", doi="10.1136/bmjhci-2021-100457", read="full", used="ILPD models show a sex disparity in false-negative rate"),
    dict(key="islam2023", doi="10.1016/j.jpi.2023.100189", read="abstract", used="CKD ML study: XGBoost accuracy 0.983 on the UCI CKD data"),
    dict(key="bouqentar2024", doi="10.1016/j.heliyon.2024.e38731", read="abstract", used="heart-disease ML with feature engineering on Cleveland/Statlog"),
]

MANUAL = [
    dict(key="uci_heart", title="Heart Disease", authors=["Janosi, Andras", "Steinbrunn, William", "Pfisterer, Matthias", "Detrano, Robert"], year=1989, venue="UCI Machine Learning Repository", doi="10.24432/C52P4X", read="full", used="Cleveland heart data; citation as suggested by the UCI page (donated 1988-06-30)"),
    dict(key="uci_ckd", title="Chronic Kidney Disease", authors=["Rubini, L.", "Soundarapandian, P.", "Eswaran, P."], year=2015, venue="UCI Machine Learning Repository", doi="10.24432/C5G020", read="full", used="CKD data (400 instances, missing values)"),
    dict(key="uci_ilpd", title="ILPD (Indian Liver Patient Dataset)", authors=["Ramana, Bendi", "Venkateswarlu, N."], year=2022, venue="UCI Machine Learning Repository", doi="10.24432/C5D02C", read="full", used="ILPD (583 instances); citation year as suggested by the UCI page (donated 2012-05-20)"),
    dict(key="ehrsmall2026", verified_note="Not in Crossref/OpenAlex/arXiv; confirmed via Europe PMC record PMC13274287 (title, authors, AMIA Jt Summits proceedings, 2026, pp. 604-613), 2026-09-21", title="Large Models for Small Tables: Adapting Tabular Foundation Models to EHR Data", authors=["Zhu, R.", "Zhou, X.", "Liang, I.", "Scherer, S. W.", "Xu, K.", "Ohno-Machado, L."], year=2026, venue="AMIA Joint Summits on Translational Science Proceedings", pages="604-613", url="https://pmc.ncbi.nlm.nih.gov/articles/PMC13274287/", read="abstract", used="fine-tuned TabPFN reported to consistently outperform conventional methods on clinical prediction tasks (abstract only)"),
    dict(key="smith1988", verified_note="OpenAlex title match resolves to the PMC copy (PMC2245318); citation details as listed in the dataset documentation", title="Using the ADAP learning algorithm to forecast the onset of diabetes mellitus", authors=["Smith, J. W.", "Everhart, J. E.", "Dickson, W. C.", "Knowler, W. C.", "Johannes, R. S."], year=1988, venue="Proc. Symposium on Computer Applications and Medical Care", pages="261-265", read="metadata", used="listed as the past-usage study in the Pima dataset documentation (details taken from that documentation)"),
    dict(key="pima_docs", verified_note="Documentation file fetched and read 2026-09-21; not a scholarly work, so not indexed", title="Pima Indians Diabetes Database: dataset documentation (pima-indians-diabetes.names)", authors=["National Institute of Diabetes and Digestive and Kidney Diseases"], year=1990, venue="mirrored at github.com/jbrownlee/Datasets", url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names", read="full", used="Pima population (female, >=21, Pima heritage), 768 instances"),
]


def main() -> None:
    records = []
    for seed in SEEDS:
        print("fetch", seed["key"])
        base = crossref(seed["doi"]) if "doi" in seed else arxiv(seed["arxiv"])
        rec = {**base, **{k: v for k, v in seed.items() if k in ("venue", "year")}}
        if "arxiv" in seed:
            rec["arxiv"] = seed["arxiv"]
        rec.update(key=seed["key"], read_depth=seed["read"], used_for=seed["used"], preprint=seed.get("preprint", False), verified_note=seed.get("verified_note"))
        records.append(rec)
        time.sleep(0.3)
    for m in MANUAL:
        rec = {k: v for k, v in m.items() if k not in ("read", "used")}
        rec.update(read_depth=m["read"], used_for=m["used"], preprint=False)
        records.append(rec)
    OUT.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"wrote {len(records)} records to {OUT}")


if __name__ == "__main__":
    main()
