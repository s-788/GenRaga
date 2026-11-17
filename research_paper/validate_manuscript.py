"""Validate paper artifacts without changing any project source evidence."""
from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent


def main():
    text = (HERE / "GenRaga_IEEE_Conference.tex").read_text()
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S).group(1)
    citations = [key for group in re.findall(r"\\cite\{([^}]+)\}", text) for key in group.split(",")]
    first_citations = list(dict.fromkeys(citations))
    bibliography = re.findall(r"\\bibitem\{([^}]+)\}", text)
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    references = re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", text)
    audit = json.loads((HERE / "tables/evidence_audit.json").read_text())
    unchanged = all(hashlib.sha256((HERE.parent / relative).read_bytes()).hexdigest() == digest
                    for relative, digest in audit["sources_sha256"].items())
    report = dict(abstract_words=len(abstract.split()),
                  section_count=len(re.findall(r"\\section\{", text)),
                  subsection_count=len(re.findall(r"\\subsection\{", text)),
                  figure_count=len(re.findall(r"\\begin\{figure\*?\}", text)),
                  reference_count=len(bibliography),
                  citations_in_first_appearance_order=first_citations == bibliography,
                  undefined_citations=sorted(set(citations) - set(bibliography)),
                  undefined_cross_references=sorted(set(references) - set(labels)),
                  duplicate_labels=len(labels) != len(set(labels)),
                  numerical_source_hashes_unchanged=unchanged)
    assert 150 <= report["abstract_words"] <= 250
    assert report["section_count"] == 8
    assert report["subsection_count"] >= 32
    assert report["figure_count"] == 5
    assert report["reference_count"] >= 12
    assert report["citations_in_first_appearance_order"]
    assert not report["undefined_citations"]
    assert not report["undefined_cross_references"]
    assert not report["duplicate_labels"]
    assert unchanged

    pdf = HERE / "GenRaga_IEEE_Conference.pdf"
    if pdf.exists():
        import pymupdf
        doc = pymupdf.open(pdf)
        report["pdf_pages"] = len(doc)
        report["page_words"] = [len(page.get_text().split()) for page in doc]
        report["page_sizes_points"] = [list(page.rect) for page in doc]
        report["page_text_beyond_edges"] = []
        for index, page in enumerate(doc):
            for word in page.get_text("words"):
                if word[0] < -1 or word[1] < -1 or word[2] > page.rect.width + 1 or word[3] > page.rect.height + 1:
                    report["page_text_beyond_edges"].append(dict(page=index + 1, text=word[4]))
        joined = "\n".join(page.get_text() for page in doc)
        report["unresolved_pdf_markers"] = "??" in joined
        report["pdf_sha256"] = hashlib.sha256(pdf.read_bytes()).hexdigest()
        assert 8 <= len(doc) <= 10, "Rendered PDF is outside the agreed 8–10-page target"
        assert not report["page_text_beyond_edges"]
        assert not report["unresolved_pdf_markers"]
        log = HERE / "GenRaga_IEEE_Conference.log"
        if log.exists():
            log_text = log.read_text(errors="replace")
            report["tex_overfull_boxes"] = re.findall(r"Overfull \\[hv]box[^\n]*", log_text)
            report["tex_undefined_warnings"] = re.findall(r"[^\n]*(?:undefined|multiply defined)[^\n]*", log_text)
    (HERE / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
