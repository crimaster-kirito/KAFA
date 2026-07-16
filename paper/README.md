# Paper draft

`main.tex` is a venue-neutral two-column research draft generated from the frozen experiment reports. It is intentionally labelled as an evaluation paper and preserves failed claim gates.

Build from this directory with a standard TeX installation:

```bash
cd figures
pdflatex framework_source.tex
mv framework_source.pdf framework.pdf
cd ../..
.venv/bin/python kafa_script/generate_paper_figures.py
cd paper/figures
pdflatex evidence_source.tex
mv evidence_source.pdf evidence.pdf
cd ..
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

From the repository root, `bash paper/build_submission.sh` regenerates both
vector figures and derived reports, builds the manuscript and checklist, and
runs structural PDF, citation, layout, font-embedding, anonymity, split/hash,
headline-number, and research-unit-test checks.

The current host now provides pdfTeX and BibTeX. On 2026-07-15 the seven-page
manuscript completed the build above with the official AAAI-27 anonymous style.
It contains two vector figures, four tables, and 23 resolved references; the
two-panel evidence plot is regenerated from validated raw JSON. The build
checks the complete AG News and TREC server matrices, GSM8K and natural-label
evidence boundaries, headline numbers, and 70 focused research tests. Build
dimensions and diagnostics are checked after every result update rather than
treated as fixed prose. The PDF uses embedded Type 1 fonts and contains no
hyperlinks or bookmarks. The separately compiled checklist is two pages; its
only diagnostics are harmless underfull-box notices inherited from the
official question layout.

AAAI-27 is the time-sensitive stretch target: its official main-track call limits submissions to seven content pages and lists abstract and paper deadlines of 2026-07-21 and 2026-07-28 (AoE), respectively. `aaai2027.sty` and `aaai2027.bst` were extracted unchanged from the official `AuthorKit27.zip` (archive SHA-256 `e28c6ac9bc6eb3b4e2d849547d2cefb5162610ee39d0a12e0dc62d1126b44a7d`). ACL-27 is the fallback; its official site has not yet announced an ARR or commitment deadline.

The required checklist is maintained separately in `ReproducibilityChecklist.tex`, as directed by the AAAI-27 submission instructions. The manuscript includes an AI-use disclosure; AAAI explicitly requires authors to document AI assistance and remain responsible for all content.

Authoritative numerical sources are the generated reports in `../kafa_script/analysis_outputs/`; do not update table values without regenerating and revalidating their raw JSON inputs.
`../kafa_script/TREC_PROTOCOL_CLARIFICATION.md` records that the frozen
`teacher_temperature` CLI field is inert metadata; the verified TREC payloads
use untempered `softmax(-candidate_loss)` probabilities.
