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

The current host now provides pdfTeX and BibTeX.  The restructured manuscript
compiles with the official AAAI-27 anonymous style to seven pages.  References
start on page six, within the seven-content-page limit.  Its current visual core
is an evidence-aware framework, an evidence ledger, and a verified IMDb result
table and vector figure; the four-page
supplement retains the legacy experimental figure, closest-work boundary, and
five detailed tables.
The submission jointly resolves 31 references after the direct-neighbor
literature audit.  The build
checks the complete AG News and TREC server matrices, GSM8K and natural-label
evidence boundaries, headline numbers, and the focused research tests. Build
dimensions and diagnostics are checked after every result update rather than
treated as fixed prose. The PDF uses embedded Type 1 fonts and contains no
hyperlinks or bookmarks. The separately compiled checklist is two pages; its
only diagnostics are harmless underfull-box notices inherited from the
official question layout.

CCF-A evidence strengthening is currently active.  The original IMDb 40-record
official matrix, 30-record matched-K addendum, and conditional implementation
contract now validate completely.  Their fail-closed generator has produced the
official result block, table, statistics ledger, and vector figure.  The frozen
negative-direction gate fails because the protocol-frozen D--C effect is
significantly positive; the manuscript reports that reversal directly.  The
cross-task generator now verifies the full 150-record input set:
40 TREC + 40 IMDb + 30 matched-\(K\) + 40 AG News records.  Its current-input
manifest binds all published server-audit asset hashes.  The latest complete
focused submission build passed 137 tests; the newer selector/server-audit
subset previously passed 33 tests.  The post-hoc IMDb selector-component
ablation now verifies all 30 records and has a separate summarizer/test path;
rerun the focused build and artifact packager after any manuscript update.
Legacy repository-wide optional-dependency failures
are tracked separately in
`../RESEARCH_AUDIT.md`.

`supplement.tex` compiles independently to a four-page anonymous
`supplement.pdf`. It retains the full server configuration, legacy server
table, closest-work boundary, detailed Co-Teaching, Hausa/Yoruba, and GSM8K
tables, and the original two-panel evidence figure, together with an
artifact-reconstruction and fail-closed failure-policy section, allowing the final main
paper to concentrate its limited content pages on the server-transfer audit.
The supplement currently compiles without overflow; its remaining diagnostics
are harmless underfull-vbox notices.

AAAI-27 is the time-sensitive stretch target: its official main-track call limits submissions to seven content pages and lists abstract and paper deadlines of 2026-07-21 and 2026-07-28 (AoE), respectively. `aaai2027.sty` and `aaai2027.bst` were extracted unchanged from the official `AuthorKit27.zip` (archive SHA-256 `e28c6ac9bc6eb3b4e2d849547d2cefb5162610ee39d0a12e0dc62d1126b44a7d`). ACL-27 is the fallback; its official site has not yet announced an ARR or commitment deadline.

The required checklist is maintained separately in `ReproducibilityChecklist.tex`, as directed by the AAAI-27 submission instructions. The manuscript includes an AI-use disclosure; AAAI explicitly requires authors to document AI assistance and remain responsible for all content.

Authoritative numerical sources are the generated reports in `../kafa_script/analysis_outputs/`; do not update table values without regenerating and revalidating their raw JSON inputs.
`../kafa_script/TREC_PROTOCOL_CLARIFICATION.md` records that the frozen
`teacher_temperature` CLI field is inert metadata; the verified TREC payloads
use untempered `softmax(-candidate_loss)` probabilities.
