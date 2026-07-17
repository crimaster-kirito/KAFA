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
compiles with the official AAAI-27 anonymous style to six pages, leaving room
for the verified IMDb result block.  Its current main-paper visual core is one
evidence-aware four-arm framework plus an evidence-ledger table; the three-page
supplement retains the legacy experimental figure and four detailed tables.
The submission jointly resolves 29 references after the direct-neighbor
literature audit.  The build
checks the complete AG News and TREC server matrices, GSM8K and natural-label
evidence boundaries, headline numbers, and the focused research tests. Build
dimensions and diagnostics are checked after every result update rather than
treated as fixed prose. The PDF uses embedded Type 1 fonts and contains no
hyperlinks or bookmarks. The separately compiled checklist is two pages; its
only diagnostics are harmless underfull-box notices inherited from the
official question layout.

CCF-A evidence strengthening is currently active. The original IMDb
development matrix and its matched-K control addendum validate completely, and
the prospective official matrix is running.  The hash-locked matched matrix is
wrapped by a separate fail-closed resource-aware coordinator: random-K and
maximum-purity-K may run two-way, while routed-K remains exclusive.  The
submission build conditionally verifies and summarizes complete new IMDb
matrices and invokes a fail-closed final asset generator only when all 70
records exist.  A separate cross-task generator remains closed until the full
40 TREC + 40 IMDb + 30 matched-\(K\) + 40 AG input set verifies; its
current-input manifest binds all 150 records and published asset hashes.  This
is not yet a release gate: IMDb and AG must finish, the generated assets must
be inserted, and the final page boundary must be rechecked.  The expanded
focused research suite passes 134 tests, including
the resource-aware schedulers and fail-closed IMDb asset generator.
legacy repository-wide optional-dependency failures are tracked separately in
`../RESEARCH_AUDIT.md`.

`supplement.tex` compiles independently to a three-page anonymous
`supplement.pdf`. It retains the full server configuration, legacy server
table, detailed Co-Teaching, Hausa/Yoruba, and GSM8K tables, and the original
two-panel evidence figure, together with an artifact-reconstruction and
fail-closed failure-policy section, allowing the final main
paper to concentrate its limited content pages on the server-transfer audit.
The supplement currently compiles without warnings or layout overflow.

AAAI-27 is the time-sensitive stretch target: its official main-track call limits submissions to seven content pages and lists abstract and paper deadlines of 2026-07-21 and 2026-07-28 (AoE), respectively. `aaai2027.sty` and `aaai2027.bst` were extracted unchanged from the official `AuthorKit27.zip` (archive SHA-256 `e28c6ac9bc6eb3b4e2d849547d2cefb5162610ee39d0a12e0dc62d1126b44a7d`). ACL-27 is the fallback; its official site has not yet announced an ARR or commitment deadline.

The required checklist is maintained separately in `ReproducibilityChecklist.tex`, as directed by the AAAI-27 submission instructions. The manuscript includes an AI-use disclosure; AAAI explicitly requires authors to document AI assistance and remain responsible for all content.

Authoritative numerical sources are the generated reports in `../kafa_script/analysis_outputs/`; do not update table values without regenerating and revalidating their raw JSON inputs.
`../kafa_script/TREC_PROTOCOL_CLARIFICATION.md` records that the frozen
`teacher_temperature` CLI field is inert metadata; the verified TREC payloads
use untempered `softmax(-candidate_loss)` probabilities.
