# IEEE-format manuscript

Two builds of the same paper.

| File | What it is |
|---|---|
| `Chaudhari_FVIII_Inhibitor_IEEE.pdf` | Rendered PDF, 6 pages, produced from `paper_ieee.html` |
| `paper_ieee.html` | IEEEtran conference geometry reproduced in CSS |
| `paper_ieee.tex` | IEEEtran LaTeX source — compile this for a submission-grade PDF |

## Which one to submit

**Submit the LaTeX build.** No LaTeX toolchain exists on the machine this was
written on, so the committed PDF was rendered through headless Edge from the
HTML. It matches IEEEtran geometry closely — US Letter, 0.75in/1.0in/0.625in
margins, two 3.5in columns with a 0.25in gutter, 10pt Times — but it is a
reproduction of the template, not the template itself. Conferences that check
formatting programmatically (IEEE PDF eXpress) will want the real thing.

Upload `paper_ieee.tex` to Overleaf, pick any IEEE conference template so
`IEEEtran.cls` resolves, and compile. Locally two passes are needed for the
table references:

```bash
pdflatex paper_ieee && pdflatex paper_ieee
```

## Length

**6 pages.** The body always fitted; the seventh page in the first build held
only the trailing footnote block, which in IEEE style belongs on page 1. It is
now a page-1 footnote (`\thanks` in the LaTeX source), so no argument was cut
to reach the limit.

## Before submitting

Three citations are deliberately incomplete and marked in both sources:

- `singh` — the 97.37% reference. Complete from the stage-1 bibliography in
  `../../RESULTS.md`.
- `eahad` — the source article for the dataset itself. Needs the author list,
  volume, pages and DOI.
- `gouw` — verify the page range.

They are described rather than reconstructed from memory so that no incorrect
citation is propagated.

## Regenerating the PDF

```bash
"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --headless --disable-gpu --no-pdf-header-footer --print-to-pdf=paper/ieee/Chaudhari_FVIII_Inhibitor_IEEE.pdf file:///ABSOLUTE/PATH/paper/ieee/paper_ieee.html
```

The HTML disables ligatures deliberately (`font-variant-ligatures: none`).
Times emits U+FB01 for "fi", which extracts as a corrupt glyph and breaks text
search and plagiarism-check tooling. Verified after each render: 6 pages, zero
U+FB00–FB06 characters, zero U+FFFD, and all eight numbered headings
monotonic document order.
