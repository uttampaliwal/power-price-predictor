# arXiv Submission Guide

Everything needed to submit is in this folder. `iex_dam_forecasting_arxiv_v1.tar.gz` is the ready-to-upload source package (verified to compile: pdflatex + bibtex, TeX Live 2026, 11 pages).

## Files

| File | Purpose |
|------|---------|
| `main.tex` | Full paper source |
| `main.bbl` | Compiled bibliography — **must be uploaded** (arXiv does not run bibtex) |
| `references.bib` | BibTeX database (keep in sync if you revise) |
| `figures/` | Both PNG figures referenced by main.tex |
| `main.pdf` | Compiled preview (do NOT upload; arXiv builds its own PDF) |
| `iex_dam_forecasting_arxiv_v1.tar.gz` | Upload this single file |

## Step-by-step submission

1. Go to <https://arxiv.org/submit> → **Start a new submission** (you need an account; register with the same email you use for scholarly identity).
2. **Subject category**: choose as below.
3. **License**: "arXiv.org perpetual, non-exclusive license" is the default and fine for a preprint.
4. **Upload files**: upload only the tarball (or the individual files: main.tex, main.bbl, references.bib, both PNGs). Do not upload main.pdf.
5. **Approve the compiled PDF preview** that arXiv generates.
6. Your submission then enters moderation (typically 1–5 business days).

## Metadata to paste into the submission form

**Title**
```
Day-Ahead Electricity Price Forecasting in the Indian Energy Exchange: Gradient-Boosted Ensembles at 15-Minute Resolution
```

**Authors** (add yourself via the author form, not just in LaTeX)
```
Uttam Paliwal
```

**Primary category**
```
q-fin.TR  (Trading and Market Microstructure)
```
**Cross-list**
```
cs.LG     (Machine Learning)
```

**Abstract** — copy from `main.tex` (the `\begin{abstract}...\end{abstract}` block).

**Comments field** (suggested)
```
11 pages, 2 figures, 3 tables. Code: https://github.com/uttampaliwal/power-price-predictor
```

**Related DOI / journal ref**: leave empty (preprint).

## Endorsement note (first-time submitters)

- Posting in **q-fin.TR** requires an endorsement *only if* you have never submitted to any arXiv category before. Check your status at <https://arxiv.org/auth/endorse>.
- If endorsement is requested: ask any arXiv author in quantitative finance / energy economics (endorsers can be found via any recent q-fin.TR paper's author list; the submission page shows eligible endorsers near you).
- Practical shortcut: cs.LG cross-listing still needs the primary category approved first, so complete q-fin.TR endorsement once and both are covered.

## Before you submit — checklist

- [ ] Decide whether to add an affiliation line (currently name + email only, per your choice). Papers are immutable after posting; affiliation cannot be added later without a version bump.
- [ ] ORCID: link it in your arXiv user profile (increases discoverability/citations).
- [ ] Read the abstract once more for tone — arXiv moderators occasionally bounce overly promotional wording (current text is neutral/factual).
- [ ] If you plan to later submit this to a workshop/conference, check its preprint policy (most ML venues allow prior arXiv posting).

## After acceptance

Add the arXiv ID to:
1. GitHub repo README (`paper/` section): "Preprint: arXiv:XXXX.XXXXX"
2. Your resume/CV with the identifier — this is the citable artifact: `Uttam Paliwal. Day-Ahead Electricity Price Forecasting in the Indian Energy Exchange: Gradient-Boosted Ensembles at 15-Minute Resolution. arXiv preprint arXiv:XXXX.XXXXX, 2026.`

BibTeX entry for others to cite:
```bibtex
@misc{paliwal2026dayahead,
  title={Day-Ahead Electricity Price Forecasting in the {I}ndian {E}nergy {E}xchange: Gradient-Boosted Ensembles at 15-Minute Resolution},
  author={Paliwal, Uttam},
  year={2026},
  eprint={XXXX.XXXXX},
  archivePrefix={arXiv},
  primaryClass={q-fin.TR}
}
```
