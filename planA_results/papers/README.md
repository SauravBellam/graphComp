# Papers

| Paper | Draft (Markdown) | Final (LaTeX) | PDF |
| --- | --- | --- | --- |
| 1. Is the Map Right? An execution-grounded audit of the code graph and embeddings handed to local SWE agents | `paper1_is_the_map_right.md` | `latex/paper1_is_the_map_right.tex` | `latex/paper1_is_the_map_right.pdf` |
| 2. Does the Test Tell a Fix From No Fix? Discriminating-test scoring for evaluating and training local coding agents | `paper2_discriminating_tests.md` | `latex/paper2_discriminating_tests.tex` | `latex/paper2_discriminating_tests.pdf` |

Showdown Markdown version of paper 1 (same text as the LaTeX, with marked image slots): `paper1_is_the_map_right.showdown.md`.
PNG figures for inserting into it: `figures/png/*.png`.

For judges: `FOR_JUDGES.md` (claim-to-file map, reproduction), `verify_paper1_numbers.py` (checks 66 claims against the result files; output in `verify_paper1_numbers.out.txt`), `supplement_paper1.zip` (scripts, results and paper files in one 7 MB bundle).

Figures: `figures/*.pdf`, drawn by `make_figures.py` from the result files under `planA_results/`
(run from `planA_results/`: `python papers/make_figures.py`); `figures/figure_data.json` holds the
numbers each figure plots.

Build: `cd latex && pdflatex paper1_is_the_map_right.tex && pdflatex paper1_is_the_map_right.tex`
(same for paper 2; two passes resolve cross-references). Needs TeX Live with `booktabs`,
`placeins`, `enumitem`, `microtype`, `lmodern`.

Word counts (body, including captions and tables, excluding references): paper 1 2,896; paper 2 about 1,600.
