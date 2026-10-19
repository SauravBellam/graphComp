# Tool tracing: what each tool call was for, and whether it reached or revealed the gold code

252 trajectories (92 resolved), 6766 tool calls. Heuristic text matching against the gold patch; association, not causation.

## Per purpose: resolved vs unresolved trajectories

| purpose | calls (res / unres) | on gold code (res / unres) | reveals gold code first (res / unres) | output followed within 3 calls (res / unres) |
|---|---|---|---|---|
| read-src | 576 / 1720 | 72% / 64% | 12% / 12% | 10% / 7% |
| search | 414 / 1337 | 50% / 45% | 9% / 7% | 23% / 13% |
| edit-src | 183 / 670 | 97% / 78% | 1% / 2% | 28% / 16% |
| reproduce | 305 / 318 | 3% / 5% | 2% / 2% | 10% / 13% |
| edit-test-or-script | 265 / 244 | 0% / 0% | 0% / 0% | 53% / 52% |
| submit | 129 / 122 | 0% / 0% | 0% / 0% | 0% / 0% |
| command-other | 68 / 44 | 0% / 2% | 0% / 0% | 0% / 0% |
| read-test | 45 / 58 | 0% / 0% | 2% / 0% | 0% / 5% |
| test | 39 / 48 | 0% / 2% | 0% / 0% | 15% / 4% |
| inspect | 9 / 73 | 89% / 78% | 0% / 15% | 0% / 0% |
| vcs | 23 / 16 | 4% / 19% | 0% / 0% | 9% / 12% |
| status | 17 / 22 | 0% / 0% | 0% / 0% | 0% / 0% |
| graph:get_code_neighbors | 3 / 9 | 33% / 33% | 0% / 0% | 0% / 0% |
| graph:search_similar_code | 1 / 7 | 100% / 14% | 0% / 0% | 0% / 0% |
| install | 0 / 1 | - / 0% | - / 0% | - / 0% |

## Who first reaches the gold code (first call whose target or new output names it)

| first purpose | resolved runs | unresolved runs |
|---|---|---|
| read-src | 45 | 79 |
| search | 43 | 70 |
| never | 0 | 9 |
| reproduce | 4 | 1 |
| edit-src | 0 | 1 |

resolved: median calls before the gold code is reached 0; edited a gold file 92/92; never reached it 0/92

unresolved: median calls before the gold code is reached 0; edited a gold file 107/160; never reached it 9/160
issue text (problem + hints) names a gold file: 22/252 trajectories; resolved 12/22 when it does vs 80/230 when not
issue text (problem + hints) names a gold function: 58/252 trajectories; resolved 22/58 when it does vs 70/194 when not

## Shortest resolved routes (the best cases)

- `fastapi_14492` (7 calls): search → read-src → edit-src → search → read-src → search → submit
- `fastapi_14786` (9 calls): read-src → edit-test-or-script → reproduce → edit-src → reproduce → search → test → command-other → submit
- `fastapi_14492` (10 calls): search → read-src → edit-src → search → read-src → search×3 → read-src → submit
- `fastapi_14492` (10 calls): search×3 → read-src×2 → search → edit-src → search → read-src → submit
- `rich_3882` (10 calls): read-src×2 → edit-test-or-script → reproduce → edit-src → edit-test-or-script×2 → reproduce → command-other → submit
- `rich_3676` (11 calls): read-src×4 → edit-src×3 → edit-test-or-script → reproduce → command-other → submit
- `rich_3905` (12 calls): read-src → search → read-src×2 → edit-test-or-script → reproduce → edit-src → reproduce → edit-test-or-script → reproduce → command-other → submit
- `fastapi_14458` (12 calls): search → read-src×2 → edit-test-or-script → reproduce → edit-src×3 → reproduce → edit-test-or-script → reproduce → submit
- `fastapi_14297` (12 calls): edit-test-or-script → reproduce → read-src×3 → edit-src → reproduce → graph:get_code_neighbors → read-src → search → read-src → submit
- `fastapi_14360` (12 calls): search×2 → read-src → edit-test-or-script → reproduce → edit-src → reproduce → edit-test-or-script → reproduce → submit → command-other → submit

## Purpose bigrams: share of resolved vs unresolved trajectories containing them (top 15 by difference)

| bigram | in resolved | in unresolved |
|---|---|---|
| command-other → submit | 65% | 19% |
| edit-src → reproduce | 65% | 27% |
| edit-test-or-script → reproduce | 84% | 48% |
| reproduce → edit-src | 51% | 19% |
| submit → command-other | 37% | 6% |
| search → search | 46% | 69% |
| edit-src → read-src | 10% | 29% |
| reproduce → submit | 34% | 16% |
| reproduce → read-src | 47% | 30% |
| read-src → edit-test-or-script | 51% | 34% |
| reproduce → edit-test-or-script | 40% | 26% |
| edit-src → submit | 1% | 16% |
| read-test → edit-test-or-script | 16% | 2% |
| edit-test-or-script → edit-test-or-script | 22% | 8% |
| read-src → search | 77% | 90% |
