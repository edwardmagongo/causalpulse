# How often a verified interpretation misreads its analysis

Corpus: 25 cases.

| Case | Design | Group | Why |
|---|---|---|---|
| cross-A | cross_section | scenario | cross-sectional scenario A |
| cross-B | cross_section | scenario | cross-sectional scenario B |
| cross-C | cross_section | scenario | cross-sectional scenario C |
| cross-D | cross_section | scenario | cross-sectional scenario D |
| cross-F | cross_section | scenario | cross-sectional scenario F |
| cross-G | cross_section | scenario | cross-sectional scenario G |
| cross-H-complete | cross_section | scenario | scenario H reduced to complete cases; as generated it is refused |
| panel-P1 | panel | scenario | panel scenario P1 |
| panel-P2 | panel | scenario | panel scenario P2 |
| panel-P3 | panel | scenario | panel scenario P3 |
| panel-P4 | panel | scenario | panel scenario P4 |
| panel-P5 | panel | scenario | panel scenario P5 |
| panel-P6 | panel | scenario | panel scenario P6 |
| panel-P7 | panel | scenario | panel scenario P7 |
| donor-S0 | donor_pool | scenario | donor-pool scenario S0 |
| donor-S1 | donor_pool | scenario | donor-pool scenario S1 |
| donor-S2 | donor_pool | scenario | donor-pool scenario S2 |
| donor-S3 | donor_pool | scenario | donor-pool scenario S3 |
| donor-S4 | donor_pool | scenario | donor-pool scenario S4 |
| donor-S5 | donor_pool | scenario | donor-pool scenario S5 |
| real-lalonde | cross_section | real_data | LaLonde (NSW and PSID) |
| real-prop99 | donor_pool | real_data | California Proposition 99 |
| awkward-fail-divergent | cross_section | awkward | strong confounding: a failed check and methods that diverge |
| awkward-divergent | cross_section | awkward | selection on untreated outcomes: methods that diverge |
| awkward-single | cross_section | awkward | one method, so no comparison |

## The run

| Prompt | Interpretations | Verified | First attempt | Refused | No answer | Calls | Cost |
|---|---|---|---|---|---|---|---|
| baseline | 100 | 99 | 84 | 1 | 0 | 116 | $16.50 |
| cite-each-status | 100 | 100 | 91 | 0 | 0 | 109 | $15.75 |

Judge: 199 verdicts, 0 unusable, 199 calls, $9.14.

## Final narratives: baseline

| Category | Per statement | Per narrative | Judge sensitivity | Judge specificity |
|---|---|---|---|---|
| D1 | 0% (at most 0.2%, 0/1997, 25 cases) | 0% (at most 3.0%, 0/99, 25 cases) | - | - |
| D2 | 0.3% (5/1997, SE 0.1%, 25 cases) | 5.1% (5/99, SE 2.1%, 25 cases) | - | - |
| D3 | 0% (at most 0.2%, 0/1997, 25 cases) | 0% (at most 3.0%, 0/99, 25 cases) | - | - |
| D4 | 0.8% (15/1997, SE 0.3%, 25 cases) | 15.2% (15/99, SE 6.1%, 25 cases) | - | - |
| D5 | 0% (at most 0.2%, 0/1997, 25 cases) | 0% (at most 3.0%, 0/99, 25 cases) | - | - |
| J1 | 0% (at most 0.2%, 0/1997, 25 cases) | 0% (at most 3.0%, 0/99, 25 cases) | 0.0% [0.0%, 65.8%], 2 labelled, n_eff 2.0 | 100.0% [96.6%, 100.0%], 148 labelled, n_eff 109.2 |
| J2 | 0.1% (1/1997, SE 0.1%, 25 cases) | 1.0% (1/99, SE 1.0%, 25 cases) | - | 100.0% [96.6%, 100.0%], 150 labelled, n_eff 109.3 |
| J3 | 0.1% (1/1997, SE 0.1%, 25 cases) | 1.0% (1/99, SE 1.0%, 25 cases) | - | 99.9% [96.5%, 100.0%], 150 labelled, n_eff 109.3 |
| J4 | 0% (at most 0.2%, 0/1997, 25 cases) | 0% (at most 3.0%, 0/99, 25 cases) | - | 100.0% [96.6%, 100.0%], 150 labelled, n_eff 109.3 |
| J5 | 0.2% (3/1997, SE 0.1%, 25 cases) | 3.0% (3/99, SE 1.7%, 25 cases) | - | 99.9% [96.5%, 100.0%], 150 labelled, n_eff 109.3 |
| J6 | 0.3% (6/1997, SE 0.1%, 25 cases) | 6.1% (6/99, SE 2.6%, 25 cases) | 1.3% [0.0%, 65.5%], 4 labelled, n_eff 2.1 | 99.7% [96.0%, 100.0%], 146 labelled, n_eff 107.2 |
| any_D | 1.0% (20/1997, SE 0.3%, 25 cases) | 19.2% (19/99, SE 6.0%, 25 cases) | - | - |
| any_J | 0.6% (11/1997, SE 0.2%, 25 cases) | 11.1% (11/99, SE 3.6%, 25 cases) | 5.1% [0.1%, 67.0%], 6 labelled, n_eff 2.2 | 99.6% [95.8%, 100.0%], 144 labelled, n_eff 107.1 |

D4 summaries: 10 true, 0 false, 5 undetermined. Comparisons D1 left to the judge: 8.

## Final narratives: cite-each-status

| Category | Per statement | Per narrative | Judge sensitivity | Judge specificity |
|---|---|---|---|---|
| D1 | 0.1% (2/2010, SE 0.1%, 25 cases) | 2.0% (2/100, SE 1.4%, 25 cases) | - | - |
| D2 | 0% (at most 0.1%, 0/2010, 25 cases) | 0% (at most 3.0%, 0/100, 25 cases) | - | - |
| D3 | 0% (at most 0.1%, 0/2010, 25 cases) | 0% (at most 3.0%, 0/100, 25 cases) | - | - |
| D4 | 0.0% (1/2010, SE 0.0%, 25 cases) | 1.0% (1/100, SE 1.0%, 25 cases) | - | - |
| D5 | 0% (at most 0.1%, 0/2010, 25 cases) | 0% (at most 3.0%, 0/100, 25 cases) | - | - |
| J1 | 0% (at most 0.1%, 0/2010, 25 cases) | 0% (at most 3.0%, 0/100, 25 cases) | 0.0% [0.0%, 65.8%], 2 labelled, n_eff 2.0 | 100.0% [96.6%, 100.0%], 148 labelled, n_eff 109.2 |
| J2 | 0% (at most 0.1%, 0/2010, 25 cases) | 0% (at most 3.0%, 0/100, 25 cases) | - | 100.0% [96.6%, 100.0%], 150 labelled, n_eff 109.3 |
| J3 | 0.1% (2/2010, SE 0.1%, 25 cases) | 2.0% (2/100, SE 1.4%, 25 cases) | - | 99.9% [96.5%, 100.0%], 150 labelled, n_eff 109.3 |
| J4 | 0% (at most 0.1%, 0/2010, 25 cases) | 0% (at most 3.0%, 0/100, 25 cases) | - | 100.0% [96.6%, 100.0%], 150 labelled, n_eff 109.3 |
| J5 | 0% (at most 0.1%, 0/2010, 25 cases) | 0% (at most 3.0%, 0/100, 25 cases) | - | 99.9% [96.5%, 100.0%], 150 labelled, n_eff 109.3 |
| J6 | 0.3% (7/2010, SE 0.1%, 25 cases) | 7.0% (7/100, SE 2.7%, 25 cases) | 1.3% [0.0%, 65.5%], 4 labelled, n_eff 2.1 | 99.7% [96.0%, 100.0%], 146 labelled, n_eff 107.2 |
| any_D | 0.1% (3/2010, SE 0.1%, 25 cases) | 3.0% (3/100, SE 1.7%, 25 cases) | - | - |
| any_J | 0.4% (9/2010, SE 0.2%, 25 cases) | 9.0% (9/100, SE 3.5%, 25 cases) | 5.1% [0.1%, 67.0%], 6 labelled, n_eff 2.2 | 99.6% [95.8%, 100.0%], 144 labelled, n_eff 107.1 |

D4 summaries: 0 true, 0 false, 1 undetermined. Comparisons D1 left to the judge: 8.

## By design (per narrative)

### baseline

| Design | D1 | D2 | D3 | D4 | D5 | J1 | J2 | J3 | J4 | J5 | J6 | any_D | any_J |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cross_section | 0% (at most 6.8%, 0/44, 11 cases) | 9.1% (4/44, SE 3.8%, 11 cases) | 0% (at most 6.8%, 0/44, 11 cases) | 4.5% (2/44, SE 4.5%, 11 cases) | 0% (at most 6.8%, 0/44, 11 cases) | 0% (at most 6.8%, 0/44, 11 cases) | 0% (at most 6.8%, 0/44, 11 cases) | 0% (at most 6.8%, 0/44, 11 cases) | 0% (at most 6.8%, 0/44, 11 cases) | 2.3% (1/44, SE 2.3%, 11 cases) | 0% (at most 6.8%, 0/44, 11 cases) | 13.6% (6/44, SE 5.2%, 11 cases) | 2.3% (1/44, SE 2.3%, 11 cases) |
| donor_pool | 0% (at most 11.1%, 0/27, 7 cases) | 0% (at most 11.1%, 0/27, 7 cases) | 0% (at most 11.1%, 0/27, 7 cases) | 14.8% (4/27, SE 11.6%, 7 cases) | 0% (at most 11.1%, 0/27, 7 cases) | 0% (at most 11.1%, 0/27, 7 cases) | 0% (at most 11.1%, 0/27, 7 cases) | 3.7% (1/27, SE 3.7%, 7 cases) | 0% (at most 11.1%, 0/27, 7 cases) | 0% (at most 11.1%, 0/27, 7 cases) | 18.5% (5/27, SE 7.1%, 7 cases) | 14.8% (4/27, SE 11.6%, 7 cases) | 22.2% (6/27, SE 8.5%, 7 cases) |
| panel | 0% (at most 10.7%, 0/28, 7 cases) | 3.6% (1/28, SE 3.6%, 7 cases) | 0% (at most 10.7%, 0/28, 7 cases) | 32.1% (9/28, SE 16.1%, 7 cases) | 0% (at most 10.7%, 0/28, 7 cases) | 0% (at most 10.7%, 0/28, 7 cases) | 3.6% (1/28, SE 3.6%, 7 cases) | 0% (at most 10.7%, 0/28, 7 cases) | 0% (at most 10.7%, 0/28, 7 cases) | 7.1% (2/28, SE 4.6%, 7 cases) | 3.6% (1/28, SE 3.6%, 7 cases) | 32.1% (9/28, SE 16.1%, 7 cases) | 14.3% (4/28, SE 7.4%, 7 cases) |

### cite-each-status

| Design | D1 | D2 | D3 | D4 | D5 | J1 | J2 | J3 | J4 | J5 | J6 | any_D | any_J |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cross_section | 2.3% (1/44, SE 2.3%, 11 cases) | 0% (at most 6.8%, 0/44, 11 cases) | 0% (at most 6.8%, 0/44, 11 cases) | 2.3% (1/44, SE 2.3%, 11 cases) | 0% (at most 6.8%, 0/44, 11 cases) | 0% (at most 6.8%, 0/44, 11 cases) | 0% (at most 6.8%, 0/44, 11 cases) | 0% (at most 6.8%, 0/44, 11 cases) | 0% (at most 6.8%, 0/44, 11 cases) | 0% (at most 6.8%, 0/44, 11 cases) | 6.8% (3/44, SE 3.5%, 11 cases) | 4.5% (2/44, SE 3.0%, 11 cases) | 6.8% (3/44, SE 3.5%, 11 cases) |
| donor_pool | 3.6% (1/28, SE 3.6%, 7 cases) | 0% (at most 10.7%, 0/28, 7 cases) | 0% (at most 10.7%, 0/28, 7 cases) | 0% (at most 10.7%, 0/28, 7 cases) | 0% (at most 10.7%, 0/28, 7 cases) | 0% (at most 10.7%, 0/28, 7 cases) | 0% (at most 10.7%, 0/28, 7 cases) | 7.1% (2/28, SE 4.6%, 7 cases) | 0% (at most 10.7%, 0/28, 7 cases) | 0% (at most 10.7%, 0/28, 7 cases) | 10.7% (3/28, SE 7.4%, 7 cases) | 3.6% (1/28, SE 3.6%, 7 cases) | 17.9% (5/28, SE 10.5%, 7 cases) |
| panel | 0% (at most 10.7%, 0/28, 7 cases) | 0% (at most 10.7%, 0/28, 7 cases) | 0% (at most 10.7%, 0/28, 7 cases) | 0% (at most 10.7%, 0/28, 7 cases) | 0% (at most 10.7%, 0/28, 7 cases) | 0% (at most 10.7%, 0/28, 7 cases) | 0% (at most 10.7%, 0/28, 7 cases) | 0% (at most 10.7%, 0/28, 7 cases) | 0% (at most 10.7%, 0/28, 7 cases) | 0% (at most 10.7%, 0/28, 7 cases) | 3.6% (1/28, SE 3.6%, 7 cases) | 0% (at most 10.7%, 0/28, 7 cases) | 3.6% (1/28, SE 3.6%, 7 cases) |

## First attempts (per narrative)

| Prompt | D1 | D2 | D3 | D4 | D5 |
|---|---|---|---|---|---|
| baseline | 0% (at most 3.0%, 0/100, 25 cases) | 5.0% (5/100, SE 2.0%, 25 cases) | 0% (at most 3.0%, 0/100, 25 cases) | 18.0% (18/100, SE 6.8%, 25 cases) | 0% (at most 3.0%, 0/100, 25 cases) |
| cite-each-status | 2.0% (2/100, SE 1.4%, 25 cases) | 0% (at most 3.0%, 0/100, 25 cases) | 0% (at most 3.0%, 0/100, 25 cases) | 4.0% (4/100, SE 2.4%, 25 cases) | 0% (at most 3.0%, 0/100, 25 cases) |

Narratives with no first attempt, excluded from these figures: baseline 0, cite-each-status 0.

## The graders against the labels

150 labelled statements. The judge's sensitivity and specificity and each grader's precision are weighted by stratum (judge-flagged, grader-only, unflagged) back to all judged statements, with Wilson intervals at the effective sample size n_eff; they are noisy, because each unflagged label stands for many statements.

| Grader | Precision |
|---|---|
| D1 | 0.0% [0.0%, 65.8%], 2 labelled, n_eff 2.0 |
| D2 | 0.0% [0.0%, 43.4%], 5 labelled, n_eff 5.0 |
| D3 | - |
| D4 | 0.0% [0.0%, 19.4%], 16 labelled, n_eff 16.0 |
| D5 | - |

## Comparison: cite-each-status minus baseline

Primary endpoint, fixed before any paid call (D4 per narrative, first attempts): -14.0 points [-26.0, -2.0], 25 cases. Normal 95% interval. Paired by case over 25 of 25 shared cases.

Exploratory, final narratives, per narrative:

| Category | Difference | Paired |
|---|---|---|
| D1 | +2.0 points [-0.7, +4.7], 25 cases | 25 of 25 shared cases |
| D2 | -5.0 points [-9.0, -1.0], 25 cases | 25 of 25 shared cases |
| D3 | +0.0 points (no spread across 25 cases; no interval) | 25 of 25 shared cases |
| D4 | -15.0 points [-28.0, -2.0], 25 cases | 25 of 25 shared cases |
| D5 | +0.0 points (no spread across 25 cases; no interval) | 25 of 25 shared cases |
| J1 | +0.0 points (no spread across 25 cases; no interval) | 25 of 25 shared cases |
| J2 | -1.0 points [-3.0, +1.0], 25 cases | 25 of 25 shared cases |
| J3 | +1.0 points [-2.4, +4.4], 25 cases | 25 of 25 shared cases |
| J4 | +0.0 points (no spread across 25 cases; no interval) | 25 of 25 shared cases |
| J5 | -3.0 points [-6.3, +0.3], 25 cases | 25 of 25 shared cases |
| J6 | +1.0 points [-5.0, +7.0], 25 cases | 25 of 25 shared cases |
| any_D | -17.0 points [-30.2, -3.8], 25 cases | 25 of 25 shared cases |
| any_J | -2.0 points [-10.0, +6.0], 25 cases | 25 of 25 shared cases |
