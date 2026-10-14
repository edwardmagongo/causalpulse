# LaLonde / NSW / CPS-1

**Source**: the `causaldata` Python package (MIT licence, Nick Huntington-Klein,
https://github.com/NickCH-K/causaldata), whose `nsw_mixtape` and `cps_mixtape` datasets are the
Dehejia & Wahba (1999) re-analysis of LaLonde (1986)'s National Supported Work (NSW) experiment,
originally distributed from Rajeev Dehejia's NBER-hosted replication page. Fetched and converted to
CSV on 2026-09-04; the originals are Stata `.dta`, read with `pandas.read_stata` and written back
out unchanged in content, with one added `unit_id` column and the two files concatenated.

**Rows**: 16,437. `data_id == "Dehejia-Wahba Sample"` (445 rows: `treat == 1` for 185 NSW
experimental treated units, `treat == 0` for 260 NSW experimental control units) plus
`data_id == "CPS1"` (15,992 rows, `treat == 0` throughout — the CPS-1 observational comparison
group). No missing values anywhere in the file.

**Columns**: `unit_id` (added, a row index), `data_id`, `treat` (0/1), `age`, `educ` (years of
schooling), `black`, `hisp`, `marr`, `nodegree` (indicator, no high-school diploma), `re74`, `re75`,
`re78` (real earnings in 1974/1975/1978 dollars).

**Treatment**: `treat`. **Outcome**: `re78`. **Covariates**: `age, educ, black, hisp, marr, nodegree,
re74, re75`.

**The intended causal question**: does the NSW job-training program raise 1978 earnings? The NSW
experimental contrast (`data_id == "Dehejia-Wahba Sample"`, treated vs control) gives the known
right answer — random assignment settles it directly, and it measures **+$1,794.34** here. The
**observational** contrast (NSW treated vs the CPS-1 comparison group) is the classic hard test:
CPS-1 was not randomly assigned to the program, so a naive comparison is confounded, and it measures
**-$8,497.52** — wrong in sign and off by more than $10,000. This is the textbook case
propensity-score and doubly-robust methods are built to handle. Scoring an observational estimator's
NSW-treated-vs-CPS-1 estimate against the NSW-treated-vs-NSW-control experimental benchmark is
exactly Dehejia & Wahba's own validation exercise.

**Where the confounding comes from**, measured rather than asserted: the two groups differ sharply
at baseline. Mean `re74` is $2,102 in NSW against $14,017 in CPS-1, and mean `re75` is $1,377
against $13,651. NSW participants are younger (25.4 against 33.2), less educated (10.2 years against
12.0), far more likely to be Black (0.834 against 0.074) and far less likely to be married (0.169
against 0.712). CPS-1 is a general population sample; NSW enrolled the long-term unemployed.

**Known limitations**:

- This project uses only the CPS-1 comparison group, not the PSID variants — narrower than the
  original "PSID, CPS" phrasing, and a deliberate scoping choice. CPS-1 alone exercises
  the same selection-bias problem, and adding PSID would not test anything qualitatively different.
- `re74` and `re75` are **not** uniformly zero in the NSW sample, contrary to a common
  simplification about the experiment's eligibility criteria. Measured here: `re74` is zero for
  73.3% of NSW rows and `re75` for 64.9%, with non-zero means of $2,102 and $1,377. Anything reading
  those columns as a constant would be reading them wrong.
- No causal claims beyond what LaLonde (1986) and Dehejia & Wahba (1999) already established are
  made from this fixture.
