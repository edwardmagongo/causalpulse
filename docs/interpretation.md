# Interpreting an analysis

## Interpreting an analysis

`causalpulse interpret <run-dir>` writes a plain-language reading of a whole analysis: what each
method estimated, each validity band and the checks behind it, what the cross-method comparison
reports, and which fits failed. It may also say which estimators the router excluded; that is not
required of it. It reads the `facts.json` that `analyze --report`
writes, so it refits nothing and can be repeated. Every `analyze` run removes a `facts.json` and
any interpretation left in its output directory by an earlier run, because they describe that
earlier analysis. With `--report` the fact table is written last: if it cannot be built, the
command prints one line, exits 1, and the estimates and the manifest are already written.

```bash
uv run causalpulse analyze --data tests/fixtures/real_data/lalonde.csv \
  --unit unit_id --treatment treat --outcome re78 \
  --covariates age,educ,black,hisp,marr,nodegree,re74,re75 --outcome-type continuous \
  --report --estimators difference_in_means,lin_regression --out out/lalonde
uv run causalpulse interpret out/lalonde
```

It needs `ANTHROPIC_API_KEY` (or an `ant auth login` profile) and makes one model call, or two
when the first reply is rejected. Measured on real output (2026-10-04), one interpretation costs
$0.07 to $0.20; the calibration record is in `docs/reports/2026-10-04-llm-calibration/`. It writes `interpretation.md`, the narrative, and
`interpretation.json`, the record: the narrative as the model wrote it, the rendered text, the
model id, the effort and the token limit it ran with, `prompt_version` (a short hash of the
system prompt and the reply schema), every attempt with its violations and token usage, and the
SHA-256 of the bytes of the `facts.json` it read.
`--model` and `--effort` override the defaults (`claude-opus-5-5`, `medium`).

### The model never writes a value

`analyze --report` flattens the analysis into a table of facts, each with an id:
`est.aipw.point`, `score.aipw.band`, `check.aipw.positivity.status`,
`cmp.ate.percentage_point.agreement`. The model is given that table and no other data from the
analysis, and writes statements in which every value is a placeholder:

```
{est.aipw.method} estimates the {est.aipw.estimand} at {est.aipw.point}, with a
{est.aipw.ci_level} interval from {est.aipw.ci_lower} to {est.aipw.ci_upper}.
```

The system substitutes the values: numbers to four significant figures, p-values to three, counts
exactly, levels as a percentage (`95%`). The model cannot write a digit, a percent sign or one of
a short list of number words, and a placeholder must stand apart from the text around it, with no
sign or operator before a number, so a figure is shown exactly as the analysis produced it. A
quantity spelled in words outside that list ("twenty units", "a third of the effect") is not caught. A value that was never
computed, such as the statistic of a check that cannot be tested, has no fact, so citing it fails instead of
rendering as a zero.

Before anything is substituted, a deterministic verifier applies ten rules, and the narrative
is written only if none is broken:

1. **Resolves.** Every placeholder names a fact in the table.
2. **No bare values.** No digit, no percent sign, and no number word from a short list, outside a
   placeholder.
3. **Controlled vocabulary.** A word that states a band, a status, an agreement verdict, a
   direction, a zero result, significance or eligibility appears only through a placeholder.
   "Strong evidence" is refused unless the word comes from a band fact; "significant" is refused
   outright, and the model has an `interval_excludes_zero` fact to cite. The lists hold exact word
   forms, so an unlisted form ("strongest", "failure") passes.
4. **Attribution.** A statement citing a fact about a method must also name that method.
5. **No bare point.** A statement citing a point estimate must cite both interval bounds.
6. **An INSUFFICIENT estimate carries its band.** A statement citing the point, an interval
   bound, the standard error or the p-value of an estimate whose validity band is `INSUFFICIENT`
   must also cite that band, so the word stands beside the number. A band cited in another
   statement does not count. Added after the evaluation's labels found two statements that
   quoted such an estimate with no caveat; a `WEAK` band needs none.
7. **Language.** The prose passes `causal_language_lint`, the lint the evidence report already
   passes. The lint reads each statement with its numbers masked, so that the full stop inside a
   decimal does not end a sentence.
8. **Delimiters.** Every placeholder stands apart from the text around it, and no sign or
   operator comes before a number. `-{point}` and `{n_treated}{n_control}` are refused: glued
   together they would render a figure the table does not hold.
9. **One line.** A statement holds no line break, control or invisible character, so it cannot
   write a heading of its own.
10. **Completeness.** The narrative must cite every estimate's point and band, every comparison's
   agreement (or its reason for not comparing), every fit failure, and every check whose status is
   `FAIL` or `WARN`.

A rejected reply is sent back to the model once, with its violations. If the second reply is
rejected too, the command prints the violations, writes nothing, and exits 1. The validity
score's limits statement is appended by the system and is not the model's to write.

A `facts.json` that is missing, damaged or from another schema version is reported in one line,
naming the file and what to rerun, and nothing is sent to the model.

The command also prints one line and exits 1, writing nothing, when no credentials are found (it
says how to supply them), when the login profile is broken or expired, when the API call fails,
and when the model refuses, or its reply is cut off at the token limit or does not fit the reply
schema. The command does not retry any of these;
the SDK itself retries rate limits and server errors. A failed run leaves an `interpretation.md`
or `interpretation.json` from an earlier run in place.

### What a verified interpretation means, and what it does not

It means every value shown is the system's own, every controlled word is backed by a fact,
nothing the completeness rule requires was left out, and the prose passes the language lint.

It does not mean the narrative is a correct reading of the analysis. The verifier does not catch:

- the words that name a quantity: "a p-value of `{est.ipw.se}`", or a lower and an upper bound
  swapped, verify. The value shown is the system's; what the sentence calls it is the model's;
- qualitative wording outside the controlled lists ("a sizeable effect");
- a wrong relationship between two correctly cited facts ("A is larger than B");
- words around a quoted check interpretation that frame it misleadingly;
- a quantity expressed without a number ("most checks");
- a number spelled in words beyond the short list ("twenty units");
- a value attributed to the wrong method inside a statement that names both;
- a statement of two sentences that names a method in one and misplaces its value in the other:
  the prompt asks for one sentence per statement, and that is not checked;
- emphasis: a failed check must be cited, not given weight;
- the negation of a cited fact ("the band is not {score.aipw.band}");
- a hedge or negation word that arrives inside a cited fact ("not eligible", or any check's
  interpretation), which the lint counts as hedging the whole sentence;
- the phrase "consistent with", which is allowed as free text because the lint itself asks for
  it, so "consistent with each other" passes whatever the comparison found, and "consistent with
  no effect" passes whatever the interval says.

How often these happen is not measured yet. That is the next phase of this sub-project.

### What leaves the machine

Of the analysis, the fact table and nothing else: aggregate statistics, method and check names,
and prose the system wrote. Beside it go the fixed instructions and, on a retry, the model's own
rejected reply with the violations. No dataset rows are sent. A fit failure's error text is the
exception's own message, which can name a column or quote a cell value (this codebase's fits
raise `could not convert string to float: '<the cell>'` for a non-numeric covariate). It is sent to
the model and, because the narrative must cite every fit failure, shown to the reader verbatim.

Text inside a fact is treated as untrusted, and the defence is partial. A fit error written to
read as an instruction reaches the model. Whatever the model then writes must still pass the
verifier, so it cannot bring in a figure or a controlled word. Wording that avoids the controlled
lists can still reach the narrative as the model's own free text.

### Testing it

`make check` never calls the model: every test but two runs against a scripted model or a mock of
the SDK's transport. `make check-llm` runs those two against the real API: this interpretation, and
`cp-llm-eval`'s judge (below).

### On the server and in the dashboard (phase 3)

A server can offer the same interpretation for an analysis run through the API. It is off unless
`CP_INTERPRETATION=on` is set for both the API and the worker. A worker with it off fails any
interpretation it is handed, without a model call, even if the API accepted the request. When it is
on, the worker also needs `ANTHROPIC_API_KEY`, and refuses to start without it. The API never reads
the key.

- **The fact table is stored when the analysis runs.** The worker writes it beside the outcome. This
  calls no model and costs nothing. If the extractor cannot build it, the analysis still succeeds,
  and it simply cannot be interpreted. Analyses run before this existed have no fact table and are
  not refitted to get one.
- **An interpretation is written only when someone asks.** `POST /v1/analyses/{id}/interpretation`
  answers 202 and queues a separate job, so an analysis never waits on the model.
  `CP_LIMIT_INTERPRETATIONS_PER_DAY` caps the requests per user, counted when accepted. The
  production stack defaults to 5.
- **A result is final under the prompt and rules it was written under.** A verified or refused
  interpretation cannot be asked for again while the server's prompt and verifier rules are the
  ones it was written under: re-asking until something passes would cost money and lower the bar.
  A failed one can be retried, such as a model that could not be reached, a job that ran out of
  time, or a worker that was lost. Once the prompt or the set of rule names changes, a verified or
  refused one can be replaced as well (see
  [Interpreting again](#interpreting-again-and-the-earlier-interpretations)).
- **What the reader sees.** On the analysis page, a verified interpretation is shown under
  "Interpretation (model-written, verified against the facts)". Beside it are the system's limits
  text, a fixed note on what "verified" establishes and what it does not, and the model and prompt
  version. A refused one is not shown; the page says so and gives the number of attempts.
- **A verified interpretation can still misread its analysis.** Phase 2's study, below, measured
  it: in a hand-labelled sample, the commonest misreading quoted an estimate whose band is
  `INSUFFICIENT` without saying so. The verifier now refuses that (rule 6, below); misreadings
  outside its rules can still pass.

### Interpreting again, and the earlier interpretations

**The interpreter version.** Each interpretation the server writes records the version of the
interpreter it was written under: the first 12 hexadecimal characters of a SHA-256 hash over the
prompt version (itself a hash of the system prompt and the reply schema) and the names of the
verifier's rules, sorted and joined by commas, the two parts separated by `|`. Two
interpretations with the same version were written under the same system prompt and reply schema
and checked against rules of the same names. The model and the effort are recorded beside it on
the row, not in it. The worker works the version out before it calls the model and stores it with
the outcome, whether verified, refused or failed.
A row still queued or running has none yet, nor has one that the API marked failed because the job
queue could not be reached or the worker was lost.

**Out of date.** An interpretation is out of date when its version is not the server's current
one. One with no recorded version counts as out of date: migration `0004` leaves every
interpretation written before it without one, since what they were written under was not
recorded. The version has a limit: a rule whose behaviour changes while its name and the prompt
stay the same does not change it, so it does not make an interpretation out of date.
`tests/llm/test_version.py` pins the rule names and the version, so a rule added, removed or
renamed fails that test and has to be accepted on purpose; it cannot notice a rule whose behaviour
changed under the same name.

**When a new interpretation can be asked for.** For an analysis that has succeeded and has a fact
table, `POST /v1/analyses/{id}/interpretation` accepts a request when the analysis has no
interpretation yet, when its newest one failed, under any version, or when its newest one is
verified or refused and out of date. It answers 409 when the newest one is:

- queued or running: "An interpretation of this analysis is already being written." (If it was
  asked for at least 30 seconds earlier and its job is gone from the queue or ended without
  recording a result, it is marked failed first, and the request goes ahead.)
- verified under the current version: "This analysis already has an interpretation written under
  the current prompt and rules."
- refused under the current version: "This analysis's interpretation could not be verified under
  the current prompt and rules, and asking again would not change them."

Every accepted request writes a new row and counts against `CP_LIMIT_INTERPRETATIONS_PER_DAY`,
whether it is the first, a retry of a failed one, or a replacement for one that is out of date.
No row is reset or overwritten: the earlier ones are kept as they were. The newest, by when it was
asked for, is the analysis's interpretation. `GET /v1/analyses/{id}` returns it as
`interpretation` and the others, newest first, as `earlier_interpretations`; each carries its
`interpreter_version` and `out_of_date`, and `interpretable` says whether a request would be
accepted, the day's limit aside.

**What the dashboard shows.** A verified or refused interpretation that is out of date carries the
line "Written under an earlier version of the prompt and rules." and, when a request would be
accepted, a **Write a new interpretation** button. A failed one keeps its **Try again** button,
shown on the same condition. Below the newest, a closed disclosure headed "Earlier
interpretations" lists the others, newest first. Each entry gives its status, when it was asked
for and the version it was written under (a dash when none was recorded). Opened, a verified one
shows its sections with their limits, the note on what "verified" establishes, and the model and
prompt version; a refused one says it is not shown; a failed one shows its error.

**For operators.** The API may not import the interpreter, so it holds its own copy of the current
version, `CURRENT_INTERPRETER_VERSION` in `cp_api.app`; `tests/api/test_interpreter_version.py`
fails when it differs from the one `cp_llm` computes. The API and the worker run from the same
image, so in a running stack they agree. During a restart in which they briefly run different
images, the API judges by one version while the worker records the other, and an analysis asked
about in that window could be offered one more interpretation than intended.

## Measuring misreadings

A verified interpretation cannot cite a figure that is not in its fact table, but it can still
misread the analysis: call a result strong when the table says weak, say a check passed that
failed, or state a conclusion the facts do not support. `cp-llm-eval` measures how often that
happens. It builds a corpus of real fits, has the model interpret each fact table several times
within a call budget, grades every final statement with five deterministic graders (D1 to D5) and
with a judge model, and compares both against statements you label by hand. The report gives each
misreading rate with a standard error clustered by case. A rate of zero carries the rule-of-three
upper bound, 3 / units capped at 100%, instead. A unanimous rate carries the mirrored lower bound,
1 - 3 / units, which the report leaves out whenever it is 0 or less (three units or fewer). Any
other rate whose cases all have the same proportion carries no standard error, and the report says
there is no spread across cases; any other rate over a single case carries none either, and the
report says so (a 0% or 100% rate over one case shows its bound instead). D4, the uncited status
summary behind the primary endpoint, reads a status word only in the clause that holds its
quantifier ("no check", "every diagnostic"), so a status word after ", and" or ", but" does not
count: "every check examines one assumption, and a problem in one may recur" is not a summary.
The judge's sensitivity and specificity and each grader's precision are weighted by label stratum
(judge-flagged, grader-only, unflagged, sampled about a third each) back to all judged statements,
with Wilson intervals at the effective sample size n_eff, `(sum of weights)^2 / sum of squared
weights`: the number of equally weighted labels that would be as informative.

The commands, in order. Every `--max-calls` is required: a paid step does not run without a ceiling.
It counts model calls as the code makes them; the SDK may retry a failed request within one counted
call, so the requests the API bills can exceed it.

```bash
uv run causalpulse llm-eval build --out out/llm-eval/corpus
uv run causalpulse llm-eval run --corpus out/llm-eval/corpus --store out/llm-eval/store --max-calls 50
uv run causalpulse llm-eval grade --corpus out/llm-eval/corpus --store out/llm-eval/store --max-calls 50
uv run causalpulse llm-eval label export --corpus out/llm-eval/corpus --store out/llm-eval/store \
  --out out/llm-eval/to-label.csv
# fill in the verdict, category and note columns of the CSV, then:
uv run causalpulse llm-eval label import --corpus out/llm-eval/corpus --store out/llm-eval/store \
  --csv out/llm-eval/to-label.csv
uv run causalpulse llm-eval report --corpus out/llm-eval/corpus --store out/llm-eval/store \
  --out out/llm-eval/report
uv run causalpulse llm-eval restore --bundle out/llm-eval/report --store out/llm-eval/store-restored
```

- `build` fits each corpus case and writes its fact table. `--case` builds only the cases named. A
  case whose fit lacks the property it was included for is refused.
- `run` writes one record per fact table, prompt and replicate into the store, keyed so that a
  repeated run skips what is already stored. `--variant`, `--replicates` (default 4), `--case` and
  `--workers` select the work. It exits 1 if the budget stopped it before every job ran, or if a
  call failed in transport.
- `grade` sends each verified interpretation to the judge (a refused one has no final narrative) and exits 1 under the same two conditions.
- `label export` writes a seeded sample of final statements (`--size`, default 150; `--seed`) to
  a CSV without the judge's verdict or any grader's flag, so that the labelling is blind. It also
  writes `sample.json` into the store. `label import` reads that file and refuses a CSV that does
  not label exactly the exported sample, or that uses a value outside the allowed verdicts and
  categories. Export only after `grade` has finished, and do not run `grade` again afterwards: the
  weights assume the judged population as it stood at export.
- `report` writes `report.md` and `report.json`, and the stored interpretations and verdicts as
  two gzipped JSON-lines files beside them. `restore` loads those two files back into a store,
  so no model call is needed to rebuild the report. It does not bring the labels back: `report`
  also needs `--corpus` and reads `labels.json` from the store (an absent file counts as no labels), so
  copy the committed `labels.json` into the store before running `report` on a restored store.

`run` and `grade` send each fact table to the Anthropic API, as `interpret` does, and need
`ANTHROPIC_API_KEY` (or an `ant auth login` profile). They are billed. Two live tests run under
`make check-llm`: phase 1's interpretation and one real judge call.

What it does not establish:

- The judge's error is measured on about 150 statements, so its intervals are wide for rare
  categories, and the report says so per category.
- The labels are one person's. No agreement between labellers is measured.
- The corpus is about 25 tables. The rates describe analyses like these, not every dataset a user
  may upload.
- One model: the rates describe `claude-opus-5-5` at its default effort.
- D1 to D5 detect patterns. A misreading phrased outside a grader's patterns is left to the judge,
  or missed; the labelled sample estimates how much.

**Results (2026-10-14).** The full study ran: 25 cases, both prompts, four replies each. The report,
its evidence and the labels are in `docs/reports/2026-10-07-llm-eval/` (`report.md`; the bundle
rebuilds the same `report.json`). Every figure below is copied from `report.json`.

- **The run.** Baseline: 100 interpretations, 99 verified (84 on the first attempt), 1 refused.
  `cite-each-status`: 100, all verified (91 on the first attempt). The judge returned 199 verdicts,
  none unusable. Recorded cost: $16.50 and $15.75 for the interpretations, $9.14 for the judge.
- **The labels.** 150 statements, labelled by hand by the project's owner: 144 `ok`, 6 misreadings.
  The sample draws from three strata, so each rate below is weighted back to all 4,007 judged
  statements: all 20 judge-flagged statements, all 23 flagged only by a grader, and 107 of the
  3,964 that nothing flagged.
- **The judge misses most misreadings.** It flagged 4 of the 6, and 4 of its 20 flags were
  misreadings. The 2 it missed came from the unflagged stratum, where each label stands for about 37
  statements, so its weighted sensitivity for any judge category is **5.1% [0.1%, 67.0%]**
  (n_eff 2.2); specificity is 99.6% [95.8%, 100.0%]. Both misses quote the estimate of a method
  whose band is `INSUFFICIENT` without a caveat.
- **So the judge-graded rates are not reliable in either direction.** It flagged a misreading in 11
  of 99 baseline narratives (11.1%) and 9 of 100 `cite-each-status` narratives (9.0%). But 16 of
  its 20 labelled flags were not misreadings, and it missed most of those that were, so these
  figures neither bound nor estimate the true rate.
- **The rule-based graders flag sentences that are true.** Of the statements flagged only by a
  grader, every one labelled was `ok`: D4 0 of 16 (precision 0.0% [0.0%, 19.4%]), D2 0 of 5, D1 0
  of 2. D4's baseline summaries were 10 true, 0 false and 5 undetermined.
- **The primary endpoint.** `cite-each-status` cut D4 per narrative on first attempts by **14.0
  points [-26.0, -2.0]** over 25 paired cases. Given D4's precision, this measures fewer uncited
  status summaries, not fewer misreadings.

What the study shows, then: verification guarantees the numbers, the commonest remaining misreading
is an unqualified `INSUFFICIENT` estimate, and neither the judge nor the graders measure it
reliably. That misreading is now refused: the verifier's rule `insufficient_caveat` (rule 6 under
the interpretation core, PR #16) requires a statement citing such an estimate to cite its band,
and the system prompt states it, so `prompt_version` moved from `a28f0812170d` to `5b08e57994d4`.
The baseline figures above were measured under `a28f0812170d`, and `cite-each-status` under its
own `5d1427d1ea82`.
