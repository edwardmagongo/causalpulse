# Provenance of the pre-registration

The endpoints in `prereg.md` were fixed before the arm was billed. The documents that fixed
them are not in this repository: they live in a separate design-documents repository, by the
same rule that keeps every other CausalPulse spec out of the source tree. What is published
here are their digests, so the documents can be checked against what was frozen without the
documents themselves being published.

A digest is not the specification. It cannot be read back into one, and it does not grant
anyone a copy. It does one thing: it makes the frozen text identifiable, so a later version
cannot be presented as the registered one.

## The documents

| Document | Bytes | SHA-256 |
|---|---|---|
| `2026-10-29-llm-eval-ungrounded-control-arm-design.md` | 21921 | `2d2b903b8f2903320f48cfbbe427996b84b285e05c9ffc2239c426120b5b5968` |
| `2026-10-29-llm-eval-ungrounded-control-arm-prereg.md` | 14552 | `47b41e273d61fb569c72fd6487ec5f308b91ee9e743f70db76d0515306d4e5a1` |

The design document's digest is the same at every commit below: it did not change once the
history began. The pre-registration's did, in §8 alone, which records the hashes and the
balance to be filled in before the first billed call and holds no analytic choice:

| Commit | SHA-256 of the pre-registration | What changed |
|---|---|---|
| `056a996a970e989ca516572bf48f0fad54695ef1` | `18e672a94fff1dd86d651385d0323569f397ed899b6a49df129b20edfbcacd12` | the substantive freeze, §8 unfilled |
| `5fcb35b1bb048fd6b05bcbbd7181f475a980b327` | `7c261edcb3b4a2f8e1ea69fc6f7997a8194fde296150b4ba346e28a0001bdaab` | §8 filled but for the balance |
| `10ae24cdf242c27618f186f2ccdb4c387f547cd5` | `47b41e273d61fb569c72fd6487ec5f308b91ee9e743f70db76d0515306d4e5a1` | §8's options costed against the balance |

To check a copy of either document:

```
shasum -a 256 2026-10-29-llm-eval-ungrounded-control-arm-prereg.md
```

The commit ids are git's own, and that repository is a SHA-1 one, which is why the SHA-256 of
each file is given as well.

## The history rewrite

Those three commit ids are not the ones the documents were first written under. On 2026-11-14
the design-documents repository was pushed to a private remote, so that the freeze no longer
lived on one disk. Before it was pushed, 172 passages of AI-attribution text were stripped from
34 of the other documents — the same text this repository's own history was rewritten to remove
on 2026-10-14 — and that changed every commit id:

| Before the rewrite | After |
|---|---|
| `76cc2e1` | `056a996a970e989ca516572bf48f0fad54695ef1` |
| `0dabfb3` | `5fcb35b1bb048fd6b05bcbbd7181f475a980b327` |
| `849758c` | `10ae24cdf242c27618f186f2ccdb4c387f547cd5` |

Neither of the two documents above was among the 34, so the rewrite left every SHA-256 in this
file untouched and a copy of either document still verifies against it. The author dates were
carried over unchanged. §8 of the pre-registration names `76cc2e1` as the commit holding it:
that is the id from before the rewrite, and `056a996` is the same tree with the attribution
removed.

## The sequence

All times UTC, from the commits themselves and from the run's log file.

| Time | Event | Witnessed by |
|---|---|---|
| 10:44:30 | `6dc1a26` merged here — the analysis code §8 names as frozen | GitHub |
| 10:45:03 | the pre-registration's substantive freeze | this machine |
| 10:46:12 | §8 filled but for the balance | this machine |
| 10:52:18 | §8's options costed | this machine |
| 10:57:40 | the arm's first billed call | this machine |
| 11:07:22 | the arm's hundredth narrative stored | this machine |
| 11:26:56 | `7b0f05a` — the run's output committed here | GitHub |
| 11:37:17 | the judge's first billed call | this machine |
| 11:40:01 | the judge's hundredth verdict stored | this machine |

## What this does not establish

GitHub independently witnesses the two ends of that sequence: the analysis code existed at
10:44:30, and the data existed at 11:26:56. Neither the pre-registration's own commits nor
the moment of the first billed call is witnessed by anything off this machine, and a local
git timestamp is written by whoever makes the commit. So the ordering above is the record of
a single machine, and it is not proof that §2's hypotheses and §3's thresholds predated the
data.

Two things do stand on their own. The analysis of both endpoints is `cp_llm_eval/prereg.py`,
written and passing before any call was billed and frozen in a commit GitHub timestamps, so
the procedure that produced `prereg.md` is verifiably not a procedure chosen after seeing the
numbers. And P0, the gate that has to pass for P1 to mean anything, is deterministic and
costs nothing: anyone can rerun it over the stored grounded narratives and get the same 0 of
4007.

The honest summary is that the thresholds were pre-registered under the author's own record,
and the analysis was pre-registered under a record neither the author nor anyone else can
quietly rewrite.
