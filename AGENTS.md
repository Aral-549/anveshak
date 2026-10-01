# AGENTS.md — Workflow Rules for AI Coding Agents

## 0. No emojis — ever
Never use emojis in code, comments, commit messages, notifications, CLI output,
or conversation replies. Plain text only. No exceptions.

Read this before writing any code in this repo or workspace. These rules exist because mixing
multiple models (or multiple passes of the same model) without a shared contract
produces bugs that are scattered and hard to trace back to a cause. Follow them
exactly, even under time pressure — especially under time pressure.

## 1. Contract before code
Before implementing or changing any module, check `contracts/` for an existing
spec. If none exists for what you're building, write one first — as plain
input → expected-output cases, not code — and stop for review before
implementing. Use the Contracts Template below (or `CONTRACTS_TEMPLATE.md`).

## 2. Never grade your own work
If you just wrote or modified implementation code, your next action is NOT to
write tests that confirm it works. Either:
- stop and hand off to a separate pass/agent for testing, or
- explicitly switch mode to "adversarial reviewer": your only goal is to find
  inputs that break what you just wrote, not confirm it's correct.

If you are the only agent available, you must still do this as two distinct,
separated passes — do not write tests in the same turn you wrote the
implementation.

## 3. Golden tests are frozen
`tests/golden/` holds hand-verified ground truth for safety-critical logic.
Never regenerate, "clean up," or rewrite these without explicit human approval.
You may add new cases; you may not modify or delete existing ones.

## 4. Every core-logic change needs a matching test change
A change that touches anything under `core/` (or this project's equivalent
safety-critical path) without a corresponding change in `tests/golden/` or
`contracts/` is incomplete, not ready for review, regardless of how confident
it looks.

## 5. Log at every pipeline stage boundary
For multi-stage pipelines (e.g. input → parser → engine → output), log
structured input/output at each boundary — not print statements. This is what
makes a scattered bug traceable to a specific stage instead of "somewhere in
the app."

## 6. Definition of done
A feature or fix is NOT done until all of the following are true:
- [ ] Contract exists in `contracts/` and matches implemented behavior
- [ ] Golden tests pass, including any new cases for this change
- [ ] Integration test against real (not synthetic/mocked) input passes
- [ ] An adversarial pass has been run against the change
- [ ] If this closes a bug, a regression case was added (see `BUGLOG_TEMPLATE.md`)

## 7. Bugs become permanent regression tests
Any bug found — by a human or an agent — gets logged using
`BUGLOG_TEMPLATE.md` (in `BUGLOG.md`), and a corresponding case is added to `tests/golden/`
before the fix counts as complete. A fix without the regression case is not
done.

---

## Templates

### `CONTRACTS_TEMPLATE.md` (for `contracts/<module>.md`)
```markdown
# Contract: <module or feature name>

## Purpose
One or two sentences: what this component is responsible for, and what it
explicitly hands off to something else.

## Inputs
- `<input 1>`: type / shape / constraints
- `<input 2>`: type / shape / constraints

## Outputs
- `<output>`: type / shape / constraints

## Behavior cases (input → expected output)
| # | Input | Expected output | Notes |
|---|-------|------------------|-------|
| 1 | | | |
| 2 | | | |
| 3 | | | |

## Edge cases that must be covered
- <ambiguous or boundary condition>
- <ambiguous or boundary condition>
- <malformed/unexpected input from the previous pipeline stage>

## Explicitly out of scope
- <what this component does NOT handle, and which module does instead>

## Status
- [ ] Drafted
- [ ] Reviewed by a human
- [ ] Implementation matches this contract
- [ ] Golden tests exist for every behavior case above
```

### `BUGLOG_TEMPLATE.md` (for `BUGLOG.md`)
```markdown
# Bug Log

Every entry here must result in a permanent case added to `tests/golden/`
before it's marked resolved. A patched bug without a regression case is not
resolved — it's just hidden until the next rewrite.

---

## YYYY-MM-DD — <short title>
- **Symptom:** what was actually observed (not the assumed cause)
- **Root cause:** what was actually wrong, once traced through stage logs
- **Stage/module:** where in the pipeline this lived
- **Regression case added:** `tests/golden/<file>` — case #<id>
- **Status:** open / fixed / verified
```
