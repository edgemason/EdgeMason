# <SPEC-ID>: <Title>

**Depends on:** <spec ids, or "none">
**Gate class:** identical | intended improvement: <metric> must <improve by X / reach Y>
**Status:** interviewing | agreed | executing | done
**Executor:** follow `docs/specs/EXECUTOR.md` (its absolute limits always apply).

Each section is written only after the planner has interviewed the user about it.

## 1. Goal
<Who it is for, the problem, and the one observable outcome that means "done".>

## 2. Schema
<The exact shapes the result must have: signatures, file formats, config keys, CLI flags, doc structure.>

## 3. Scope
**May create or change:**
- `path`: <what>

**Out of scope (do not touch, even if tempting):**
- <item>

## 4. Behaviour (tests first)
| # | Given | When | Then |
|---|---|---|---|
| 1 | | | |

**Acceptance (run exactly these; all must pass):**
```bash
uv run pytest -q
```

**Tests that may be deleted:** none

## 5. Failures
<Every way this can go wrong, and what the executor does in each case: stop and report, or the exact fallback.>

## Rollback
Discard the worktree: `git worktree remove --force ../edgemason-<spec-id>` and `git branch -D spec/<spec-id>`.
