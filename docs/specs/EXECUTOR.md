# Executor Rules

Read this before carrying out any spec in `docs/specs/`. These rules apply to every spec, and a spec cannot loosen them.

You may be any model or coding agent (Claude, opencode, or another). Everything you need is in this file, `TESTING.md` and the spec. Use only plain `git`, `uv` and shell commands. Don't rely on any agent-specific tool.

## Absolute limits (no spec can override these)
- **Redesign only.** No new features, no new user-facing behaviour, no new endpoints, frameworks, targets or options. If something looks like a missing feature, write it down in your report. Don't build it.
- **Never go beyond the sandbox.** Never write outside your worktree. Never push, open PRs, publish, upload, or install anything system-wide. Network access is allowed only for `uv sync` and for asset downloads that a spec names.
- **Allowed writes outside the worktree** (approved by the user on 2026-10-14). These are caches, never code:
  - uv's package cache (`~/.cache/uv`)
  - uv-managed Python installs
  - podman's image store, when a spec asks for a container check

  Nothing else outside the worktree may be written.

## Isolation
1. Work only in a **git worktree** created for the spec, never in the main working tree:
   ```bash
   git worktree add ../edgemason-<spec-id> -b spec/<spec-id> main
   cd ../edgemason-<spec-id>
   ```
2. Never write to the main working tree. That includes its `models/` folder. Read-only use is allowed only where a spec explicitly says so.
3. If the spec needs model or dataset files, provision them inside the worktree with `scripts/provision_assets.py` (once P0.5 exists). Never copy or symlink the user's `models/` folder.

## Scope
4. Touch **only** the files listed in the spec's *Scope*. If you find you need to change anything else, **stop and report**.
5. Never change any file under `api/`, `scripts/`, `frontend/` or `profiles/`, or `reports/EdgeMason_Report.md`, unless the spec's Scope lists it.
6. Never edit golden or baseline files (`tests/golden/`, `baseline/`) unless the spec says so explicitly. Changing them changes the baseline, which needs the user's approval.
7. Never delete or weaken an existing test unless the spec lists it under *Tests that may be deleted*.

## Order of work (TDD)
8. Write the spec's test cases first and run them. Confirm they **fail for the stated reason**, or, for characterization tests, that they **pass against current behaviour**, as the spec says.
9. Implement until they pass. Then run the spec's *Acceptance* commands exactly as written.
10. Don't improvise. If the spec is ambiguous, contradicts the code, or an acceptance command fails for a reason outside your scope, **stop and report**. A half-guessed implementation is worse than none.

## Hand-off
11. Commit on the spec branch. The message is `<spec-id>: <title>`, and the body lists the acceptance commands you ran and their output summaries.
12. Do not merge, push or open a PR. Finish with a report containing:
    - the files changed
    - each acceptance command with its pass/fail result
    - anything you stopped on
    - any finding about the code that the spec didn't anticipate
