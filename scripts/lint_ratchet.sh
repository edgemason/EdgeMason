#!/usr/bin/env bash
# Ratchet lint: full ruff + basic pyright on files *added* relative to BASE_REF.
# No-op success when no .py file was added.
set -o pipefail

BASE_REF="${BASE_REF:-origin/cleanup}"

candidates=()
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  candidates+=("$line")
done < <(
  {
    git diff --name-only --diff-filter=A "${BASE_REF}...HEAD" -- '*.py' 2>/dev/null || true
    git diff --name-only --diff-filter=A "${BASE_REF}" -- '*.py' 2>/dev/null || true
    git diff --cached --name-only --diff-filter=A "${BASE_REF}" -- '*.py' 2>/dev/null || true
    git ls-files --others --exclude-standard -- '*.py' 2>/dev/null || true
  } | sort -u
)

new_files=()
for f in "${candidates[@]}"; do
  [[ -f "$f" ]] || continue
  case "$f" in *.py) ;; *) continue ;; esac
  new_files+=("$f")
done

if [[ ${#new_files[@]} -eq 0 ]]; then
  echo "lint ratchet: no new .py files; skipping"
  exit 0
fi

echo "lint ratchet: checking new files:"
printf '  %s\n' "${new_files[@]}"

status=0

echo "== ruff (full: E,F,W,I,B,UP,SIM) on new files =="
if ! uv run ruff check --select E,F,W,I,B,UP,SIM -- "${new_files[@]}"; then
  status=1
fi

echo "== pyright (basic) on new files =="
if ! uv run pyright "${new_files[@]}"; then
  status=1
fi

exit "$status"
