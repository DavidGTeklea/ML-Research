#!/usr/bin/env bash
# merge_verb_outputs.sh
set -euo pipefail

# Where chunk files live (current dir) and where to put merged files
SRC_DIR="${1:-.}"
OUT_DIR="${2:-outputs}"

mkdir -p "$OUT_DIR"

keys=(agent_location agent_instrument agent_patient all_roles)

for key in "${keys[@]}"; do
  # Grab matching chunk files, sort by the numeric chunk id
  mapfile -t chunks < <(ls -1 "${SRC_DIR}/verb_outputs_${key}_chunk"*.txt 2>/dev/null | sort -V)

  if (( ${#chunks[@]} == 0 )); then
    echo "[WARN] No chunks found for ${key} (pattern: ${SRC_DIR}/verb_outputs_${key}_chunk*.txt)"
    continue
  fi

  tmp="${OUT_DIR}/.verb_outputs_${key}.tmp"
  out="${OUT_DIR}/verb_outputs_${key}.txt"

  # Build atomically: write to tmp, then mv into place
  : > "$tmp"
  for f in "${chunks[@]}"; do
    echo "==== merging $(basename "$f") ====" >> "$tmp"
    cat "$f" >> "$tmp"
    printf "\n" >> "$tmp"
  done

  mv -f "$tmp" "$out"
  echo "[OK] Wrote ${out} (from ${#chunks[@]} chunks)"
done
