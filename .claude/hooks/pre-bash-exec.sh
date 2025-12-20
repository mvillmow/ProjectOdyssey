#!/usr/bin/env bash
# Claude pretooluse-input hook
# Blocks destructive bash commands
# Uses pixi-provided jq
#
# Contract:
# - exit 0  => allow
# - exit !=0 => block (must emit JSON)

set -u
trap 'exit 0' ERR  # fail-open

EXIT_ALLOW=0
EXIT_BLOCK=1

PAYLOAD="$(cat)"

block() {
  pixi run jq -n --arg msg "$1" \
    '{ "action": "block", "reason": $msg }'
  exit $EXIT_BLOCK
}

# ---- dependencies ----
command -v pixi >/dev/null 2>&1 || block "pixi is required for this hook"
pixi run jq --version >/dev/null 2>&1 || block "jq not available via pixi"

# ---- parse payload ----
TOOL="$(pixi run jq -r '.tool // empty' <<<"$PAYLOAD")"
[[ "$TOOL" != "bash" ]] && exit $EXIT_ALLOW

CMD="$(pixi run jq -r '.input.command // empty' <<<"$PAYLOAD")"
CWD="$(pixi run jq -r '.input.cwd // empty' <<<"$PAYLOAD")"

[[ -z "$CMD" ]] && exit $EXIT_ALLOW
[[ -z "$CWD" ]] && CWD="$(pwd)"

PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$CWD" rev-parse --show-toplevel 2>/dev/null || pwd)}"
HOME_DIR="$(cd ~ && pwd)"

canon() {
  realpath -P -- "$1" 2>/dev/null
}

validate_path() {
  local p="$1"

  # expand ~ and literal $HOME
  case "$p" in
    "~"*) p="$HOME/${p#~}" ;;
    '$HOME'*) p="$HOME/${p#\$HOME}" ;;
  esac

  # make absolute if relative
  [[ "$p" != /* ]] && p="$CWD/$p"

  local abs
  abs="$(canon "$p")" || block "path does not resolve: $p"

  # block dangerous locations
  [[ "$abs" == "/" ]] && block "targeting filesystem root"
  [[ "$abs" == */.git* ]] && block "targeting .git directory"

  # block everything in HOME except project root
  if [[ "$abs" == "$HOME"* ]] && [[ "$abs" != "$PROJECT_ROOT"* ]]; then
    block "targeting home directory outside project root"
  fi

  # block anything outside project root
  [[ "$abs" != "$PROJECT_ROOT"* ]] && block "outside project root: $abs"
}

# ---- global shell-danger checks ----
grep -Eq '(`|\$\(|\$\{)' <<<"$CMD" && block "dangerous shell expansion detected"

# Normalize sudo
CMD="${CMD#sudo }"

# ---- rm / unlink ----
if [[ "$CMD" =~ (^|[[:space:]])(rm|unlink)([[:space:]]|$) ]]; then
  grep -Eq '\-\-no-preserve-root' <<<"$CMD" && block "rm uses --no-preserve-root"

  args=()
  for a in $CMD; do
    [[ "$a" == rm || "$a" == unlink ]] && continue
    [[ "$a" == -* ]] && continue
    args+=("$a")
  done

  [[ "${#args[@]}" -eq 0 ]] && block "rm with no paths"

  for p in "${args[@]}"; do
    validate_path "$p"
  done
fi

# ---- find destructive ----
grep -Eq 'find .* (-delete|-exec[[:space:]]+rm)' <<<"$CMD" \
  && block "destructive find usage"

# ---- xargs rm ----
grep -Eq 'xargs[[:space:]]+rm' <<<"$CMD" \
  && block "xargs rm is blocked"

# ---- git clean ----
grep -Eq 'git[[:space:]]+clean.*-[fdx]' <<<"$CMD" \
  && block "git clean with force flags"

# ---- rsync --delete ----
grep -Eq 'rsync .*--delete' <<<"$CMD" \
  && block "rsync --delete is blocked"

# ---- recursive permission changes ----
grep -Eq '(chmod|chown|chgrp).* -R' <<<"$CMD" \
  && block "recursive permission or ownership change blocked"

# ---- dangerous system tools ----
grep -Eq '(^| )(dd|mkfs|wipefs|mount|umount)( |$)' <<<"$CMD" \
  && block "dangerous system-level command blocked"

# ---- archive extraction ----
grep -Eq 'tar .* -[^ ]*x' <<<"$CMD" \
  && block "tar extraction blocked"

grep -Eq 'unzip ' <<<"$CMD" \
  && block "unzip blocked"

exit $EXIT_ALLOW
