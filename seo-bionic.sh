#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Missing workflow command. Example: ./seo-bionic.sh status --job <job>" >&2
  exit 2
fi

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SEO_WORKFLOW_ARGUMENTS
SEO_WORKFLOW_ARGUMENTS="$(printf '%s\n' "$@")"

cd -- "$script_directory"
exec powershell.exe \
  -NoLogo \
  -NoProfile \
  -NonInteractive \
  -ExecutionPolicy Bypass \
  -File "./seo.ps1"
