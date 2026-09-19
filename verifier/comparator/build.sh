#!/usr/bin/env bash
# Build with an already-installed Lean 4.28.0; no global toolchain changes.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_dir=${1:?usage: build.sh EMPTY_BUILD_DIR}
mkdir -p "$build_dir"
test -z "$(ls -A "$build_dir")" || { echo 'Build directory must be empty' >&2; exit 1; }
git clone --quiet https://github.com/leanprover/comparator.git "$build_dir"
git -C "$build_dir" checkout --quiet d03acab154d269c06e60e4de7e4cc85deebff94b
cp "$here"/CodeProver*.lean "$here/lakefile.toml" "$build_dir/"
printf 'leanprover/lean4:v4.28.0\n' > "$build_dir/lean-toolchain"
rm "$build_dir/lake-manifest.json"
cd "$build_dir"
lean --version | grep -F 'version 4.28.0'
lake update
# Backport upstream exporter safety fixes through v4.34 (076e8e5): full Quot
# export, duplicate declaration rejection, missing/partial declaration handling.
git -C .lake/packages/lean4export apply --check "$here/export-4.28.patch"
git -C .lake/packages/lean4export apply "$here/export-4.28.patch"
lake build codeprover_comparator codeprover_targets lean4export CodeProverCompile
