#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
built=${1:?usage: build-image.sh BUILT_COMPARATOR_DIR IMAGE_TAG}
image=${2:?image tag required}
context=$(mktemp -d)
trap 'rm -rf -- "$context"' EXIT
mkdir "$context/bin" "$context/lean"
cp "$built/.lake/build/bin/codeprover_comparator" "$built/.lake/build/bin/codeprover_targets" "$context/bin/"
cp "$built/.lake/packages/lean4export/.lake/build/bin/lean4export" "$context/bin/"
strip "$context/bin/"*
for suffix in olean olean.private olean.server; do
  cp "$built/.lake/build/lib/lean/CodeProverCompile.${suffix}" "$context/lean/"
done
cp "$built/.lake/build/lib/lean/CodeProverCompile.ir" "$context/lean/"
cp "$here/Dockerfile" "$here/phase.py" "$context/"
docker build --tag "$image" "$context"
