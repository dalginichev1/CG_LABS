#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if command -v glslc >/dev/null 2>&1; then
  compiler="glslc"
elif command -v glslangValidator >/dev/null 2>&1; then
  compiler="glslangValidator"
else
  echo "Не найден GLSL-компилятор: установите glslc или glslangValidator." >&2
  exit 1
fi

for source in shaders/shader.vert shaders/shader.frag shaders/unlit.vert shaders/unlit.frag; do
  output="${source}.spv"
  if [[ "$compiler" == "glslc" ]]; then
    glslc "$source" -o "$output"
  else
    glslangValidator -V "$source" -o "$output"
  fi
  echo "Скомпилирован: $output"
done
