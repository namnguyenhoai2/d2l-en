#!/usr/bin/env bash
# Build a local HTML preview from Git Bash. This script never installs packages.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/build-html.sh [--clean] [--evaluate]

  --clean     Remove previous RST and HTML output before building.
  --evaluate  Re-evaluate notebooks. The default is a fast translation preview
              that uses existing notebook outputs.
EOF
}

clean=false
evaluate=false
for arg in "$@"; do
  case "$arg" in
    --clean) clean=true ;;
    --evaluate) evaluate=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

# d2lbook writes generated Sphinx files using Python's default text encoding.
# On Windows that can be CP1252, which cannot represent Vietnamese text.
export PYTHONUTF8=1
# Load the repository-local Windows path compatibility shim before d2lbook.
export PYTHONPATH="$project_root/scripts${PYTHONPATH:+:$PYTHONPATH}"

# Prefer an activated environment, but conveniently use the conventional uv
# environment when it exists. Neither branch installs or updates packages.
if command -v d2lbook >/dev/null 2>&1; then
  d2lbook_cmd=(d2lbook)
  python_cmd=(python)
  sphinx_cmd=(sphinx-build)
elif [[ -x .venv/Scripts/d2lbook.exe ]]; then
  d2lbook_cmd=(.venv/Scripts/d2lbook.exe)
  python_cmd=(.venv/Scripts/python.exe)
  sphinx_cmd=(.venv/Scripts/sphinx-build.exe)
else
  cat >&2 <<'EOF'
d2lbook was not found. Create and prepare the environment yourself, then retry:

  uv venv --python 3.9 .venv
  uv pip install --system-certs --python .venv/Scripts/python.exe d2lbook
  uv pip install --system-certs --python .venv/Scripts/python.exe -e .
EOF
  exit 1
fi

if ! command -v pandoc >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Pandoc was not found. nbconvert needs it to build HTML.
Install it yourself from Git Bash, then open a new Git Bash window and retry:

  winget install --id JohnMacFarlane.Pandoc -e

Verify the installation with: pandoc --version
EOF
  exit 1
fi

# notedown, used by this older D2L builder, still imports pkg_resources.
# Setuptools 82 removed that module; explain the one-time compatibility fix
# rather than changing the user's environment automatically.
if ! "${python_cmd[@]}" -c 'import pkg_resources' >/dev/null 2>&1; then
  cat >&2 <<'EOF'
This d2lbook version requires pkg_resources through notedown.
Your environment has setuptools 82 or newer, where pkg_resources was removed.
Run this once, then rerun the build:

  uv pip install --system-certs --python .venv/Scripts/python.exe "setuptools<82"
EOF
  exit 1
fi

if [[ "$clean" == true ]]; then
  rm -rf _build
fi

config_backup="$(mktemp)"
cp config.ini "$config_backup"
restore_config() {
  cp "$config_backup" config.ini
  rm -f "$config_backup"
}
trap restore_config EXIT

if [[ "$evaluate" == false ]]; then
  sed -i 's/^eval_notebook[[:space:]]*=[[:space:]]*.*/eval_notebook = False/' config.ini
fi

# This repository's [build] tabs setting. Keep this list aligned with config.ini
# when adding or removing a framework.
tabs=(pytorch mxnet jax tensorflow)

# `rst --tab all` merges the per-framework notebooks from _build/eval*. Run
# eval first even for a fast preview: with eval_notebook=False it converts the
# Markdown only and does not execute notebook cells.
for tab in "${tabs[@]}"; do
  "${d2lbook_cmd[@]}" build eval --tab "$tab"
done
"${d2lbook_cmd[@]}" build rst --tab all
cp static/frontpage/frontpage.html _build/rst_all/

# Do not use `d2lbook build html` here. Its HTML command additionally builds
# Colab and SageMaker notebook copies, which calls the legacy `ipynb` target.
# That target shells out to `cp` and is unreliable on Windows. The RST target
# above already prepares Sphinx's configuration, so build the same HTML output
# directly without the unrelated notebook-export step.
"${sphinx_cmd[@]}" _build/rst_all _build/html -b html -c _build/rst_all -j 4
mkdir -p _build/html/_images
cp -R static/frontpage/_images/. _build/html/_images/

if [[ ! -f _build/html/index.html ]]; then
  echo 'Build completed but _build/html/index.html was not created.' >&2
  exit 1
fi

echo "HTML preview is ready: $project_root/_build/html/index.html"
