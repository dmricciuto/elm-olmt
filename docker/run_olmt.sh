#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Run OLMT inside the local Docker image from the host.

Usage:
  docker/run_olmt.sh
  docker/run_olmt.sh config_files/SPRUCE.cfg
  docker/run_olmt.sh --shell
  docker/run_olmt.sh -- python -m py_compile model_ELM/makepointdata.py

Defaults:
  image:      elmv3
  code mount: parent of this repository mounted at /code
  inputdata:  sibling inputdata directory mounted at /inputdata
  output:     Docker volume elmoutput mounted at /output

Environment overrides:
  OLMT_DOCKER_IMAGE    Docker image tag, e.g. elmv3:amd64
  OLMT_MODELS_DIR      Host directory mounted at /code
  OLMT_INPUTDATA       Host inputdata directory mounted at /inputdata
  OLMT_OUTPUT          Host path or Docker volume mounted at /output
  OLMT_DOCKER_PLATFORM Optional Docker platform, e.g. linux/amd64
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
models_dir="${OLMT_MODELS_DIR:-$(cd "${repo_dir}/.." && pwd)}"
inputdata_dir="${OLMT_INPUTDATA:-${models_dir}/inputdata}"
output_mount="${OLMT_OUTPUT:-elmoutput}"
image="${OLMT_DOCKER_IMAGE:-elmv3}"
workdir="/code/elm-olmt"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

if [[ ! -d "${models_dir}" ]]; then
  echo "ERROR: models directory does not exist: ${models_dir}" >&2
  exit 2
fi

if [[ ! -d "${inputdata_dir}" ]]; then
  echo "ERROR: inputdata directory does not exist: ${inputdata_dir}" >&2
  echo "Set OLMT_INPUTDATA to the host inputdata path if it lives elsewhere." >&2
  exit 2
fi

docker_args=(
  run --rm
  --hostname=docker
  --user=modeluser
  -v "${models_dir}:/code"
  -v "${inputdata_dir}:/inputdata"
  -v "${output_mount}:/output"
  -w "${workdir}"
  -e MPLCONFIGDIR=/tmp/matplotlib
  -e PYTHONUNBUFFERED=1
)

if [[ -n "${OLMT_DOCKER_PLATFORM:-}" ]]; then
  docker_args+=(--platform "${OLMT_DOCKER_PLATFORM}")
fi

if [[ -t 0 && -t 1 ]]; then
  docker_args+=(-it)
fi

if [[ "${1:-}" == "--shell" ]]; then
  shift
  exec docker "${docker_args[@]}" "${image}" /bin/bash "$@"
fi

if [[ "${1:-}" == "--" ]]; then
  shift
  if [[ $# -eq 0 ]]; then
    echo "ERROR: -- requires a command to run inside the container." >&2
    exit 2
  fi
  cmd=("$@")
elif [[ $# -eq 0 ]]; then
  cmd=(python elm_olmt.py --config config_files/SPRUCE.cfg)
elif [[ "${1}" == *.cfg || -f "${1}" ]]; then
  config="$1"
  shift
  cmd=(python elm_olmt.py --config "${config}" "$@")
else
  cmd=("$@")
fi

echo "Running in Docker image ${image}: ${cmd[*]}"
exec docker "${docker_args[@]}" "${image}" "${cmd[@]}"
