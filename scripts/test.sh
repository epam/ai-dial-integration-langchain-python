#!/usr/bin/env bash
set -x

PY_FACTOR=${PYTHON:+py${PYTHON//./}}
echo "PYTHON=$PYTHON"
echo "PY_FACTOR=$PY_FACTOR"

if [[ "$PYTHON" == "3.13" ]]; then
  PARALLEL=1
else
  PARALLEL=auto
fi

"$UV" tool run tox -p ${PARALLEL} \
  --parallel-no-spinner \
  -f test_openai $PY_FACTOR \
  -f test_monkey_patch $PY_FACTOR
