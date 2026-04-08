#!/usr/bin/env bash
set -x

PY_FACTOR=${PYTHON:+py${PYTHON//./}}

echo "PYTHON=$PYTHON"
echo "PY_FACTOR=$PY_FACTOR"
echo "PARALLEL=$PARALLEL"

"$UV" tool run tox \
  --parallel auto \
  --parallel-no-spinner \
  --parallel-live \
  -f test_openai $PY_FACTOR \
  -f test_monkey_patch $PY_FACTOR
