#!/usr/bin/env bash
set -x

PY_FACTOR=${PYTHON:+py${PYTHON//./}}
echo "PYTHON=$PYTHON"
echo "PY_FACTOR=$PY_FACTOR"

"$UV" tool run tox -p 4 \
  --parallel-no-spinner \
  -f test_openai $PY_FACTOR \
  -f test_monkey_patch $PY_FACTOR