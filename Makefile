VENV_DIR ?= .venv
UV ?= $(VENV_DIR)/bin/uv
UV_VERSION ?= 0.7.3

all: install

init_env:
	python -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install uv==$(UV_VERSION) --quiet
	$(UV) tool install tox==4.15.1 --with tox-uv

install: init_env
	$(UV) sync --quiet --all-extras

build: install
	$(UV) build

publish: build
	$(UV) publish --username __token__ --password $(PYPI_TOKEN)

clean:
	rm -rf $(VENV_DIR)
	rm -rf .tox
	rm -rf .pytest_cache
	rm -rf dist
	find . -type d -name __pycache__ | xargs rm -r

lint: install
	$(UV) tool run tox -e lint

format: install
	$(UV) tool run tox -e format

test: install
	UV=$(UV) PYTHON=$(PYTHON) ./scripts/test.sh

help:
	@echo '===================='
	@echo 'build                        - build the library'
	@echo 'clean                        - clean virtual env and build artifacts'
	@echo 'publish                      - publish the library to Pypi'
	@echo '-- LINTING --'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo '-- TESTS --'
	@echo 'test                         - run unit tests'
	@echo 'test PYTHON=<python_version> - run unit tests with the specific python version'
