.PHONY: install test lint benchmark demo cpp-test
install:
	python -m pip install -e '.[dev]'
test:
	pytest -q
lint:
	ruff check src tests benchmarks
benchmark:
	python benchmarks/run_benchmark.py
demo:
	python -m llm_server.main
cpp-test:
	cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure

