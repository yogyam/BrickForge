# The virtualenv lives outside the project because ~/Desktop is iCloud-synced on this machine and the
# sync daemon marks files under dot-directories as hidden, which makes Python skip the editable .pth.
export UV_PROJECT_ENVIRONMENT ?= $(HOME)/.venvs/brickforge

.PHONY: setup test fixtures demo bench clean

setup:
	uv sync

test:
	uv run pytest -q

fixtures:
	uv run brickforge fixtures

demo: fixtures
	uv run brickforge generate --mesh fixtures/meshes/sphere.stl --out runs/sphere
	@echo "--- intentional failure: the open box is not watertight and must be rejected (exit 1 is expected) ---"
	-@uv run brickforge generate --mesh fixtures/meshes/open_box_invalid.stl --out runs/open_box_invalid
	@echo "--- done. Open viewer/index.html and load runs/sphere/build.json (and voxels.json for the target) ---"

bench:
	uv run brickforge-bench fetch-thingi --n 40
	uv run brickforge-bench run --suite all --generators greedy_v1 --budgets 50 100 200 --out docs/benchmarks/baseline_v1

clean:
	rm -rf runs
