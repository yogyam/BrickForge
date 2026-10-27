# The virtualenv lives outside the project because ~/Desktop is iCloud-synced on this machine and the
# sync daemon marks files under dot-directories as hidden, which makes Python skip the editable .pth.
export UV_PROJECT_ENVIRONMENT ?= $(HOME)/.venvs/brickforge

.PHONY: setup test fixtures demo clean

setup:
	uv sync

test:
	uv run pytest -q

fixtures:
	uv run brickforge fixtures

demo: fixtures
	uv run brickforge generate --mesh fixtures/meshes/sphere.stl --out runs/sphere
	-uv run brickforge generate --mesh fixtures/meshes/open_box_invalid.stl --out runs/open_box_invalid
	@echo "open viewer/index.html and load runs/sphere/build.json (and voxels.json for the target)"

clean:
	rm -rf runs
