#!/bin/sh
# Run the BrickForge CLI with the virtualenv kept outside the iCloud-synced project folder.
# Usage: ./bf generate --mesh fixtures/meshes/sphere.stl --out runs/sphere
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/brickforge}"
exec uv run --quiet brickforge "$@"
