#!/bin/sh
# Start the BrickForge MCP server over stdio with the venv kept outside the iCloud-synced project folder.
cd "$(dirname "$0")/.." || exit 1
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/brickforge}"
exec uv run --quiet python -m brickcore.mcp_server
