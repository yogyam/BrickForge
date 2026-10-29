# MCP server (M2.6, demo only)

`brickcore/mcp_server.py` exposes the pipeline as Model Context Protocol tools over stdio so an AI assistant
can drive it: `list_fixtures`, `generate`, `validate`, `plan`, `stability`, `inspect_step`, `bom`,
`describe_build`. Paths are confined to the project directory. The server instructions forbid describing any
build as physically validated.

Claude Code picks it up from `.mcp.json` when started in this folder (approve the project server when asked).
Any other MCP client can run `scripts/mcp.sh`.

Example session: "generate fixtures/meshes/bridge.stl into runs/bridge with 40 pieces, then tell me which step
has the highest tension and what it attaches to."

This has no bearing on the research results; it exists so the demo can be driven conversationally.
