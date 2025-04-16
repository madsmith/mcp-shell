# MCP Shell

A Python MCP server implementation with support for both stdio and SSE transport, providing command and script execution capabilities.

## Features

- Dual transport support (stdio and SSE)
- Command execution with `run_command` tool
- Script execution with `run_script` tool
- Working directory support for both commands and scripts
- Full stdout/stderr capture
- Error handling and reporting

## Setup

Install the package:
```bash
pip install .
```

## Running the Server

The server can be run in either stdio mode (default) or SSE server mode:

```bash
# Run in stdio mode (default)
mcp-shell

# Run in SSE server mode
mcp-shell --server

# Run in SSE server mode with custom host and port
mcp-shell --server -H localhost -p 8080
```

### Command Line Options

- `--server`: Run in SSE server mode (default: stdio mode)
- `-H, --host`: Host to run the server on (default: 0.0.0.0)
- `-p, --port`: Port to run the server on (default: 8050)

## Available Tools

### run_command
Execute shell commands with optional working directory support.

Example:
```python
result = await run_command(ctx, "ls -la", cwd="/path/to/dir")
```

### run_script
Execute scripts using a specified interpreter with optional working directory support.

Example:
```python
result = await run_script(ctx, 
    interpreter="python",
    script="print('Hello World')",
    cwd="/path/to/dir"
)
