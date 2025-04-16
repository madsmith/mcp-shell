from mcp.server.fastmcp import FastMCP, Context
import asyncio
import json
import argparse
import os
import sys
import platform
import socket
from functools import wraps
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Callable, TypeVar, Union, Awaitable
import re

F = TypeVar('F', bound=Callable[..., Union[Any, Awaitable[Any]]])

def with_sys_info() -> Callable[[F], F]:
    """Decorator that adds system information to a function's docstring.
    
    Adds details about the platform, hostname, user environment, and other
    system-specific information that helps LLMs understand the execution context.
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
            
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Get system information
        sys_info = [
            f"Platform: {sys.platform}",
            f"OS: {platform.system()} {platform.release()}",
            f"Architecture: {platform.machine()}",
            f"Hostname: {socket.gethostname()}",
            f"Python Version: {sys.version.split()[0]}"
        ]
        
        # Add environment variables if they exist
        env_vars = {
            "User Home Directory": "HOME",
            "User": "USER",
            "Shell": "SHELL"
        }
        
        for label, var in env_vars.items():
            if var in os.environ:
                sys_info.append(f"{label}: {os.environ[var]}")
        
        # Combine original docstring with system info
        original_doc = func.__doc__ or ""
        sys_info_str = "\n\nSystem Information:\n" + "\n".join(sys_info)
        
        # Choose the appropriate wrapper based on whether the function is async
        wrapper = async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        wrapper.__doc__ = original_doc + sys_info_str
        
        return wrapper
    return decorator

def hostname_suffix() -> Callable[[F], F]:
    """Decorator that renames the function by appending a sanitized hostname.
    
    This ensures unique function names across different machines running the same MCP server.
    The hostname is sanitized to only include alphanumeric characters and underscores.
    """
    def decorator(func: F) -> F:
        # Get hostname and sanitize it
        hostname = socket.gethostname().lower()
        sanitized_hostname = re.sub(r'[^a-z0-9_]', '_', hostname)
        
        # Create wrapper with the same signature
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
            
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
            
        # Choose appropriate wrapper
        wrapper = async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        # Rename the function
        wrapper.__name__ = f"{func.__name__}_{sanitized_hostname}"
        
        return wrapper
    return decorator

# Initialize FastMCP server
mcp = FastMCP(
    "mcp-shell",
    description="MCP server for command and script execution"
)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MCP server for command and script execution"
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run in SSE server mode (default: stdio mode)"
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8050,
        help="Port to run the server on (default: 8050)"
    )
    parser.add_argument(
        "-H", "--host",
        default="0.0.0.0",
        help="Host to run the server on (default: 0.0.0.0)"
    )
    return parser.parse_args()

@dataclass
class CommandResult:
    """Result of a command execution."""
    stdout: str
    stderr: str
    returncode: int
    error_message: Optional[str] = None

async def execute_command(command: str, cwd: Optional[str] = None) -> CommandResult:
    """Execute a shell command and return the result.
    
    Args:
        command: The command to execute
        cwd: Optional working directory for command execution
    
    Returns:
        CommandResult containing stdout, stderr, and return code
    """
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd
        )
        stdout, stderr = await process.communicate()
        return CommandResult(
            stdout=stdout.decode() if stdout else "",
            stderr=stderr.decode() if stderr else "",
            returncode=process.returncode or 0
        )
    except Exception as e:
        return CommandResult(
            stdout="",
            stderr="",
            returncode=1,
            error_message=str(e)
        )

async def execute_script(interpreter: str, script: str, cwd: Optional[str] = None) -> CommandResult:
    """Execute a script using the specified interpreter.
    
    Args:
        interpreter: The interpreter to use (e.g., python, node)
        script: The script content to execute
        cwd: Optional working directory for script execution
    
    Returns:
        CommandResult containing stdout, stderr, and return code
    """
    try:
        process = await asyncio.create_subprocess_exec(
            interpreter,
            "-c" if interpreter in ["python", "python3"] else "-e",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd
        )
        stdout, stderr = await process.communicate()
        return CommandResult(
            stdout=stdout.decode() if stdout else "",
            stderr=stderr.decode() if stderr else "",
            returncode=process.returncode or 0
        )
    except Exception as e:
        return CommandResult(
            stdout="",
            stderr="",
            returncode=1,
            error_message=str(e)
        )

def format_result(result: CommandResult) -> List[Dict[str, str]]:
    """Format command result into a list of text content messages."""
    messages = []
    
    if result.error_message:
        messages.append({
            "type": "text",
            "text": result.error_message,
            "name": "ERROR"
        })
    
    if result.stdout:
        messages.append({
            "type": "text",
            "text": result.stdout,
            "name": "STDOUT"
        })
    
    if result.stderr:
        messages.append({
            "type": "text",
            "text": result.stderr,
            "name": "STDERR"
        })
    
    return messages

@mcp.tool()
@hostname_suffix()
@with_sys_info()
async def run_command(ctx: Context, command: str, cwd: Optional[str] = None) -> str:
    """Run a shell command.

    Args:
        ctx: The MCP server context
        command: The command to execute
        cwd: Optional working directory for command execution
    """
    if not command:
        return json.dumps({
            "isError": True,
            "content": [{"type": "text", "text": "Command is required", "name": "ERROR"}]
        })

    result = await execute_command(command, cwd)
    return json.dumps({
        "isError": result.returncode != 0,
        "content": format_result(result)
    })

@mcp.tool()
@hostname_suffix()
@with_sys_info()
async def run_script(ctx: Context, interpreter: str, script: str, cwd: Optional[str] = None) -> str:
    """Run a script using the specified interpreter.

    Args:
        ctx: The MCP server context
        interpreter: The interpreter to use (e.g., python, node)
        script: The script content to execute
        cwd: Optional working directory for script execution
    """
    if not interpreter:
        return json.dumps({
            "isError": True,
            "content": [{"type": "text", "text": "Interpreter is required", "name": "ERROR"}]
        })
    
    if not script:
        return json.dumps({
            "isError": True,
            "content": [{"type": "text", "text": "Script is required", "name": "ERROR"}]
        })

    result = await execute_script(interpreter, script, cwd)
    return json.dumps({
        "isError": result.returncode != 0,
        "content": format_result(result)
    })

async def run_server(args: argparse.Namespace):
    """Run the MCP server with the specified arguments."""
    # Update server settings
    mcp.settings.host = args.host
    mcp.settings.port = args.port

    if args.server:
        # Run the MCP server with SSE transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()

def main():
    """Entry point for the MCP shell server."""
    args = parse_args()
    return asyncio.run(run_server(args))

if __name__ == "__main__":
    main()
