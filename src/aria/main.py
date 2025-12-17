"""Main entry point for ARIA CLI.

This module provides the command-line interface for ARIA using Click,
including the interactive chat loop and utility commands.
"""

import asyncio
import sys
import signal
from typing import NoReturn

import click

from aria import __version__
from aria.config import get_settings
from aria.llm import OllamaClient, OllamaConnectionError, ChatMessage
from aria.ui.console import ARIAConsole, get_console
from aria.ui.prompts import confirm
from aria.agent import Agent
from aria.tools import get_registry
from aria.tools.examples import EchoTool, SystemInfoTool
from aria.tools.filesystem import (
    ListDirectoryTool,
    ReadFileTool,
    AnalyzeFileTool,
    AnalyzeDirectoryTool,
    OrganizeFilesTool,
    MoveFileTool,
    CopyFileTool,
    CreateDirectoryTool,
    DeleteFileTool,
)
from aria.tools.documents import (
    OCRTool,
    ProcessDocumentTool,
    ProcessInboxTool,
    DocumentClassifier,
    DocumentProcessor,
)
from aria.tools.organization import (
    UndoOrganizationTool,
    ListOrganizationLogsTool,
)
from aria.approval import ApprovalHandler
from aria.logging import setup_logging


# Global flag for graceful shutdown
_shutdown_requested = False


def signal_handler(sig, frame):
    """Handle interrupt signals gracefully."""
    global _shutdown_requested
    _shutdown_requested = True


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)


@click.group(invoke_without_command=True)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--debug", "-d", is_flag=True, help="Enable debug mode (very detailed logging)")
@click.option("--no-color", is_flag=True, help="Disable colored output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, debug: bool, no_color: bool):
    """ARIA - AI Research & Intelligence Assistant

    A local-first agentic AI assistant with privacy-preserving design.
    """
    # Store options in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["debug"] = debug
    ctx.obj["no_color"] = no_color

    # If no subcommand, default to chat
    if ctx.invoked_subcommand is None:
        ctx.invoke(chat, verbose=verbose, debug=debug, no_color=no_color)


@cli.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--debug", "-d", is_flag=True, help="Enable debug mode (very detailed logging)")
@click.option("--no-color", is_flag=True, help="Disable colored output")
@click.option("--session", "-s", default=None, help="Resume existing session by ID")
@click.option("--new", "-n", is_flag=True, help="Force new session (ignore existing)")
def chat(verbose: bool, debug: bool, no_color: bool, session: str | None, new: bool):
    """Start interactive chat session (default command)."""
    asyncio.run(run_chat_loop(
        verbose=verbose,
        debug=debug,
        no_color=no_color,
        session_id=session,
        force_new=new,
    ))


@cli.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--debug", "-d", is_flag=True, help="Enable debug mode (very detailed logging)")
@click.option("--no_color", is_flag=True, help="Disable colored output")
def models(verbose: bool, debug: bool, no_color: bool):
    """List available Ollama models."""
    asyncio.run(list_models(verbose=verbose, debug=debug, no_color=no_color))


@cli.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--debug", "-d", is_flag=True, help="Enable debug mode (very detailed logging)")
@click.option("--no_color", is_flag=True, help="Disable colored output")
def health(verbose: bool, debug: bool, no_color: bool):
    """Check Ollama connection status."""
    asyncio.run(check_health(verbose=verbose, debug=debug, no_color=no_color))


@cli.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--debug", "-d", is_flag=True, help="Enable debug mode (very detailed logging)")
@click.option("--no_color", is_flag=True, help="Disable colored output")
def config(verbose: bool, debug: bool, no_color: bool):
    """Show current configuration."""
    show_config(verbose=verbose, debug=debug, no_color=no_color)


@cli.command()
def version():
    """Show version information."""
    click.echo(f"ARIA version {__version__}")


@cli.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--debug", "-d", is_flag=True, help="Enable debug mode (very detailed logging)")
@click.option("--no-color", is_flag=True, help="Disable colored output")
def test_timing(verbose: bool, debug: bool, no_color: bool):
    """Test and diagnose LLM timing and performance issues."""
    asyncio.run(run_timing_test(verbose=verbose, debug=debug, no_color=no_color))


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--scheme", type=click.Choice(["category", "date", "extension", "date_category"]), default="category", help="Organization scheme")
@click.option("--dest", type=click.Path(), default=None, help="Destination directory (default: same as source)")
@click.option("--execute", is_flag=True, help="Actually organize files (default is dry run)")
@click.option("--recursive", "-r", is_flag=True, help="Process subdirectories")
@click.option("--no-hidden", is_flag=True, help="Skip hidden files")
@click.option("--conflict", type=click.Choice(["skip", "rename"]), default="rename", help="Conflict resolution strategy")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--no-color", is_flag=True, help="Disable colored output")
def organize(
    path: str,
    scheme: str,
    dest: str | None,
    execute: bool,
    recursive: bool,
    no_hidden: bool,
    conflict: str,
    verbose: bool,
    no_color: bool,
):
    """Organize files in a directory by category, date, or extension.

    By default, runs in dry-run mode showing what would be done.
    Use --execute to actually move files.
    """
    asyncio.run(run_organize(
        path=path,
        scheme=scheme,
        dest=dest,
        execute=execute,
        recursive=recursive,
        skip_hidden=no_hidden,
        conflict=conflict,
        verbose=verbose,
        no_color=no_color,
    ))


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--recursive", "-r", is_flag=True, help="Analyze subdirectories")
@click.option("--detailed", "-d", is_flag=True, help="Show detailed file listings")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--no-color", is_flag=True, help="Disable colored output")
def analyze(
    path: str,
    recursive: bool,
    detailed: bool,
    output_json: bool,
    verbose: bool,
    no_color: bool,
):
    """Analyze a directory and show file statistics.

    Shows breakdown by category, top extensions, largest files, etc.
    """
    asyncio.run(run_analyze(
        path=path,
        recursive=recursive,
        detailed=detailed,
        output_json=output_json,
        verbose=verbose,
        no_color=no_color,
    ))


@cli.command()
@click.option("--list", "list_logs", is_flag=True, help="List recent organization operations")
@click.option("--log", type=click.Path(exists=True), default=None, help="Specific log file to undo")
@click.option("--execute", is_flag=True, help="Actually undo (default is dry run)")
@click.option("--limit", type=int, default=10, help="Number of logs to show (with --list)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--no-color", is_flag=True, help="Disable colored output")
def undo(
    list_logs: bool,
    log: str | None,
    execute: bool,
    limit: int,
    verbose: bool,
    no_color: bool,
):
    """Undo a file organization operation.

    By default, shows what would be undone (dry run).
    Use --execute to actually undo the operation.
    """
    asyncio.run(run_undo(
        list_logs=list_logs,
        log_file=log,
        execute=execute,
        limit=limit,
        verbose=verbose,
        no_color=no_color,
    ))


# =============================================================================
# Command implementations
# =============================================================================


async def run_chat_loop(
    verbose: bool = False,
    debug: bool = False,
    no_color: bool = False,
    session_id: str | None = None,
    force_new: bool = False,
) -> None:
    """Run the interactive chat loop.

    Args:
        verbose: Enable verbose output
        debug: Enable debug mode with detailed logging
        no_color: Disable colored output
        session_id: Optional session ID to resume
        force_new: Force creation of new session
    """
    # Setup logging based on flags
    if debug:
        log_level = "DEBUG"
    elif verbose:
        log_level = "INFO"
    else:
        log_level = None  # Use config default

    setup_logging(level=log_level)

    console = get_console(no_color=no_color, verbose=verbose or debug)
    settings = get_settings()

    # Display welcome banner
    console.welcome()
    console.info(f"Using model: {settings.ollama_model}")
    console.info(f"Ollama host: {settings.ollama_host}\n")

    # Register tools
    registry = get_registry()

    # Example tools
    if "echo" not in registry:
        registry.register(EchoTool())
    if "system_info" not in registry:
        registry.register(SystemInfoTool())

    # Filesystem tools
    if "list_directory" not in registry:
        registry.register(ListDirectoryTool())
    if "read_file" not in registry:
        registry.register(ReadFileTool())
    if "analyze_file" not in registry:
        registry.register(AnalyzeFileTool())
    if "analyze_directory" not in registry:
        registry.register(AnalyzeDirectoryTool())
    if "organize_files" not in registry:
        registry.register(OrganizeFilesTool())
    if "copy_file" not in registry:
        registry.register(CopyFileTool())
    if "move_file" not in registry:
        registry.register(MoveFileTool())
    if "create_directory" not in registry:
        registry.register(CreateDirectoryTool())
    if "delete_file" not in registry:
        registry.register(DeleteFileTool())

    # Organization tools
    if "list_organization_logs" not in registry:
        registry.register(ListOrganizationLogsTool())
    if "undo_organization" not in registry:
        registry.register(UndoOrganizationTool())

    console.info(f"Loaded {len(registry)} tool(s)")
    if verbose:
        for tool_name in registry.get_tool_names():
            tool = registry.get(tool_name)
            console.debug(f"  - {tool_name} ({tool.risk_level.value})")

    # Test connection to Ollama
    try:
        async with OllamaClient() as client:
            with console.thinking("Connecting to Ollama..."):
                await client.health_check()

            console.success("Connected to Ollama successfully\n")

            # Register document processing tools (need client for LLM classification)
            if "ocr_extract" not in registry:
                registry.register(OCRTool())

            if "process_document" not in registry:
                # Initialize document processing components
                doc_classifier = DocumentClassifier(client, settings)
                doc_processor = DocumentProcessor(
                    ocr_tool=OCRTool(),
                    classifier=doc_classifier,
                    settings=settings,
                )
                registry.register(ProcessDocumentTool(doc_processor))

            if "process_inbox" not in registry:
                # Reuse or create processor
                if "process_document" in registry:
                    # Get the processor from the existing tool
                    process_doc_tool = registry.get("process_document")
                    doc_processor = process_doc_tool.processor
                else:
                    # Create new processor
                    doc_classifier = DocumentClassifier(client, settings)
                    doc_processor = DocumentProcessor(
                        ocr_tool=OCRTool(),
                        classifier=doc_classifier,
                        settings=settings,
                    )
                registry.register(ProcessInboxTool(doc_processor, settings))

            if verbose:
                console.debug(f"Registered document processing tools")

            # Initialize conversation store
            conversation_store = None
            if settings.auto_save_conversations:
                from aria.memory import ConversationStore
                conversation_store = ConversationStore(settings.conversation_db_path)
                await conversation_store.initialize()
                logger.debug("Conversation store initialized")

            # Create agent
            approval_handler = ApprovalHandler(console)
            agent = Agent(
                client=client,
                registry=registry,
                console=console,
                approval_handler=approval_handler,
                max_iterations=settings.agent_max_iterations,
                conversation_store=conversation_store,
            )

            console.success(f"Agent initialized with {len(registry)} tools\n")

            # Start or resume session
            if conversation_store:
                if force_new or session_id is None:
                    # Create new session
                    active_session_id = await agent.start_session()
                    console.info(f"New session started: [cyan]{active_session_id}[/cyan]")
                else:
                    # Resume existing session
                    try:
                        active_session_id = await agent.start_session(session_id)
                        session = await conversation_store.get_session(active_session_id)
                        console.info(f"Session resumed: [cyan]{active_session_id}[/cyan]")
                        console.info(f"Title: {session.title}")
                        console.info(f"Messages: {session.message_count}")
                    except Exception as e:
                        console.error(f"Failed to resume session: {e}")
                        console.info("Starting new session instead...")
                        active_session_id = await agent.start_session()
                        console.info(f"New session: [cyan]{active_session_id}[/cyan]")
                console.print()  # Blank line

            # Show helpful commands
            console.print("[dim]Commands:[/dim]")
            console.print("[dim]  - Type 'exit' or 'quit' to leave[/dim]")
            console.print("[dim]  - Press Ctrl+C to cancel current request[/dim]")
            console.print("[dim]  - Press Ctrl+D to exit[/dim]\n")

            console.divider()

            # Initialize conversation history
            conversation_history: list[ChatMessage] = []

            # Chat loop
            while True:
                global _shutdown_requested
                _shutdown_requested = False

                try:
                    # Get user input
                    user_input = console.console.input("\n[cyan bold]You:[/cyan bold] ")

                    # Handle empty input
                    if not user_input.strip():
                        continue

                    # Check for exit commands
                    if user_input.strip().lower() in ["exit", "quit", "bye"]:
                        if confirm("Are you sure you want to exit?", default=True):
                            console.success("Goodbye!")
                            break
                        else:
                            continue

                    # Get response from agent
                    try:
                        # Use chat() if we have a conversation store, otherwise run()
                        if conversation_store:
                            # chat() handles history loading and saving automatically
                            response = await agent.chat(user_message=user_input)
                        else:
                            # Fall back to run() with in-memory history
                            response, _ = await agent.run(
                                user_message=user_input,
                                conversation_history=conversation_history,
                            )

                            # Add to in-memory history
                            if response and not _shutdown_requested:
                                conversation_history.append(ChatMessage.user(user_input))
                                conversation_history.append(ChatMessage.assistant(response))

                            # Trim history if too long
                            if len(conversation_history) > settings.aria_max_history:
                                conversation_history = conversation_history[-settings.aria_max_history:]

                        # Display the response
                        if response and not _shutdown_requested:
                            console.assistant_message(response)

                    except OllamaConnectionError as e:
                        console.error("Connection to Ollama lost", exception=e)
                        console.info("Please check if Ollama is still running")
                        break

                    except Exception as e:
                        console.error(f"Error during chat: {e}", exception=e if verbose else None)
                        continue

                except KeyboardInterrupt:
                    # Ctrl+C cancels current request but doesn't exit
                    console.console.print("\n")
                    console.warning("Request cancelled. Type 'exit' to quit.")
                    _shutdown_requested = False
                    continue

                except EOFError:
                    # Ctrl+D exits
                    console.console.print("\n")
                    console.success("Goodbye!")
                    break

    except OllamaConnectionError as e:
        console.error("Failed to connect to Ollama", exception=e)
        console.console.print("\n[yellow]Troubleshooting tips:[/yellow]")
        console.console.print("  1. Make sure Ollama is running")
        console.console.print("  2. Check OLLAMA_HOST in your .env file")
        console.console.print("  3. For WSL: verify host.docker.internal or Windows IP")
        console.console.print(f"\n  Current host: {settings.ollama_host}")
        sys.exit(1)

    except Exception as e:
        console.error(f"Unexpected error: {e}", exception=e if verbose else None)
        sys.exit(1)


async def list_models(verbose: bool = False, debug: bool = False, no_color: bool = False) -> None:
    """List available Ollama models.

    Args:
        verbose: Enable verbose output
        debug: Enable debug mode
        no_color: Disable colored output
    """
    # Setup logging
    if debug:
        setup_logging(level="DEBUG")
    elif verbose:
        setup_logging(level="INFO")

    console = get_console(no_color=no_color, verbose=verbose or debug)

    try:
        async with OllamaClient() as client:
            with console.thinking("Fetching models..."):
                models = await client.list_models()

            if not models:
                console.warning("No models found")
                console.info("Download a model with: ollama pull <model-name>")
                return

            # Format model data for display
            model_data = []
            for model in models:
                name = model.name
                size = f"{model.size_gb:.2f} GB" if model.size_gb else "Unknown"
                modified = (
                    model.modified_at.strftime("%Y-%m-%d %H:%M")
                    if model.modified_at
                    else "Unknown"
                )
                model_data.append((name, size, modified))

            console.show_models(model_data)

    except OllamaConnectionError as e:
        console.error("Failed to connect to Ollama", exception=e)
        sys.exit(1)


async def check_health(verbose: bool = False, debug: bool = False, no_color: bool = False) -> None:
    """Check Ollama connection status.

    Args:
        verbose: Enable verbose output
        debug: Enable debug mode
        no_color: Disable colored output
    """
    # Setup logging
    if debug:
        setup_logging(level="DEBUG")
    elif verbose:
        setup_logging(level="INFO")

    console = get_console(no_color=no_color, verbose=verbose or debug)
    settings = get_settings()

    console.info(f"Checking connection to {settings.ollama_host}...")

    try:
        async with OllamaClient() as client:
            with console.thinking("Testing connection..."):
                is_healthy = await client.health_check()

            if is_healthy:
                console.success("Ollama is running and accessible")

                # Get model info if verbose
                if verbose:
                    models = await client.list_models()
                    console.info(f"Available models: {len(models)}")

                    # Check if default model exists
                    if await client.model_exists(settings.ollama_model):
                        console.success(f"Default model '{settings.ollama_model}' is available")
                    else:
                        console.warning(f"Default model '{settings.ollama_model}' not found")
                        console.info(f"Download with: ollama pull {settings.ollama_model}")

            else:
                console.error("Ollama health check failed")
                sys.exit(1)

    except OllamaConnectionError as e:
        console.error("Failed to connect to Ollama", exception=e)
        console.console.print("\n[yellow]Troubleshooting tips:[/yellow]")
        console.console.print("  1. Make sure Ollama is running")
        console.console.print("  2. Check OLLAMA_HOST in your .env file")
        console.console.print(f"\n  Current host: {settings.ollama_host}")
        sys.exit(1)


def show_config(verbose: bool = False, debug: bool = False, no_color: bool = False) -> None:
    """Show current configuration.

    Args:
        verbose: Enable verbose output
        debug: Enable debug mode
        no_color: Disable colored output
    """
    # Setup logging
    if debug:
        setup_logging(level="DEBUG")
    elif verbose:
        setup_logging(level="INFO")

    console = get_console(no_color=no_color, verbose=verbose or debug)
    settings = get_settings()

    config_dict = settings.model_dump_safe()

    console.show_config(config_dict)

    if verbose:
        console.console.print("\n[dim]Configuration loaded from:[/dim]")
        console.console.print(f"  - Environment variables")
        console.console.print(f"  - .env file (if present)")


async def run_timing_test(verbose: bool = False, debug: bool = False, no_color: bool = False) -> None:
    """Run timing tests to diagnose performance issues.

    Args:
        verbose: Enable verbose output
        debug: Enable debug mode
        no_color: Disable colored output
    """
    import time

    # Setup logging
    setup_logging(level="DEBUG" if debug else "INFO")

    console = get_console(no_color=no_color, verbose=True)
    settings = get_settings()

    console.console.print("[bold cyan]ARIA Timing Diagnostic Tool[/bold cyan]\n")
    console.info(f"Testing Ollama at: {settings.ollama_host}")
    console.info(f"Model: {settings.ollama_model}\n")

    try:
        async with OllamaClient() as client:
            # Test 1: Health check
            console.console.print("[yellow]Test 1: Health Check[/yellow]")
            start = time.perf_counter()
            await client.health_check()
            elapsed = time.perf_counter() - start
            console.success(f"Health check: {elapsed:.3f}s\n")

            # Test 2: Simple chat (no tools)
            console.console.print("[yellow]Test 2: Simple Chat (no tools)[/yellow]")
            messages = [ChatMessage.user("What is 2+2?")]
            start = time.perf_counter()
            response = await client.chat(messages=messages)
            elapsed = time.perf_counter() - start
            console.success(f"Simple chat: {elapsed:.3f}s")
            console.console.print(f"[dim]Response: {response.message.content[:100]}...[/dim]\n")

            # Test 3: Chat with varying numbers of tools
            from aria.llm.models import ToolDefinition

            # Create a realistic time tool that the model would want to call
            def create_time_tool() -> ToolDefinition:
                return ToolDefinition.from_schema(
                    name="get_current_time",
                    description="Get the current date and time in a specified timezone",
                    parameters={
                        "type": "object",
                        "properties": {
                            "timezone": {
                                "type": "string",
                                "description": "Timezone name (e.g., 'America/New_York', 'UTC')",
                                "default": "UTC",
                            },
                            "format": {
                                "type": "string",
                                "description": "Time format ('12h' or '24h')",
                                "enum": ["12h", "24h"],
                                "default": "24h",
                            },
                        },
                        "required": [],
                    },
                )

            # Create dummy tools for filling out the tool list
            def create_dummy_tool(i: int) -> ToolDefinition:
                return ToolDefinition.from_schema(
                    name=f"calculator_{i}",
                    description=f"Perform mathematical calculation operation {i}",
                    parameters={
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "Mathematical expression to evaluate"},
                        },
                        "required": ["expression"],
                    },
                )

            # Test with different tool counts
            for tool_count in [1, 3, 5, 9]:
                console.console.print(f"[yellow]Test 3.{tool_count}: Chat with {tool_count} tools[/yellow]")
                # Always include the time tool + additional dummy tools
                tools = [create_time_tool()] + [create_dummy_tool(i) for i in range(tool_count - 1)]
                messages = [ChatMessage.user("What time is it?")]

                start = time.perf_counter()
                response = await client.chat_with_tools(messages=messages, tools=tools)
                elapsed = time.perf_counter() - start

                console.success(f"Chat with {tool_count} tools: {elapsed:.3f}s")
                if response.has_tool_calls:
                    console.console.print(f"[dim]✓ Tool calls: {len(response.message.tool_calls)}[/dim]")
                    # Show which tools were called
                    for tc in response.message.tool_calls:
                        console.console.print(f"[dim]  → {tc.function.name}({tc.function.arguments})[/dim]")
                else:
                    console.console.print(f"[dim]✗ No tool calls - Response: {response.message.content[:100]}...[/dim]")
                console.console.print()

            # Test 4: Actual ARIA agent with real tools
            console.console.print("[yellow]Test 4: Full Agent with Real Tools[/yellow]")
            registry = get_registry()

            # Register example tools
            if "echo" not in registry:
                registry.register(EchoTool())
            if "system_info" not in registry:
                registry.register(SystemInfoTool())

            # Register filesystem tools
            for tool_class in [ListDirectoryTool, ReadFileTool, CopyFileTool]:
                tool = tool_class()
                if tool.name not in registry:
                    registry.register(tool)

            tools = registry.get_tools_for_ollama()
            console.info(f"Testing with {len(tools)} real tools")

            messages = [ChatMessage.user("What time is it and what operating system are we running on?")]
            start = time.perf_counter()
            response = await client.chat_with_tools(messages=messages, tools=tools)
            elapsed = time.perf_counter() - start

            console.success(f"Chat with {len(tools)} real tools: {elapsed:.3f}s")
            if response.has_tool_calls:
                console.console.print(f"[dim]✓ Tool calls: {len(response.message.tool_calls)}[/dim]")
                for tc in response.message.tool_calls:
                    console.console.print(f"[dim]  → {tc.function.name}[/dim]")
            else:
                console.console.print(f"[dim]✗ No tool calls - Response: {response.message.content[:100]}...[/dim]")
            console.console.print()

            # Summary
            console.console.print("[bold green]Timing Test Complete[/bold green]")
            console.console.print("\n[bold]Key Insights:[/bold]")
            console.console.print("  • Compare timing across different tool counts")
            console.console.print("  • Large tool schemas may cause slow LLM processing")
            console.console.print("  • Debug logs (--debug) show detailed timing for each component")
            console.console.print("  • Tool descriptions matter: model won't call tools that don't match the query")
            console.console.print("\n[bold yellow]Note:[/bold yellow] qwen3:30b-a3b includes 'thinking' in responses even with think=False")

    except OllamaConnectionError as e:
        console.error("Failed to connect to Ollama", exception=e)
        sys.exit(1)
    except Exception as e:
        console.error(f"Timing test failed: {e}", exception=e)
        sys.exit(1)


async def run_organize(
    path: str,
    scheme: str,
    dest: str | None,
    execute: bool,
    recursive: bool,
    skip_hidden: bool,
    conflict: str,
    verbose: bool,
    no_color: bool,
) -> None:
    """Run the organize command."""
    import json
    from rich.table import Table
    from rich.panel import Panel
    from aria.tools.filesystem import OrganizeFilesTool, OrganizeFilesParams

    console = get_console(no_color=no_color, verbose=verbose)

    try:
        # Create organize tool
        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=path,
            destination_path=dest,
            organization_scheme=scheme,
            dry_run=not execute,
            recursive=recursive,
            skip_hidden=skip_hidden,
            conflict_resolution=conflict,
        )

        # Show header
        mode = "[yellow]DRY RUN[/yellow]" if not execute else "[red]EXECUTE[/red]"
        console.console.print(f"\n[bold cyan]File Organization[/bold cyan] {mode}\n")
        console.console.print(f"Source: {path}")
        if dest:
            console.console.print(f"Destination: {dest}")
        console.console.print(f"Scheme: {scheme}")
        console.console.print(f"Conflict resolution: {conflict}")
        console.console.print()

        # Run organization
        with console.thinking("Analyzing files..."):
            result = await tool.execute(params)

        if not result.success:
            console.error(f"Organization failed: {result.error}")
            sys.exit(1)

        data = result.data

        # Show summary
        summary = data["summary"]

        if not execute:
            # Dry run - show what would be done
            console.console.print(Panel(
                f"[bold]Summary[/bold]\n\n"
                f"Total files: {summary['total_files']}\n"
                f"To move: {summary.get('to_move', 0)}\n"
                f"To skip: {summary.get('to_skip', 0)}",
                title="Dry Run Results",
                border_style="yellow",
            ))

            # Show by destination
            if summary.get("by_destination"):
                console.console.print("\n[bold]Planned Organization:[/bold]")
                table = Table(show_header=True, header_style="bold cyan")
                table.add_column("Destination", style="cyan")
                table.add_column("Files", justify="right", style="green")

                for dest_name, count in sorted(summary["by_destination"].items()):
                    table.add_row(dest_name, str(count))

                console.console.print(table)

            if verbose and data.get("operations"):
                console.console.print("\n[bold]Operations (first 10):[/bold]")
                for op in data["operations"][:10]:
                    if op["action"] == "move":
                        src_name = Path(op["source"]).name
                        dest_rel = Path(op["destination"]).relative_to(path if not dest else dest)
                        console.console.print(f"  • {src_name} → {dest_rel}")

                if len(data["operations"]) > 10:
                    console.console.print(f"  ... and {len(data['operations']) - 10} more")

            console.console.print("\n[dim]Run with --execute to actually move files[/dim]")

        else:
            # Actual execution - show results
            console.console.print(Panel(
                f"[bold]Summary[/bold]\n\n"
                f"Total files: {summary['total_files']}\n"
                f"Completed: {summary.get('completed', 0)} ✓\n"
                f"Errors: {summary.get('errors', 0)} ✗",
                title="Organization Complete",
                border_style="green" if summary.get("errors", 0) == 0 else "yellow",
            ))

            # Show by destination
            if summary.get("by_destination"):
                console.console.print("\n[bold]Final Organization:[/bold]")
                table = Table(show_header=True, header_style="bold cyan")
                table.add_column("Destination", style="cyan")
                table.add_column("Files", justify="right", style="green")

                for dest_name, count in sorted(summary["by_destination"].items()):
                    table.add_row(dest_name, str(count))

                console.console.print(table)

            # Show log file
            if data.get("log_file"):
                console.console.print(f"\n[dim]Log saved: {data['log_file']}[/dim]")
                console.console.print("[dim]Use 'aria undo' to reverse this operation[/dim]")

    except Exception as e:
        console.error(f"Organize command failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


async def run_analyze(
    path: str,
    recursive: bool,
    detailed: bool,
    output_json: bool,
    verbose: bool,
    no_color: bool,
) -> None:
    """Run the analyze command."""
    import json
    from rich.table import Table
    from rich.panel import Panel
    from aria.tools.filesystem import AnalyzeDirectoryTool, AnalyzeDirectoryParams

    console = get_console(no_color=no_color, verbose=verbose)

    try:
        # Create analyze tool
        tool = AnalyzeDirectoryTool()
        params = AnalyzeDirectoryParams(
            path=path,
            recursive=recursive,
            include_hidden=False,
        )

        # Run analysis
        with console.thinking("Analyzing directory..."):
            result = await tool.execute(params)

        if not result.success:
            console.error(f"Analysis failed: {result.error}")
            sys.exit(1)

        data = result.data

        # Output as JSON if requested
        if output_json:
            print(json.dumps(data, indent=2))
            return

        # Show summary
        console.console.print(f"\n[bold cyan]Directory Analysis[/bold cyan]\n")
        console.console.print(f"Path: {data['path']}")
        console.console.print()

        console.console.print(Panel(
            f"[bold]Summary[/bold]\n\n"
            f"Total files: {data['total_files']}\n"
            f"Total size: {data['total_size_human']}\n"
            f"Permission errors: {data.get('permission_errors', 0)}",
            title="Statistics",
            border_style="cyan",
        ))

        # Show categories
        if data.get("categories"):
            console.console.print("\n[bold]By Category:[/bold]")
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Category", style="cyan")
            table.add_column("Files", justify="right")
            table.add_column("Size", justify="right", style="green")
            table.add_column("Extensions", style="dim")

            for category, info in sorted(data["categories"].items()):
                extensions = ", ".join(sorted(info["extensions"])[:5])
                if len(info["extensions"]) > 5:
                    extensions += f" +{len(info['extensions']) - 5}"

                table.add_row(
                    category.title(),
                    str(info["count"]),
                    info["size_human"],
                    extensions if extensions else "-",
                )

            console.console.print(table)

        # Show top extensions
        if data.get("by_extension") and detailed:
            console.console.print("\n[bold]Top Extensions:[/bold]")
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Extension", style="cyan")
            table.add_column("Files", justify="right")
            table.add_column("Size", justify="right", style="green")

            for ext, info in list(data["by_extension"].items())[:10]:
                from aria.tools.filesystem.analyze_file import format_size
                table.add_row(
                    ext,
                    str(info["count"]),
                    format_size(info["size_bytes"]),
                )

            console.console.print(table)

        # Show largest files
        if data.get("largest_files") and detailed:
            console.console.print("\n[bold]Largest Files:[/bold]")
            for i, file_info in enumerate(data["largest_files"][:10], 1):
                console.console.print(
                    f"  {i:2d}. {file_info['name']:40s} {file_info['size_human']:>10s} ({file_info['category']})"
                )

        # Show potential duplicates
        if data.get("potential_duplicates") and len(data["potential_duplicates"]) > 0 and detailed:
            console.console.print(f"\n[bold yellow]Potential Duplicates:[/bold yellow]")
            for dup in data["potential_duplicates"][:5]:
                console.console.print(f"\n  {dup['name']} ({dup['size_human']}) - {dup['count']} copies:")
                for loc in dup["locations"][:3]:
                    console.console.print(f"    • {loc}")

    except Exception as e:
        console.error(f"Analyze command failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


async def run_undo(
    list_logs: bool,
    log_file: str | None,
    execute: bool,
    limit: int,
    verbose: bool,
    no_color: bool,
) -> None:
    """Run the undo command."""
    from rich.table import Table
    from rich.panel import Panel
    from datetime import datetime
    from aria.tools.organization import (
        UndoOrganizationTool,
        UndoOrganizationParams,
        ListOrganizationLogsTool,
        ListOrganizationLogsParams,
    )

    console = get_console(no_color=no_color, verbose=verbose)

    try:
        # List logs if requested
        if list_logs:
            list_tool = ListOrganizationLogsTool()
            list_params = ListOrganizationLogsParams(limit=limit)

            with console.thinking("Loading organization logs..."):
                result = await list_tool.execute(list_params)

            if not result.success:
                console.error(f"Failed to list logs: {result.error}")
                sys.exit(1)

            data = result.data

            if data["total_logs"] == 0:
                console.console.print("[yellow]No organization logs found[/yellow]")
                return

            console.console.print(f"\n[bold cyan]Organization History[/bold cyan]\n")
            console.console.print(f"Showing {data['showing']} of {data['total_logs']} logs\n")

            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("#", justify="right", style="dim")
            table.add_column("Date/Time", style="cyan")
            table.add_column("Source", style="green")
            table.add_column("Scheme")
            table.add_column("Files", justify="right")
            table.add_column("Status")

            for i, log in enumerate(data["logs"], 1):
                # Parse timestamp
                try:
                    dt = datetime.fromisoformat(log["timestamp"])
                    time_str = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    time_str = log["timestamp"][:16]

                # Status
                if log["failed"] > 0:
                    status = f"✓ {log['completed']} ✗ {log['failed']}"
                else:
                    status = f"✓ {log['completed']}"

                # Truncate source path
                source = log["source_path"]
                if len(source) > 40:
                    source = "..." + source[-37:]

                table.add_row(
                    str(i),
                    time_str,
                    source,
                    log["scheme"],
                    str(log["total_files"]),
                    status,
                )

            console.console.print(table)
            console.console.print("\n[dim]Use 'aria undo --log <log_file> --execute' to undo a specific operation[/dim]")
            return

        # Undo operation
        undo_tool = UndoOrganizationTool()
        undo_params = UndoOrganizationParams(
            log_file=log_file,
            dry_run=not execute,
        )

        mode = "[yellow]DRY RUN[/yellow]" if not execute else "[red]EXECUTE[/red]"
        console.console.print(f"\n[bold cyan]Undo Organization[/bold cyan] {mode}\n")

        with console.thinking("Loading operation log..."):
            result = await undo_tool.execute(undo_params)

        if not result.success:
            console.error(f"Undo failed: {result.error}")
            sys.exit(1)

        data = result.data
        summary = data["summary"]

        # Show what would be or was undone
        console.console.print(f"Original source: {data['original_source']}")
        console.console.print(f"Organization scheme: {data['original_scheme']}")
        console.console.print()

        if not execute:
            # Dry run
            console.console.print(Panel(
                f"[bold]Summary[/bold]\n\n"
                f"Can undo: {summary['can_undo']}\n"
                f"Cannot undo: {summary['cannot_undo']}",
                title="Undo Preview",
                border_style="yellow",
            ))

            # Show what can be undone
            if data.get("can_undo") and verbose:
                console.console.print("\n[bold]Can Undo (first 10):[/bold]")
                for op in data["can_undo"][:10]:
                    src_name = Path(op["source"]).name
                    console.console.print(f"  • {src_name}")

            # Show what cannot be undone
            if data.get("cannot_undo"):
                console.console.print("\n[bold yellow]Cannot Undo:[/bold yellow]")
                for op in data["cannot_undo"][:5]:
                    src_name = Path(op["source"]).name
                    console.console.print(f"  • {src_name}: {op['reason']}")

            if summary["can_undo"] > 0:
                console.console.print("\n[dim]Run with --execute to actually undo the organization[/dim]")

        else:
            # Actual undo
            console.console.print(Panel(
                f"[bold]Summary[/bold]\n\n"
                f"Undone: {summary['undone']} ✓\n"
                f"Failed: {summary['failed']} ✗\n"
                f"Cannot undo: {summary['cannot_undo']} ⊘",
                title="Undo Complete",
                border_style="green" if summary['failed'] == 0 else "yellow",
            ))

            # Show failures if any
            if data.get("failed"):
                console.console.print("\n[bold red]Failed:[/bold red]")
                for op in data["failed"]:
                    src_name = Path(op["source"]).name
                    console.console.print(f"  • {src_name}: {op.get('error_message', 'Unknown error')}")

    except Exception as e:
        console.error(f"Undo command failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main() -> NoReturn:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
