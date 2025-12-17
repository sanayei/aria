"""Rich console wrapper for ARIA with consistent styling and theming.

This module provides the ARIAConsole class which wraps Rich Console with
ARIA-specific styling, themes, and convenience methods for common outputs.
"""

from typing import Any
from contextlib import contextmanager

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner
from rich.theme import Theme

from aria import __version__


# ARIA color scheme
ARIA_THEME = Theme({
    # Primary colors
    "aria.primary": "cyan",
    "aria.secondary": "blue",
    "aria.accent": "magenta",

    # Status colors
    "aria.success": "green",
    "aria.error": "red bold",
    "aria.warning": "yellow",
    "aria.info": "blue",

    # Message roles
    "aria.user": "cyan bold",
    "aria.assistant": "green bold",
    "aria.system": "dim",
    "aria.tool": "magenta",

    # Special elements
    "aria.thinking": "yellow italic",
    "aria.code": "cyan",
    "aria.header": "cyan bold",
    "aria.footer": "dim",
})


class ARIAConsole:
    """Enhanced Rich console with ARIA-specific styling.

    This class wraps Rich Console and provides convenience methods for
    displaying different types of messages with consistent styling.

    Attributes:
        console: The underlying Rich Console instance
    """

    def __init__(self, no_color: bool = False, verbose: bool = False):
        """Initialize the ARIA console.

        Args:
            no_color: Disable colored output
            verbose: Enable verbose output
        """
        self.console = Console(
            theme=ARIA_THEME,
            highlight=False,  # Disable auto-highlighting to avoid conflicts
            force_terminal=not no_color,
        )
        self.verbose = verbose
        self.no_color = no_color

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print to console (passthrough to Rich Console).

        Args:
            *args: Arguments to print
            **kwargs: Keyword arguments for console.print
        """
        self.console.print(*args, **kwargs)

    def welcome(self) -> None:
        """Display the ARIA welcome banner."""
        banner = Text()
        banner.append("╔═══════════════════════════════════════════════════════════╗\n", style="aria.primary")
        banner.append("║                                                           ║\n", style="aria.primary")
        banner.append("║           ", style="aria.primary")
        banner.append("ARIA", style="aria.primary bold")
        banner.append(" - AI Research & Intelligence Assistant        ║\n", style="aria.primary")
        banner.append("║                                                           ║\n", style="aria.primary")
        banner.append("╚═══════════════════════════════════════════════════════════╝", style="aria.primary")

        self.console.print(banner)
        self.console.print(
            f"Version {__version__} | Local-First AI Assistant\n",
            style="aria.footer",
        )

    def user_message(self, text: str) -> None:
        """Display a user message.

        Args:
            text: The user's message text
        """
        self.console.print(f"You: {text}", style="aria.user")

    def assistant_message(self, text: str, stream: bool = False) -> None:
        """Display an assistant message.

        Args:
            text: The assistant's message text
            stream: Whether this is part of a streaming response
        """
        if stream:
            # For streaming, just print the chunk without prefix
            self.console.print(text, end="", style="aria.assistant")
        else:
            # For complete messages, show with prefix
            self.console.print("\nARIA: ", style="aria.assistant", end="")

            # Try to render as markdown if it looks like markdown
            if self._looks_like_markdown(text):
                self.console.print(Markdown(text))
            else:
                self.console.print(text, style="aria.assistant")

    def assistant_message_start(self) -> None:
        """Display the start of an assistant message (for streaming)."""
        self.console.print("\nARIA: ", style="aria.assistant", end="")

    def assistant_message_end(self) -> None:
        """Display the end of an assistant message (for streaming)."""
        self.console.print()  # New line after streaming

    def tool_call(self, name: str, args: dict[str, Any]) -> None:
        """Display a tool call.

        Args:
            name: Name of the tool being called
            args: Arguments passed to the tool
        """
        self.console.print(
            f"🔧 Calling tool: [bold]{name}[/bold]",
            style="aria.tool",
        )
        if args and self.verbose:
            import json
            args_json = json.dumps(args, indent=2)
            self.console.print(
                Syntax(args_json, "json", theme="monokai", padding=1),
            )

    def tool_result(self, name: str, result: str, error: bool = False) -> None:
        """Display a tool result.

        Args:
            name: Name of the tool that was called
            result: Result from the tool
            error: Whether the result is an error
        """
        style = "aria.error" if error else "aria.tool"
        icon = "❌" if error else "✓"

        self.console.print(
            f"{icon} Tool result: [bold]{name}[/bold]",
            style=style,
        )

        if result and self.verbose:
            # Truncate very long results
            if len(result) > 500:
                result = result[:500] + "... (truncated)"

            self.console.print(
                Panel(result, border_style=style, padding=(0, 1)),
            )

    def error(self, message: str, exception: Exception | None = None) -> None:
        """Display an error message.

        Args:
            message: Error message
            exception: Optional exception object
        """
        self.console.print(f"✗ Error: {message}", style="aria.error")

        if exception and self.verbose:
            self.console.print_exception(show_locals=False)

    def warning(self, message: str) -> None:
        """Display a warning message.

        Args:
            message: Warning message
        """
        self.console.print(f"⚠ Warning: {message}", style="aria.warning")

    def success(self, message: str) -> None:
        """Display a success message.

        Args:
            message: Success message
        """
        self.console.print(f"✓ {message}", style="aria.success")

    def info(self, message: str) -> None:
        """Display an informational message.

        Args:
            message: Info message
        """
        self.console.print(f"ℹ {message}", style="aria.info")

    def debug(self, message: str) -> None:
        """Display a debug message (only in verbose mode).

        Args:
            message: Debug message
        """
        if self.verbose:
            self.console.print(f"[DEBUG] {message}", style="dim")

    @contextmanager
    def thinking(self, message: str = "Thinking..."):
        """Context manager for showing a thinking spinner.

        Args:
            message: Message to show while thinking

        Yields:
            The spinner object
        """
        with self.console.status(
            f"[aria.thinking]{message}[/aria.thinking]",
            spinner="dots",
        ) as status:
            yield status

    def show_models(self, models: list[tuple[str, str, str]]) -> None:
        """Display a table of available models.

        Args:
            models: List of (name, size, modified) tuples
        """
        table = Table(title="Available Ollama Models", show_header=True)
        table.add_column("Model Name", style="aria.primary")
        table.add_column("Size", style="aria.info", justify="right")
        table.add_column("Last Modified", style="aria.footer")

        for name, size, modified in models:
            table.add_row(name, size, modified)

        self.console.print(table)

    def show_config(self, config_dict: dict[str, Any]) -> None:
        """Display configuration settings.

        Args:
            config_dict: Dictionary of configuration settings
        """
        table = Table(title="ARIA Configuration", show_header=True)
        table.add_column("Setting", style="aria.primary")
        table.add_column("Value", style="aria.info")

        for key, value in config_dict.items():
            table.add_row(key, str(value))

        self.console.print(table)

    def divider(self, title: str | None = None) -> None:
        """Print a divider line.

        Args:
            title: Optional title for the divider
        """
        if title:
            self.console.rule(f"[aria.header]{title}[/aria.header]")
        else:
            self.console.rule(style="aria.footer")

    def _looks_like_markdown(self, text: str) -> bool:
        """Check if text appears to contain markdown formatting.

        Args:
            text: Text to check

        Returns:
            bool: True if text looks like markdown
        """
        # Simple heuristic: check for common markdown patterns
        markdown_indicators = [
            "# ",      # Headers
            "## ",
            "* ",      # Lists
            "- ",
            "```",     # Code blocks
            "[",       # Links
            "](",
            "**",      # Bold
            "__",
            "*",       # Italic (but not at start to avoid false positives)
        ]

        return any(indicator in text for indicator in markdown_indicators)

    def clear(self) -> None:
        """Clear the console screen."""
        self.console.clear()


# Global console instance
_console: ARIAConsole | None = None


def get_console(no_color: bool = False, verbose: bool = False) -> ARIAConsole:
    """Get the global ARIA console instance.

    Args:
        no_color: Disable colored output
        verbose: Enable verbose output

    Returns:
        ARIAConsole: The global console instance
    """
    global _console
    if _console is None:
        _console = ARIAConsole(no_color=no_color, verbose=verbose)
    return _console
