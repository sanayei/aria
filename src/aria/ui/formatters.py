"""Output formatting utilities for ARIA.

This module provides functions for formatting various types of content
for display in the terminal, including markdown, code blocks, JSON,
timestamps, and more.
"""

import json
from datetime import datetime, timedelta
from typing import Any

from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text


def format_markdown(text: str) -> Markdown:
    """Format text as markdown.

    Args:
        text: Text with markdown formatting

    Returns:
        Markdown: Rich Markdown object
    """
    return Markdown(text)


def format_code(
    code: str,
    language: str = "python",
    line_numbers: bool = False,
    theme: str = "monokai",
) -> Syntax:
    """Format code with syntax highlighting.

    Args:
        code: Code to format
        language: Programming language for syntax highlighting
        line_numbers: Whether to show line numbers
        theme: Color theme for syntax highlighting

    Returns:
        Syntax: Rich Syntax object
    """
    return Syntax(
        code,
        language,
        theme=theme,
        line_numbers=line_numbers,
        word_wrap=True,
    )


def format_json(data: dict[str, Any] | list[Any], indent: int = 2) -> Syntax:
    """Format JSON data with syntax highlighting.

    Args:
        data: Dictionary or list to format as JSON
        indent: Indentation level

    Returns:
        Syntax: Rich Syntax object with JSON highlighting
    """
    json_str = json.dumps(data, indent=indent, ensure_ascii=False)
    return Syntax(json_str, "json", theme="monokai", word_wrap=True)


def format_panel(
    content: str,
    title: str | None = None,
    border_style: str = "blue",
    padding: int | tuple[int, int] = 1,
) -> Panel:
    """Format content in a panel.

    Args:
        content: Content to display in the panel
        title: Optional panel title
        border_style: Style for the panel border
        padding: Padding (int or (vertical, horizontal))

    Returns:
        Panel: Rich Panel object
    """
    return Panel(
        content,
        title=title,
        border_style=border_style,
        padding=padding,
    )


def truncate_text(
    text: str,
    max_length: int = 500,
    suffix: str = "... (truncated)",
) -> str:
    """Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated

    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def truncate_with_preview(
    text: str,
    max_length: int = 500,
    show_more_text: str = "... [dim](use --verbose to see full output)[/dim]",
) -> str:
    """Truncate text with a preview indicator.

    Args:
        text: Text to truncate
        max_length: Maximum length
        show_more_text: Text to show when truncated

    Returns:
        str: Truncated text with preview indicator
    """
    if len(text) <= max_length:
        return text

    return text[:max_length] + "\n" + show_more_text


def format_timestamp(
    dt: datetime | None = None,
    format: str = "%Y-%m-%d %H:%M:%S",
) -> str:
    """Format a timestamp.

    Args:
        dt: Datetime object (uses current time if None)
        format: strftime format string

    Returns:
        str: Formatted timestamp
    """
    if dt is None:
        dt = datetime.now()

    return dt.strftime(format)


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to human-readable form.

    Args:
        seconds: Duration in seconds

    Returns:
        str: Human-readable duration (e.g., "2.5s", "1m 30s", "1h 5m")
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_size(bytes: int) -> str:
    """Format file size in bytes to human-readable form.

    Args:
        bytes: Size in bytes

    Returns:
        str: Human-readable size (e.g., "1.5 KB", "2.3 MB", "1.2 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0

    return f"{bytes:.1f} PB"


def format_list(
    items: list[str],
    style: str = "bullet",
    indent: int = 2,
) -> str:
    """Format a list of items.

    Args:
        items: List of items to format
        style: List style ("bullet", "numbered", "dash")
        indent: Indentation level

    Returns:
        str: Formatted list
    """
    if style == "bullet":
        prefix = "•"
    elif style == "numbered":
        return "\n".join(
            f"{' ' * indent}{i + 1}. {item}" for i, item in enumerate(items)
        )
    else:  # dash
        prefix = "-"

    return "\n".join(f"{' ' * indent}{prefix} {item}" for item in items)


def format_dict_table(
    data: dict[str, Any],
    key_style: str = "cyan",
    value_style: str = "white",
) -> str:
    """Format a dictionary as a simple key-value table.

    Args:
        data: Dictionary to format
        key_style: Style for keys
        value_style: Style for values

    Returns:
        str: Formatted table as string
    """
    lines = []
    max_key_length = max(len(str(k)) for k in data.keys()) if data else 0

    for key, value in data.items():
        key_str = str(key).ljust(max_key_length)
        lines.append(f"[{key_style}]{key_str}[/{key_style}] : [{value_style}]{value}[/{value_style}]")

    return "\n".join(lines)


def format_error_trace(
    error: Exception,
    include_type: bool = True,
) -> str:
    """Format an error message.

    Args:
        error: Exception object
        include_type: Whether to include the exception type

    Returns:
        str: Formatted error message
    """
    if include_type:
        return f"{type(error).__name__}: {str(error)}"
    else:
        return str(error)


def wrap_text(
    text: str,
    width: int = 80,
    indent: int = 0,
) -> str:
    """Wrap text to a specified width.

    Args:
        text: Text to wrap
        width: Maximum width
        indent: Indentation for wrapped lines

    Returns:
        str: Wrapped text
    """
    import textwrap

    return textwrap.fill(
        text,
        width=width,
        initial_indent=" " * indent,
        subsequent_indent=" " * indent,
    )


def format_percentage(
    value: float,
    decimal_places: int = 1,
) -> str:
    """Format a value as a percentage.

    Args:
        value: Value to format (0.0 to 1.0)
        decimal_places: Number of decimal places

    Returns:
        str: Formatted percentage (e.g., "75.5%")
    """
    return f"{value * 100:.{decimal_places}f}%"


def highlight_keywords(
    text: str,
    keywords: list[str],
    style: str = "bold yellow",
) -> str:
    """Highlight keywords in text.

    Args:
        text: Text to process
        keywords: List of keywords to highlight
        style: Rich style for highlighting

    Returns:
        str: Text with keywords highlighted
    """
    result = text
    for keyword in keywords:
        # Case-insensitive replacement
        import re
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        result = pattern.sub(f"[{style}]{keyword}[/{style}]", result)

    return result


def strip_ansi(text: str) -> str:
    """Strip ANSI color codes from text.

    Args:
        text: Text with ANSI codes

    Returns:
        str: Text without ANSI codes
    """
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)
