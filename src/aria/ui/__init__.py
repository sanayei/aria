"""User interface components for ARIA.

This module provides the CLI interface, console output, formatting utilities,
and user input handling for the ARIA assistant.
"""

from aria.ui.console import ARIAConsole, get_console, ARIA_THEME
from aria.ui.prompts import (
    confirm,
    prompt,
    prompt_validated,
    select,
    multiselect,
    prompt_int,
    prompt_float,
)
from aria.ui.formatters import (
    format_markdown,
    format_code,
    format_json,
    format_panel,
    truncate_text,
    truncate_with_preview,
    format_timestamp,
    format_duration,
    format_size,
    format_list,
    format_dict_table,
    format_error_trace,
    wrap_text,
    format_percentage,
    strip_ansi,
)

__all__ = [
    # Console
    "ARIAConsole",
    "get_console",
    "ARIA_THEME",
    # Prompts
    "confirm",
    "prompt",
    "prompt_validated",
    "select",
    "multiselect",
    "prompt_int",
    "prompt_float",
    # Formatters
    "format_markdown",
    "format_code",
    "format_json",
    "format_panel",
    "truncate_text",
    "truncate_with_preview",
    "format_timestamp",
    "format_duration",
    "format_size",
    "format_list",
    "format_dict_table",
    "format_error_trace",
    "wrap_text",
    "format_percentage",
    "strip_ansi",
]
