"""User input handling for ARIA.

This module provides functions for getting user input with proper
formatting and validation, including confirmations, prompts, and selections.
"""

from typing import Callable

from rich.console import Console
from rich.prompt import Prompt, Confirm


# Console instance for prompts
_prompt_console = Console()


def confirm(
    message: str,
    default: bool = False,
    console: Console | None = None,
) -> bool:
    """Ask the user for yes/no confirmation.

    Args:
        message: The question to ask
        default: Default value if user just presses Enter
        console: Optional console instance (uses default if None)

    Returns:
        bool: True if user confirmed, False otherwise
    """
    console = console or _prompt_console

    return Confirm.ask(
        f"[yellow]?[/yellow] {message}",
        default=default,
        console=console,
    )


def prompt(
    message: str,
    default: str = "",
    password: bool = False,
    console: Console | None = None,
) -> str:
    """Get text input from the user.

    Args:
        message: Prompt message
        default: Default value if user just presses Enter
        password: Whether to hide input (for passwords)
        console: Optional console instance (uses default if None)

    Returns:
        str: User's input
    """
    console = console or _prompt_console

    return Prompt.ask(
        f"[cyan]>[/cyan] {message}",
        default=default,
        password=password,
        console=console,
    )


def prompt_validated(
    message: str,
    validator: Callable[[str], bool],
    error_message: str = "Invalid input, please try again.",
    default: str = "",
    console: Console | None = None,
) -> str:
    """Get validated text input from the user.

    Args:
        message: Prompt message
        validator: Function that returns True if input is valid
        error_message: Message to show on validation failure
        default: Default value if user just presses Enter
        console: Optional console instance (uses default if None)

    Returns:
        str: Validated user input
    """
    console = console or _prompt_console

    while True:
        value = prompt(message, default=default, console=console)

        if validator(value):
            return value
        else:
            console.print(f"[red]✗[/red] {error_message}")


def select(
    message: str,
    options: list[str],
    default: str | None = None,
    console: Console | None = None,
) -> str:
    """Let the user select from a list of options.

    Args:
        message: Prompt message
        options: List of options to choose from
        default: Default option if user just presses Enter
        console: Optional console instance (uses default if None)

    Returns:
        str: Selected option

    Example:
        >>> choice = select("Pick a color", ["red", "green", "blue"])
    """
    console = console or _prompt_console

    # Show the options
    console.print(f"\n[cyan]{message}[/cyan]")
    for i, option in enumerate(options, 1):
        console.print(f"  {i}. {option}")

    # Get the selection
    while True:
        choice = Prompt.ask(
            "\n[cyan]>[/cyan] Enter choice (number or name)",
            default=str(default) if default else None,
            console=console,
        )

        # Try to parse as number
        try:
            index = int(choice) - 1
            if 0 <= index < len(options):
                return options[index]
        except ValueError:
            pass

        # Try to match by name
        for option in options:
            if option.lower() == choice.lower():
                return option

        console.print("[red]✗[/red] Invalid choice, please try again.")


def multiselect(
    message: str,
    options: list[str],
    console: Console | None = None,
) -> list[str]:
    """Let the user select multiple items from a list.

    Args:
        message: Prompt message
        options: List of options to choose from
        console: Optional console instance (uses default if None)

    Returns:
        list[str]: List of selected options

    Example:
        >>> choices = multiselect("Pick colors", ["red", "green", "blue"])
    """
    console = console or _prompt_console

    # Show the options
    console.print(f"\n[cyan]{message}[/cyan]")
    for i, option in enumerate(options, 1):
        console.print(f"  {i}. {option}")

    console.print("\n[dim]Enter numbers or names separated by commas, or 'all' for all options[/dim]")

    while True:
        choice = Prompt.ask(
            "\n[cyan]>[/cyan] Enter choices",
            console=console,
        )

        # Handle 'all'
        if choice.lower() == "all":
            return options.copy()

        # Parse the input
        selected = []
        parts = [p.strip() for p in choice.split(",")]

        valid = True
        for part in parts:
            # Try to parse as number
            try:
                index = int(part) - 1
                if 0 <= index < len(options):
                    selected.append(options[index])
                    continue
            except ValueError:
                pass

            # Try to match by name
            matched = False
            for option in options:
                if option.lower() == part.lower():
                    selected.append(option)
                    matched = True
                    break

            if not matched:
                console.print(f"[red]✗[/red] Invalid choice: {part}")
                valid = False
                break

        if valid and selected:
            return list(set(selected))  # Remove duplicates

        console.print("[red]✗[/red] Please try again.")


def prompt_int(
    message: str,
    default: int | None = None,
    min_value: int | None = None,
    max_value: int | None = None,
    console: Console | None = None,
) -> int:
    """Get integer input from the user.

    Args:
        message: Prompt message
        default: Default value if user just presses Enter
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        console: Optional console instance (uses default if None)

    Returns:
        int: User's integer input
    """
    console = console or _prompt_console

    while True:
        value_str = prompt(
            message,
            default=str(default) if default is not None else "",
            console=console,
        )

        try:
            value = int(value_str)

            # Validate range
            if min_value is not None and value < min_value:
                console.print(f"[red]✗[/red] Value must be at least {min_value}")
                continue

            if max_value is not None and value > max_value:
                console.print(f"[red]✗[/red] Value must be at most {max_value}")
                continue

            return value

        except ValueError:
            console.print("[red]✗[/red] Please enter a valid number")


def prompt_float(
    message: str,
    default: float | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    console: Console | None = None,
) -> float:
    """Get float input from the user.

    Args:
        message: Prompt message
        default: Default value if user just presses Enter
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        console: Optional console instance (uses default if None)

    Returns:
        float: User's float input
    """
    console = console or _prompt_console

    while True:
        value_str = prompt(
            message,
            default=str(default) if default is not None else "",
            console=console,
        )

        try:
            value = float(value_str)

            # Validate range
            if min_value is not None and value < min_value:
                console.print(f"[red]✗[/red] Value must be at least {min_value}")
                continue

            if max_value is not None and value > max_value:
                console.print(f"[red]✗[/red] Value must be at most {max_value}")
                continue

            return value

        except ValueError:
            console.print("[red]✗[/red] Please enter a valid number")
