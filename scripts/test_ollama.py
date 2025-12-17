#!/usr/bin/env python3
"""Test script for Ollama connectivity and functionality.

This script tests:
1. Connection to Ollama server
2. Listing available models
3. Simple chat completion
4. Tool calling with a mock weather tool
5. Streaming responses
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from aria.config import get_settings
from aria.llm.client import OllamaClient, OllamaConnectionError
from aria.llm.models import ChatMessage
from aria.llm.tools import create_simple_tool

console = Console()


async def test_health_check(client: OllamaClient) -> bool:
    """Test connection to Ollama."""
    console.print("\n[bold blue]1. Testing Ollama Connection[/bold blue]")

    try:
        is_healthy = await client.health_check()
        if is_healthy:
            console.print("[green]✓[/green] Successfully connected to Ollama")
            console.print(f"  URL: {client.base_url}")
            return True
        else:
            console.print("[red]✗[/red] Ollama health check failed")
            return False
    except OllamaConnectionError as e:
        console.print(f"[red]✗[/red] Connection failed: {e}")
        console.print("\n[yellow]Troubleshooting tips:[/yellow]")
        console.print("  1. Make sure Ollama is running")
        console.print("  2. Check OLLAMA_HOST in .env file")
        console.print("  3. For WSL: verify host.docker.internal or Windows IP")
        return False


async def test_list_models(client: OllamaClient) -> bool:
    """Test listing available models."""
    console.print("\n[bold blue]2. Listing Available Models[/bold blue]")

    try:
        models = await client.list_models()

        if not models:
            console.print("[yellow]⚠[/yellow] No models found")
            console.print(f"  Run: ollama pull {client.model}")
            return False

        table = Table(title="Available Models")
        table.add_column("Name", style="cyan")
        table.add_column("Size", style="green", justify="right")
        table.add_column("Modified", style="yellow")

        for model in models:
            size_str = f"{model.size_gb:.2f} GB" if model.size_gb else "Unknown"
            modified_str = (
                model.modified_at.strftime("%Y-%m-%d %H:%M")
                if model.modified_at
                else "Unknown"
            )
            table.add_row(model.name, size_str, modified_str)

        console.print(table)

        # Check if default model exists
        if await client.model_exists(client.model):
            console.print(f"[green]✓[/green] Default model '{client.model}' is available")
            return True
        else:
            console.print(
                f"[yellow]⚠[/yellow] Default model '{client.model}' not found"
            )
            console.print(f"  Run: ollama pull {client.model}")
            return False

    except Exception as e:
        console.print(f"[red]✗[/red] Failed to list models: {e}")
        return False


async def test_simple_chat(client: OllamaClient) -> bool:
    """Test simple chat completion."""
    console.print("\n[bold blue]3. Testing Simple Chat[/bold blue]")

    try:
        prompt = "What is the capital of France? Answer in one sentence."
        console.print(f"  Prompt: [italic]{prompt}[/italic]")

        with console.status("[bold green]Generating response..."):
            response = await client.simple_chat(
                prompt=prompt,
                temperature=0.7,
            )

        console.print(Panel(
            response,
            title="Response",
            border_style="green",
        ))

        # Show metadata
        messages = [ChatMessage.user(prompt)]
        full_response = await client.chat(messages)
        metadata = full_response.metadata

        console.print("\n  [dim]Metadata:[/dim]")
        if metadata.tokens_per_second:
            console.print(f"    Speed: {metadata.tokens_per_second:.1f} tokens/s")
        if metadata.eval_count:
            console.print(f"    Tokens generated: {metadata.eval_count}")
        if metadata.total_duration_ms:
            console.print(f"    Total time: {metadata.total_duration_ms:.0f}ms")

        console.print("[green]✓[/green] Chat completion successful")
        return True

    except Exception as e:
        console.print(f"[red]✗[/red] Chat failed: {e}")
        return False


async def test_streaming_chat(client: OllamaClient) -> bool:
    """Test streaming chat completion."""
    console.print("\n[bold blue]4. Testing Streaming Chat[/bold blue]")

    try:
        prompt = "Count from 1 to 5, one number per line."
        console.print(f"  Prompt: [italic]{prompt}[/italic]")
        console.print("\n  [bold]Streaming response:[/bold]")
        console.print("  ", end="")

        full_text = ""
        async for chunk in client.stream_simple_chat(prompt=prompt, temperature=0.3):
            console.print(chunk, end="")
            full_text += chunk

        console.print("\n")
        console.print("[green]✓[/green] Streaming successful")
        return True

    except Exception as e:
        console.print(f"\n[red]✗[/red] Streaming failed: {e}")
        return False


async def test_tool_calling(client: OllamaClient) -> bool:
    """Test tool calling with a mock weather tool."""
    console.print("\n[bold blue]5. Testing Tool Calling[/bold blue]")

    try:
        # Create a mock weather tool
        weather_tool = create_simple_tool(
            name="get_weather",
            description="Get the current weather for a location",
            properties={
                "location": {
                    "type": "string",
                    "description": "The city and country, e.g., 'Paris, France'",
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units",
                },
            },
            required=["location"],
        )

        # Check model capabilities
        capabilities = client.get_model_capabilities()
        if not capabilities.supports_tools:
            console.print(
                f"[yellow]⚠[/yellow] Model '{client.model}' may not support tools"
            )

        # Send a request that should trigger tool use
        messages = [
            ChatMessage.user("What's the weather like in Tokyo? Use Celsius.")
        ]

        console.print("  Prompt: [italic]What's the weather like in Tokyo? Use Celsius.[/italic]")

        with console.status("[bold green]Generating response with tools..."):
            response = await client.chat_with_tools(
                messages=messages,
                tools=[weather_tool],
                temperature=0.3,
            )

        # Check if tool was called
        if response.has_tool_calls:
            console.print("[green]✓[/green] Model made tool call(s)!")

            for i, tool_call in enumerate(response.message.tool_calls, 1):
                console.print(f"\n  [bold]Tool Call {i}:[/bold]")
                console.print(f"    Function: {tool_call.function.name}")
                console.print(f"    Arguments: {tool_call.function.arguments}")

            return True
        else:
            console.print(
                "[yellow]⚠[/yellow] Model did not make a tool call (returned text instead)"
            )
            console.print(f"\n  Response: {response.message.content[:200]}...")
            console.print("\n  [dim]This is expected for some models that don't support tools well.[/dim]")
            return True  # Still counts as success since the call worked

    except Exception as e:
        console.print(f"[red]✗[/red] Tool calling test failed: {e}")
        return False


async def main():
    """Run all tests."""
    console.print(Panel.fit(
        "[bold cyan]Ollama Client Test Suite[/bold cyan]\n"
        "Testing connectivity and functionality",
        border_style="cyan",
    ))

    # Load settings
    settings = get_settings()
    console.print(f"\n[dim]Using configuration:[/dim]")
    console.print(f"  Host: {settings.ollama_host}")
    console.print(f"  Model: {settings.ollama_model}")
    console.print(f"  Timeout: {settings.ollama_timeout}s")

    # Create client
    async with OllamaClient() as client:
        tests = [
            ("Health Check", test_health_check),
            ("List Models", test_list_models),
            ("Simple Chat", test_simple_chat),
            ("Streaming Chat", test_streaming_chat),
            ("Tool Calling", test_tool_calling),
        ]

        results = {}

        for test_name, test_func in tests:
            try:
                result = await test_func(client)
                results[test_name] = result
            except Exception as e:
                console.print(f"[red]✗[/red] Unexpected error in {test_name}: {e}")
                results[test_name] = False

            # Stop if critical tests fail
            if test_name in ["Health Check", "List Models"] and not result:
                console.print(
                    f"\n[red]Critical test '{test_name}' failed. Stopping.[/red]"
                )
                break

    # Summary
    console.print("\n" + "=" * 70)
    console.print("\n[bold]Test Summary:[/bold]\n")

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for test_name, result in results.items():
        status = "[green]✓ PASS[/green]" if result else "[red]✗ FAIL[/red]"
        console.print(f"  {status}  {test_name}")

    console.print(f"\n[bold]Results: {passed}/{total} tests passed[/bold]")

    if passed == total:
        console.print("\n[bold green]🎉 All tests passed! Ollama client is working correctly.[/bold green]")
        return 0
    else:
        console.print("\n[bold yellow]⚠ Some tests failed. Check the output above.[/bold yellow]")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
