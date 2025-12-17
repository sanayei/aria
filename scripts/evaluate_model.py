#!/usr/bin/env python3
"""Evaluate qwen3:30b-a3b on curated hard tasks and report latency metrics."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Make project src importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aria.config import get_settings
from aria.llm import OllamaClient, ChatMessage

console = Console()

BASE_SYSTEM_PROMPT = (
    "You are a meticulous reasoning assistant. Think through each problem step by "
    "step before answering. Always finish with a standalone line that begins with "
    "'Final answer:' followed by the concise result requested."
)

ValidationFn = Callable[[str], tuple[bool, str]]


def normalize_text(text: str) -> str:
    """Create a comparable version of a string for fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def simple_validator(expected_answer: str) -> ValidationFn:
    """Return a validator that checks for an expected answer substring."""
    expected_norm = normalize_text(expected_answer)

    def _validator(response_text: str) -> tuple[bool, str]:
        response_norm = normalize_text(response_text)
        success = expected_norm in response_norm
        reason = "Matched expected answer" if success else "Expected answer not found"
        return success, reason

    return _validator


@dataclass(slots=True)
class HardTask:
    """Definition of a hard evaluation task."""

    name: str
    prompt: str
    expected_answer: str
    tags: tuple[str, ...] = ()
    validator: ValidationFn | None = None

    def evaluate(self, response_text: str) -> tuple[bool, str]:
        """Evaluate model output for this task."""
        validator = self.validator or simple_validator(self.expected_answer)
        return validator(response_text)


@dataclass(slots=True)
class HardTaskResult:
    """Stores metrics for a completed task."""

    task: HardTask
    success: bool
    reason: str
    response_text: str
    latency_s: float
    total_duration_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    tokens_per_second: float | None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert to a JSON serializable dict."""
        return {
            "task": {
                "name": self.task.name,
                "tags": list(self.task.tags),
                "expected_answer": self.task.expected_answer,
            },
            "success": self.success,
            "reason": self.reason,
            "error": self.error,
            "latency_s": self.latency_s,
            "total_duration_ms": self.total_duration_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tokens_per_second": self.tokens_per_second,
            "response_text": self.response_text,
        }


def build_hard_tasks() -> list[HardTask]:
    """Return the curated list of hard evaluation tasks."""
    return [
        HardTask(
            name="Weighted consecutive integers",
            tags=("math", "algebra", "multi-step"),
            prompt=(
                "Three consecutive even integers are arranged from smallest to largest as "
                "x, x+2, and x+4. If the smallest integer plus twice the middle integer "
                "plus three times the largest equals 400, what are the integers? "
                "Explain the algebra you use and end with 'Final answer: a, b, c' using "
                "ascending order."
            ),
            expected_answer="Final answer: 64, 66, 68",
        ),
        HardTask(
            name="Chinese remainder puzzle",
            tags=("math", "number theory"),
            prompt=(
                "Find the smallest positive integer n such that n leaves a remainder of 1 "
                "when divided by 5, a remainder of 3 when divided by 6, and a remainder of "
                "2 when divided by 7. Show the modular reasoning and conclude with "
                "'Final answer: <number>'."
            ),
            expected_answer="Final answer: 51",
        ),
        HardTask(
            name="Three-year investment swing",
            tags=("finance", "reasoning"),
            prompt=(
                "An analyst invests $12,000 in a fund that gains 4% the first year, loses "
                "3% the second year, and gains 5% the third year. Compute the value at the "
                "end of the third year, rounding to two decimals. Provide the percentage "
                "math you use and finish with \"Final answer: <amount>\" without the "
                "dollar sign."
            ),
            expected_answer="Final answer: 12710.88",
        ),
        HardTask(
            name="Constrained grid walk",
            tags=("combinatorics", "reasoning"),
            prompt=(
                "You move on a 5x5 grid from the bottom-left corner (0,0) to the top-right "
                "corner (4,4), only moving one step up or one step right at a time. The cell "
                "at coordinates (2,2) is blocked and cannot be used. How many distinct paths "
                "reach the destination without touching the blocked cell? Derive the count "
                "and end with 'Final answer: <number>'."
            ),
            expected_answer="Final answer: 34",
        ),
        HardTask(
            name="Ciphered phrase shuffle",
            tags=("language", "manipulation"),
            prompt=(
                "Take the phrase 'hard problems sharpen minds' and perform the following "
                "operations: (1) shift every alphabetical letter forward by exactly one "
                "place in the alphabet (z wraps to a), keeping spaces in place; "
                "(2) reverse the order of the words; (3) convert the result to uppercase. "
                "Explain each transformation and conclude with "
                "\"Final answer: <transformed phrase>\"."
            ),
            expected_answer="Final answer: NJOET TIBSQFO QSPCMFNT IBSE",
        ),
    ]


async def run_task(
    client: OllamaClient,
    task: HardTask,
    temperature: float,
) -> HardTaskResult:
    """Execute a single task and capture timing/token metrics."""
    messages = [
        ChatMessage.system(BASE_SYSTEM_PROMPT),
        ChatMessage.user(task.prompt),
    ]

    start = time.perf_counter()
    try:
        response = await client.chat(messages=messages, temperature=temperature)
        latency = time.perf_counter() - start
        text = response.message.content.strip()
        success, reason = task.evaluate(text)
        metadata = response.metadata
        return HardTaskResult(
            task=task,
            success=success,
            reason=reason,
            response_text=text,
            latency_s=latency,
            total_duration_ms=metadata.total_duration_ms,
            prompt_tokens=metadata.prompt_eval_count,
            completion_tokens=metadata.eval_count,
            tokens_per_second=metadata.tokens_per_second,
        )
    except Exception as exc:  # noqa: BLE001
        latency = time.perf_counter() - start
        return HardTaskResult(
            task=task,
            success=False,
            reason="Exception during chat call",
            response_text="",
            latency_s=latency,
            total_duration_ms=None,
            prompt_tokens=None,
            completion_tokens=None,
            tokens_per_second=None,
            error=str(exc),
        )


def render_results(results: Sequence[HardTaskResult]) -> None:
    """Pretty-print per-task and aggregate evaluation metrics."""
    table = Table(title="Hard Task Evaluation", show_lines=False)
    table.add_column("Task", style="cyan", no_wrap=True)
    table.add_column("Tags", style="magenta")
    table.add_column("Result", justify="center")
    table.add_column("Latency (s)", justify="right")
    table.add_column("Tokens→", justify="right")
    table.add_column("Tok/s", justify="right")
    table.add_column("Notes", style="green")

    success_latencies = []
    total_latencies = []
    generation_speeds = []

    for result in results:
        icon = "✅" if result.success else "❌"
        latency = f"{result.latency_s:.2f}" if result.latency_s else "?"
        tokens_out = (
            str(result.completion_tokens) if result.completion_tokens is not None else "?"
        )
        tok_speed = (
            f"{result.tokens_per_second:.1f}"
            if result.tokens_per_second is not None
            else "?"
        )
        note = result.reason if result.error is None else result.error

        table.add_row(
            result.task.name,
            ", ".join(result.task.tags),
            icon,
            latency,
            tokens_out,
            tok_speed,
            note,
        )

        total_latencies.append(result.latency_s)
        if result.success:
            success_latencies.append(result.latency_s)
        if result.tokens_per_second:
            generation_speeds.append(result.tokens_per_second)

    console.print(table)

    summary_lines = []
    total_tasks = len(results)
    successes = sum(1 for r in results if r.success)
    success_rate = (successes / total_tasks * 100) if total_tasks else 0
    summary_lines.append(f"Tasks passed: {successes}/{total_tasks} ({success_rate:.1f}%)")

    if total_latencies:
        avg_latency = statistics.mean(total_latencies)
        summary_lines.append(f"Avg wall-clock latency: {avg_latency:.2f}s")

    if success_latencies:
        avg_success_latency = statistics.mean(success_latencies)
        summary_lines.append(f"Avg latency on successes: {avg_success_latency:.2f}s")

    if generation_speeds:
        avg_speed = statistics.mean(generation_speeds)
        summary_lines.append(f"Avg generation speed: {avg_speed:.1f} tokens/s")

    console.print(Panel("\n".join(summary_lines), title="Summary", border_style="blue"))


def save_results(path: Path, results: Sequence[HardTaskResult]) -> None:
    """Persist raw evaluation data to disk."""
    payload = {
        "summary": {
            "total_tasks": len(results),
            "passed": sum(1 for r in results if r.success),
        },
        "results": [r.to_dict() for r in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    console.print(f"[dim]Saved detailed results to {path}[/dim]")


async def main_async(args: argparse.Namespace) -> None:
    """Run the evaluation end to end."""
    settings = get_settings()
    client = OllamaClient(
        base_url=args.host or settings.ollama_host,
        model=args.model or settings.ollama_model,
        timeout=args.timeout or settings.ollama_timeout,
        settings=settings,
    )

    tasks = build_hard_tasks()
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]

    console.print(
        f"[bold]Evaluating {len(tasks)} hard tasks on model '{client.model}' via {client.base_url}[/bold]"
    )

    results: list[HardTaskResult] = []
    try:
        for task in tasks:
            console.print(f"\n[cyan]→ Running:[/cyan] {task.name}")
            result = await run_task(client, task, temperature=args.temperature)
            results.append(result)
            if args.show_responses:
                console.print(
                    Panel(
                        result.response_text or "(no response)",
                        title=f"{task.name} response",
                        border_style="green" if result.success else "red",
                    )
                )
    finally:
        await client.close()

    render_results(results)

    if args.output:
        save_results(Path(args.output).expanduser(), results)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Evaluate qwen3:30b-a3b on curated hard reasoning tasks.",
    )
    parser.add_argument("--model", help="Override the Ollama model name")
    parser.add_argument("--host", help="Override the Ollama host URL")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature to use (default: 0.2)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Request timeout in seconds (defaults to config value)",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write raw JSON results",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        help="Only run the first N tasks",
    )
    parser.add_argument(
        "--show-responses",
        action="store_true",
        help="Print full model responses for each task",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Entry point."""
    args = parse_args(sys.argv[1:])
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
