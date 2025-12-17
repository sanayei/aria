"""Demo script showing the analyze_file tool in action."""

import asyncio
from pathlib import Path

from aria.tools.filesystem import AnalyzeFileTool, AnalyzeFileParams


async def demo():
    """Demonstrate file analysis capabilities."""
    tool = AnalyzeFileTool()

    # Analyze this demo script itself
    script_path = Path(__file__)
    params = AnalyzeFileParams(path=str(script_path))

    result = await tool.execute(params)

    if result.success:
        print("File Analysis Results:")
        print("=" * 50)
        for key, value in result.data.items():
            print(f"{key:15s}: {value}")
    else:
        print(f"Error: {result.error}")


if __name__ == "__main__":
    asyncio.run(demo())
