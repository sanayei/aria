# ARIA - AI Research & Intelligence Assistant

## Project Overview

ARIA is a local-first agentic AI personal assistant designed to run entirely on a high-end personal PC. This is a hobby project with a single user (the developer), emphasizing privacy and local control over cloud dependencies.

### Core Objectives

- **File Management**: Automated organization, intelligent categorization, duplicate detection
- **Research & Information Retrieval**: Web search, document analysis, knowledge synthesis
- **Document Analysis**: PDF extraction, summarization, Q&A over documents
- **Email Management**: Gmail integration with intelligent categorization, labeling, and drafting
- **System Administration**: Routine maintenance tasks, monitoring, automation

### Success Criteria

- Modular, extensible architecture where components can be developed/tested independently
- Human-in-the-loop confirmation for critical or irreversible actions
- Intelligent automation for routine, low-risk tasks
- Privacy-preserving: all core functionality runs locally

---

## Hardware & Environment

### Hardware Specs

- **RAM**: 64 GB
- **CPU**: AMD Ryzen 9 9900X (12-Core, 24-Thread)
- **GPU**: NVIDIA RTX 3090 (24 GB VRAM)

### Development Environment

**Hybrid Windows/WSL Setup:**

- **Windows (native)**: Ollama inference server for optimal GPU performance
- **WSL (Ubuntu)**: Python development environment, application code

**Communication**: ARIA in WSL connects to Ollama on Windows via `http://host.docker.internal:11434` or the Windows host IP.

### Primary LLM

- **Model**: Qwen3-32B via Ollama
- **Reasoning**: Strong tool-calling capabilities, good balance of intelligence and speed for local inference
- **VRAM Usage**: ~20GB at Q4 quantization, leaving headroom for other models

---

## Architecture

### Agent Pattern: ReAct (Reason + Act)

The agent follows a Think-Plan-Execute-Reflect loop:

```
1. THINK: Analyze the user's request, understand intent
2. PLAN: Break down into steps, identify required tools
3. EXECUTE: Call tools, gather information
4. REFLECT: Evaluate results, determine if goal is met
5. RESPOND: Provide final answer or ask for clarification
```

### Risk-Based Approval System

Actions are classified by risk level:

| Risk Level | Examples | Approval Required |
|------------|----------|-------------------|
| **LOW** | Read files, search, analyze | None (auto-execute) |
| **MEDIUM** | Send emails, modify files | User confirmation |
| **HIGH** | Delete files, financial actions | Explicit approval + confirmation |
| **CRITICAL** | System changes, bulk operations | Double confirmation + summary |

### Modular Tool Architecture

Each tool is a self-contained module with:
- Clear input/output schemas (Pydantic models)
- Risk level classification
- Rollback capability where applicable
- Comprehensive logging

---

## Tech Stack

### Core Dependencies

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Runtime** | Python 3.11+ | Main application language |
| **Package Manager** | uv | Fast, modern dependency management |
| **LLM Inference** | Ollama | Local model serving |
| **Primary Model** | Qwen3-32B | Tool-calling, reasoning |
| **Vector Store** | ChromaDB | Embeddings, semantic search |
| **Database** | SQLite | Conversation history, caching |
| **CLI Interface** | Rich | Beautiful terminal output |
| **Async HTTP** | httpx | API calls to Ollama, external services |
| **Data Validation** | Pydantic | Schemas, settings, tool I/O |

### Email Integration

| Component | Technology |
|-----------|------------|
| **API** | Gmail API (not IMAP) |
| **Auth** | google-api-python-client, google-auth-oauthlib |
| **Cache** | SQLite (local email metadata cache) |

### Future Additions (Not Yet Implemented)

| Component | Technology | Notes |
|-----------|------------|-------|
| **TTS (Interactive)** | Kokoro | Fast, lightweight |
| **TTS (Batch)** | Fish Speech | Higher quality for summaries |
| **PDF Extraction** | MinerU or Docling | Evaluate based on document types |
| **Orchestration** | LangGraph | If workflow complexity requires it |

---

## Project Structure

```
aria/
├── CLAUDE.md                 # This file - project context
├── pyproject.toml            # uv/Python project configuration
├── uv.lock                   # Locked dependencies
├── README.md                 # User-facing documentation
├── .env.example              # Environment variables template
├── .gitignore
│
├── src/
│   └── aria/
│       ├── __init__.py
│       ├── main.py           # Entry point, CLI setup
│       ├── config.py         # Settings, configuration management
│       │
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── core.py       # Main agent loop (ReAct)
│       │   ├── planner.py    # Task decomposition
│       │   ├── executor.py   # Tool execution with approval
│       │   └── prompts.py    # System prompts, templates
│       │
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── client.py     # Ollama client wrapper
│       │   ├── models.py     # Model configurations
│       │   └── tools.py      # Tool schema generation
│       │
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── base.py       # Base tool class, registry
│       │   ├── filesystem/   # File operations
│       │   ├── email/        # Gmail integration
│       │   ├── search/       # Web search, local search
│       │   └── documents/    # PDF processing
│       │
│       ├── memory/
│       │   ├── __init__.py
│       │   ├── conversation.py   # Chat history (SQLite)
│       │   ├── vectors.py        # ChromaDB integration
│       │   └── knowledge.py      # Long-term knowledge store
│       │
│       ├── approval/
│       │   ├── __init__.py
│       │   ├── classifier.py     # Risk classification
│       │   └── handler.py        # User approval flow
│       │
│       └── ui/
│           ├── __init__.py
│           ├── console.py        # Rich console interface
│           └── formatters.py     # Output formatting
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Pytest fixtures
│   ├── test_agent/
│   ├── test_tools/
│   └── test_llm/
│
├── scripts/
│   ├── setup_gmail.py        # Gmail OAuth setup helper
│   └── test_ollama.py        # Verify Ollama connection
│
└── data/                     # Local data (gitignored)
    ├── chroma/               # Vector store
    ├── cache/                # SQLite databases
    └── logs/                 # Application logs
```

---

## Development Phases

### Phase 1: Foundation (Current)
- [x] Architecture design
- [x] Tech stack selection
- [ ] Project initialization (uv, structure)
- [ ] Configuration management (Pydantic settings)
- [ ] Ollama client wrapper with tool support
- [ ] Basic CLI interface

### Phase 2: Agent Core
- [ ] ReAct agent loop implementation
- [ ] Tool registry and base class
- [ ] Risk classification system
- [ ] User approval workflows
- [ ] Conversation history (SQLite)

### Phase 3: First Tools
- [ ] Gmail API integration (read, search, label)
- [ ] Gmail send with approval flow
- [ ] Local file system tools (read, list, search)
- [ ] Basic web search

### Phase 4: Memory & Context
- [ ] ChromaDB integration
- [ ] Document ingestion pipeline
- [ ] Semantic search over documents
- [ ] Long-term memory/knowledge base

### Phase 5: Advanced Features
- [ ] PDF extraction (MinerU/Docling)
- [ ] Voice output (Kokoro TTS)
- [ ] Daily summary generation
- [ ] Calendar integration

### Phase 6: Polish
- [ ] Comprehensive error handling
- [ ] Performance optimization
- [ ] Documentation
- [ ] Backup/restore functionality

---

## Coding Guidelines

### General Principles

1. **Type hints everywhere** - Use modern Python typing (3.11+ syntax)
2. **Pydantic for data** - All configs, tool inputs/outputs, API responses
3. **Async by default** - Use `async/await` for I/O operations
4. **Explicit over implicit** - Clear naming, no magic
5. **Fail gracefully** - Comprehensive error handling, helpful messages

### Code Style

```python
# Use modern type hints
def process_email(email_id: str, labels: list[str] | None = None) -> EmailResult:
    ...

# Pydantic models for structured data
class ToolInput(BaseModel):
    """Base class for tool inputs."""
    model_config = ConfigDict(strict=True)

class EmailSearchInput(ToolInput):
    """Input schema for email search tool."""
    query: str
    max_results: int = Field(default=10, ge=1, le=100)
    include_spam: bool = False

# Async functions for I/O
async def fetch_emails(query: str) -> list[Email]:
    async with httpx.AsyncClient() as client:
        ...

# Rich for CLI output
from rich.console import Console
console = Console()
console.print("[green]✓[/green] Email sent successfully")
```

### Logging

- Use `structlog` or standard `logging` with structured output
- Log levels: DEBUG for development, INFO for operations, WARNING/ERROR for issues
- Include context (tool name, action, relevant IDs)

### Testing

- Use `pytest` with `pytest-asyncio` for async tests
- Fixtures in `conftest.py` for common setup
- Mock external services (Ollama, Gmail API) in tests
- Aim for high coverage on core agent logic

### Git Practices

- Meaningful commit messages
- Feature branches for new tools/capabilities
- Keep `main` stable and working

---

## Key Constraints

### Privacy & Local-First

- All core functionality must work offline (except email sync, web search)
- No telemetry or data collection
- Credentials stored locally, never logged
- User data never leaves the local machine

### Human-in-the-Loop

- NEVER auto-execute destructive or irreversible actions
- Always show what will be done before doing it for medium+ risk
- Provide clear rollback options where possible
- When in doubt, ask for confirmation

### Resource Management

- Monitor VRAM usage (24GB total, ~20GB for primary LLM)
- Implement model unloading for TTS/other models when not in use
- Use quantized models appropriately
- SQLite for caching to reduce repeated API calls

### Error Handling

- Never crash on recoverable errors
- Provide actionable error messages
- Log errors with full context for debugging
- Graceful degradation (e.g., if Ollama is down, inform user clearly)

---

## Current Status

**Starting Phase 1: Foundation**

The next immediate tasks are:
1. Initialize the project with `uv`
2. Create the directory structure
3. Set up configuration management
4. Implement the Ollama client wrapper
5. Create basic CLI interface

---

## Quick Reference

### Commands

```bash
# Project location
cd ~/projects/aria

# Run ARIA (once implemented)
uv run aria

# Run tests
uv run pytest

# Add a dependency
uv add <package>

# Add a dev dependency
uv add --dev <package>
```

### Environment Variables

```bash
# .env file
OLLAMA_HOST=http://host.docker.internal:11434  # or Windows IP
ARIA_LOG_LEVEL=INFO
ARIA_DATA_DIR=./data
```

### Ollama Connection Test

```bash
# From WSL, test Ollama connectivity
curl http://host.docker.internal:11434/api/tags
```

---

## Notes for Claude Code

When working on this project:

1. **Always use `uv`** for dependency management, not pip directly
2. **Follow the project structure** - place new code in appropriate modules
3. **Use async/await** for any I/O operations
4. **Add type hints** to all functions
5. **Create Pydantic models** for any structured data
6. **Ask before** implementing major architectural changes
7. **Test Ollama connectivity** before implementing LLM features
8. **Keep the human-in-the-loop** - never auto-implement destructive actions

When adding new tools:
1. Create a new directory under `src/aria/tools/`
2. Inherit from the base `Tool` class
3. Define clear Pydantic input/output schemas
4. Assign appropriate risk level
5. Add tests in `tests/test_tools/`
