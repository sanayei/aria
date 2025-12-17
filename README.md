# ARIA - AI Research & Intelligence Assistant

A local-first agentic AI personal assistant designed to run entirely on your PC, emphasizing privacy and local control over cloud dependencies.

## Features

- **File Management**: Automated organization, intelligent categorization, duplicate detection
- **Research & Information Retrieval**: Web search, document analysis, knowledge synthesis
- **Document Analysis**: PDF extraction, summarization, Q&A over documents
- **Email Management**: Gmail integration with intelligent categorization, labeling, and drafting
- **System Administration**: Routine maintenance tasks, monitoring, automation

## Design Principles

- **Local-First**: All core functionality runs locally with no cloud dependencies
- **Privacy-Preserving**: Your data never leaves your machine
- **Human-in-the-Loop**: Critical actions require explicit user approval
- **Modular Architecture**: Extensible tool system with clear interfaces

## Requirements

### Hardware

- **RAM**: 16 GB minimum, 32+ GB recommended
- **GPU**: NVIDIA GPU with 16+ GB VRAM for optimal performance
- **Storage**: 50+ GB free space for models and data

### Software

- **Python**: 3.11 or higher
- **Ollama**: For local LLM inference
- **uv**: Modern Python package manager
- **poppler-utils**: Required for PDF processing (install with `sudo apt-get install poppler-utils` on Ubuntu/Debian)

## Quick Start

### 1. Install Dependencies

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd aria

# Install Python dependencies
uv sync
```

### 2. Set Up Ollama

**On Windows (for optimal GPU performance):**

1. Download and install [Ollama](https://ollama.ai/)
2. Pull the primary model:
   ```cmd
   ollama pull qwen3:30b-a3b
   ```
3. Ensure Ollama is running (starts automatically)

**On Linux/Mac:**

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the primary model
ollama pull qwen3:30b-a3b
```

### 3. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your settings
```

**For WSL users**: You need to find your Windows host IP to connect to Ollama:

```bash
# Find your Windows host IP from WSL
ip route show | grep -i default | awk '{ print $3}'

# Example output: 127.0.0.1
# Use this in your .env file as: OLLAMA_HOST=http://127.0.0.1:11434
```

### 4. Test Ollama Connection

```bash
# From WSL (replace with your Windows host IP)
curl http://127.0.0.1:11434/api/tags

# From Linux/Mac (local Ollama)
curl http://localhost:11434/api/tags
```

### 5. Run ARIA

```bash
uv run aria
```

## Configuration

ARIA is configured via environment variables in the `.env` file. Key settings:

- `OLLAMA_HOST`: Ollama server URL
- `OLLAMA_MODEL`: Primary LLM model (default: qwen3:30b-a3b)
- `ARIA_LOG_LEVEL`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `ARIA_DATA_DIR`: Local data storage directory
- `TOOL_AUTO_APPROVE_LOW_RISK`: Auto-approve read-only operations

See [.env.example](.env.example) for all available options.

## Project Structure

```
aria/
├── src/aria/           # Main application code
│   ├── agent/          # ReAct agent implementation
│   ├── llm/            # Ollama client and model management
│   ├── tools/          # Modular tool system
│   ├── memory/         # Conversation history and knowledge base
│   ├── approval/       # Risk-based approval system
│   └── ui/             # CLI interface
├── tests/              # Test suite
├── scripts/            # Utility scripts
└── data/               # Local data (gitignored)
```

## Development

### Running Tests

```bash
uv run pytest
```

### Adding Dependencies

```bash
# Runtime dependency
uv add <package>

# Development dependency
uv add --dev <package>
```

### Code Style

This project follows:
- Type hints everywhere (Python 3.11+ syntax)
- Pydantic for data validation
- Async/await for I/O operations
- Rich for beautiful CLI output

## Gmail Integration (Optional)

To enable email management features:

1. Create a Google Cloud project
2. Enable the Gmail API
3. Download OAuth credentials as `credentials.json`
4. Run the setup script: `uv run python scripts/setup_gmail.py`
5. Follow the OAuth flow in your browser

## Architecture

ARIA uses the **ReAct (Reason + Act)** pattern:

1. **THINK**: Analyze user request
2. **PLAN**: Break down into steps
3. **EXECUTE**: Call tools with appropriate risk management
4. **REFLECT**: Evaluate results
5. **RESPOND**: Provide answer or ask for clarification

### Risk-Based Approval

| Risk Level   | Examples                        | Approval Required   |
| ------------ | ------------------------------- | ------------------- |
| **LOW**      | Read files, search, analyze     | None (auto-execute) |
| **MEDIUM**   | Send emails, modify files       | User confirmation   |
| **HIGH**     | Delete files, financial actions | Explicit approval   |
| **CRITICAL** | System changes, bulk operations | Double confirmation |

## Roadmap

- [x] Project initialization and configuration
- [ ] Ollama client with tool support
- [ ] Basic CLI interface
- [ ] ReAct agent loop
- [ ] Tool registry and base classes
- [ ] Gmail integration
- [ ] File system tools
- [ ] Vector store and semantic search
- [ ] PDF extraction
- [ ] Voice output (TTS)

## Privacy & Security

- **No telemetry**: ARIA doesn't send any usage data
- **Local storage**: All data stays on your machine
- **Credential safety**: API keys and tokens stored locally only
- **Human oversight**: Critical actions require explicit approval

## License

[Add your license here]

## Contributing

This is currently a personal hobby project. Feel free to fork and adapt for your own use!

## Support

For issues and questions, please see [CLAUDE.md](CLAUDE.md) for development context.
