# Quickstart Guide

Get started with ARIA in 5 minutes!

## 1. First Run

After [installation](installation.md), start ARIA:

```bash
uv run aria chat
```

You'll see an interactive chat interface:

```
🤖 ARIA - AI Research & Intelligence Assistant
Type 'help' for available commands, 'exit' to quit

You:
```

## 2. Basic Commands

### Ask a Question

```
You: What files are in my current directory?
```

ARIA will use tools to explore your filesystem and respond.

### Web Search

```
You: Search the web for latest Python 3.12 features
```

### Email Management (requires Gmail setup)

```
You: Show me unread emails from today
```

## 3. Document Processing

### Scan and Index Documents

```bash
uv run aria scan process --source /path/to/scanned/pdfs
```

This will:
1. Extract text using OCR
2. Classify documents by person and category
3. Archive with intelligent naming
4. Index in vector database

### Search Archived Documents

```
You: Find all medical documents for John from 2024
```

## 4. Web Interface

### Start the Web Server

```bash
# Option 1: Using the script
bash scripts/start_web.sh

# Option 2: Directly with uvicorn
uv run uvicorn aria.web.app:app --host 0.0.0.0 --port 8000 --reload
```

### Access the Interface

Open http://localhost:8000

Features:
- 📄 **Document Browser**: View, search, filter archived documents
- 👥 **User Management**: Admin panel for managing users and permissions
- 🏷️ **Tag Management**: Organize documents with custom tags
- 📊 **Statistics**: View document statistics by person, category, year

## 5. Common Workflows

### Daily Email Summary

```
You: Summarize my emails from today and draft responses for important ones
```

### Document Search

```
You: Find all tax documents for 2024
```

### File Organization

```
You: Organize my Downloads folder by file type
```

## Command Reference

### CLI Commands

```bash
# Interactive chat
uv run aria chat

# Scan documents
uv run aria scan process --source /path/to/docs

# Deduplicate files
uv run aria scan deduplicate --source /path/to/source --execute

# Start web server
uv run aria web
```

### Chat Commands

While in chat mode:

- `help` - Show available commands
- `history` - View conversation history
- `clear` - Clear conversation history
- `tools` - List available tools
- `exit` - Exit ARIA

## Configuration Tips

### Adjust LLM Temperature

In `.env`:
```bash
OLLAMA_TEMPERATURE=0.7  # Lower = more focused, Higher = more creative
```

### Enable Auto-Approval

For read-only operations:
```bash
TOOL_AUTO_APPROVE_LOW_RISK=true
```

### Customize Document Categories

In `src/aria/config.py`, edit:
```python
document_categories: list[str] = [
    "medical",
    "financial",
    "government",
    "personal",
    "other",
]
```

## Next Steps

- [Configuration Guide](configuration.md) - Detailed configuration options
- [User Guide](user_guide/overview.md) - Comprehensive feature documentation
- [Tutorials](tutorials/scanning_documents.md) - Step-by-step guides

## Getting Help

- Check the [User Guide](user_guide/overview.md)
- Review [API Documentation](api/agent.md)
- Report issues on GitHub
