# Configuration

ARIA uses environment variables and a Pydantic settings system for configuration.

## Environment Variables

All configuration is done through the `.env` file. Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

### Ollama Configuration

```bash
# Ollama server URL
OLLAMA_HOST=http://localhost:11434

# Primary LLM model
OLLAMA_MODEL=qwen3:30b-a3b

# API timeout (seconds)
OLLAMA_TIMEOUT=300

# Generation temperature (0.0-2.0)
OLLAMA_TEMPERATURE=0.7
```

**WSL Configuration:**
When running ARIA in WSL and Ollama on Windows:

```bash
# Try host.docker.internal first
OLLAMA_HOST=http://host.docker.internal:11434

# Or use your Windows IP
OLLAMA_HOST=http://192.168.1.50:11434
```

### Application Settings

```bash
# Logging level
ARIA_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Data directory
ARIA_DATA_DIR=./data

# Conversation history limit
ARIA_MAX_HISTORY=50
```

### Database Configuration

```bash
# SQLite database path (optional, uses default if not set)
DB_PATH=./data/cache/aria.db
```

### Vector Store Configuration

```bash
# ChromaDB storage path
CHROMA_PATH=./data/chroma

# Collection name
CHROMA_COLLECTION=aria_knowledge
```

### Gmail Configuration

```bash
# Gmail API credentials (download from Google Cloud Console)
GMAIL_CREDENTIALS_PATH=./credentials.json

# OAuth token (auto-generated)
GMAIL_TOKEN_PATH=./token.json

# Cache TTL in seconds
GMAIL_CACHE_TTL=300
```

See [Gmail Setup Tutorial](tutorials/setting_up_gmail.md) for details.

### Tool Execution Settings

```bash
# Auto-approve low-risk tools (read-only operations)
TOOL_AUTO_APPROVE_LOW_RISK=true

# Require confirmation for medium+ risk
TOOL_REQUIRE_CONFIRMATION=true

# Tool execution timeout
TOOL_TIMEOUT=60
```

### Agent Settings

```bash
# Maximum agent loop iterations
AGENT_MAX_ITERATIONS=20

# Verbose logging
AGENT_VERBOSE=false
```

### Web Application Security

```bash
# JWT Secret Key (REQUIRED for web app)
# Generate with: python -c 'import secrets; print(secrets.token_urlsafe(32))'
JWT_SECRET_KEY=your-secret-key-here
```

**Security Note:** Never commit your `.env` file to version control!

## Programmatic Configuration

You can also access settings in code:

```python
from aria.config import get_settings

settings = get_settings()

print(f"Ollama host: {settings.ollama_host}")
print(f"Model: {settings.ollama_model}")
```

### Custom Settings

To add custom settings:

1. Edit `src/aria/config.py`
2. Add field to `Settings` class
3. Add to `.env.example`

Example:

```python
# In src/aria/config.py
class Settings(BaseSettings):
    # ... existing fields ...

    custom_feature_enabled: bool = Field(
        default=False,
        description="Enable custom feature"
    )
```

```bash
# In .env
CUSTOM_FEATURE_ENABLED=true
```

## Document Classification

Configure recognized persons and categories in `src/aria/config.py`:

```python
class Settings(BaseSettings):
    # Family members (for document classification)
    family_members: list[str] = Field(
        default=["person1", "person2", "person3"],
        description="List of family member names"
    )

    # Document categories
    document_categories: list[str] = Field(
        default=[
            "medical",
            "financial",
            "government",
            "personal",
            "other",
        ],
        description="Document classification categories"
    )
```

## Archive Structure

Documents are archived using this structure:

```
ARCHIVE_BASE_DIR/
├── YYYY/
│   ├── person1/
│   │   ├── medical/
│   │   ├── financial/
│   │   └── government/
│   └── person2/
│       └── ...
```

Configure base directory:

```python
archive_base_dir: Path = Field(
    default=Path.home() / "Documents" / "Archive",
    description="Base directory for archived documents"
)
```

## Performance Tuning

### LLM Performance

- **Lower temperature** (0.1-0.3): More focused, deterministic responses
- **Higher temperature** (0.7-1.0): More creative, varied responses

### Vector Store Performance

```bash
# Larger batch sizes = faster indexing, more memory
CHROMA_BATCH_SIZE=100
```

### Database Performance

ARIA uses SQLite with WAL mode for better concurrency:

```python
# Automatically configured, but you can tune:
SQLITE_CACHE_SIZE=-64000  # 64MB cache
```

## Logging

### Log Levels

- `DEBUG`: Detailed diagnostic information
- `INFO`: General informational messages
- `WARNING`: Warning messages (default)
- `ERROR`: Error messages
- `CRITICAL`: Critical errors

### Log Files

Logs are written to:
- Console (stderr)
- File: `data/logs/aria.log` (rotated daily)

Configure in code:

```python
from aria.logging import get_logger

logger = get_logger("aria.mymodule")
logger.info("Custom log message")
```

## Advanced Configuration

### Custom Embeddings Model

```bash
EMBEDDING_MODEL=nomic-embed-text
```

### Custom OCR Settings

In `src/aria/tools/documents/ocr.py`:

```python
# Tesseract language
OCR_LANGUAGE = "eng"

# Confidence threshold
MIN_CONFIDENCE = 0.6
```

## Environment-Specific Configs

Use different `.env` files for different environments:

```bash
# Development
cp .env.development .env

# Production
cp .env.production .env
```

## Validation

ARIA validates configuration on startup. Invalid settings will raise errors:

```python
ValidationError: 1 validation error for Settings
ollama_timeout
  Input should be less than or equal to 600
```

## Next Steps

- [Quickstart Guide](quickstart.md)
- [User Guide](user_guide/overview.md)
- [Development Guide](development/architecture.md)
