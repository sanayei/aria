# Installation

## Requirements

ARIA is designed to run on a high-end personal PC with the following minimum specifications:

- **RAM**: 64 GB (recommended)
- **CPU**: Multi-core processor (AMD Ryzen 9 or Intel equivalent)
- **GPU**: NVIDIA GPU with 24GB VRAM (for local LLM inference)
- **OS**: Windows with WSL2 (Ubuntu) or native Linux

## Prerequisites

1. **Python 3.11+**
   ```bash
   python --version  # Should be 3.11 or higher
   ```

2. **UV Package Manager**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Ollama** (for local LLM inference)
   - **Windows**: Download from [ollama.com](https://ollama.com)
   - **Linux**:
     ```bash
     curl -fsSL https://ollama.com/install.sh | sh
     ```

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/aria.git
cd aria
```

### 2. Install Dependencies

```bash
uv sync
```

This will:
- Create a virtual environment in `.venv/`
- Install all dependencies from `pyproject.toml`
- Set up development tools

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and configure:

```bash
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:30b-a3b

# Web Application
JWT_SECRET_KEY=<generate-with-secrets-command>

# Data Directories
ARIA_DATA_DIR=./data
```

Generate a secure JWT secret:
```bash
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

### 4. Pull LLM Model

```bash
ollama pull qwen3:30b-a3b
```

### 5. Initialize Database

The database will be automatically initialized on first run.

## Verify Installation

```bash
# Test CLI
uv run aria --help

# Test web server
uv run uvicorn aria.web.app:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000 in your browser.

Default login:
- **Username**: admin
- **Password**: admin123 (change immediately!)

## Troubleshooting

### Ollama Connection Issues

**WSL to Windows Ollama:**
```bash
# In .env, try:
OLLAMA_HOST=http://host.docker.internal:11434
# Or your Windows IP:
OLLAMA_HOST=http://192.168.1.50:11434
```

Test connection:
```bash
curl http://host.docker.internal:11434/api/tags
```

### GPU Not Detected

Ensure NVIDIA drivers are installed:
```bash
nvidia-smi
```

### Import Errors

Reinstall dependencies:
```bash
uv sync --reinstall
```

## Next Steps

- [Quickstart Guide](quickstart.md)
- [Configuration](configuration.md)
- [User Guide](user_guide/overview.md)
