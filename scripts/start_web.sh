#!/bin/bash
# Start ARIA web application

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Starting ARIA Web Application..."
echo "Access at: http://localhost:8000"
echo "Default login: username=admin, password=admin123"
echo ""

uv run uvicorn aria.web.app:app --host 0.0.0.0 --port 8000 --reload
