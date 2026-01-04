#!/bin/bash
# Build ARIA documentation

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DOCS_DIR="$PROJECT_DIR/docs"

cd "$DOCS_DIR"

echo "=========================================="
echo "Building ARIA Documentation"
echo "=========================================="
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/
echo "✓ Cleaned"
echo ""

# Build HTML documentation
echo "Building HTML documentation..."
uv run sphinx-build -b html source build/html
echo "✓ HTML documentation built"
echo ""

# Build PDF documentation (optional)
if command -v latexmk &> /dev/null; then
    echo "Building PDF documentation..."
    uv run sphinx-build -b latex source build/latex
    cd build/latex
    latexmk -pdf -quiet ARIA.tex
    echo "✓ PDF documentation built"
    cd ../..
else
    echo "⚠ latexmk not found, skipping PDF generation"
    echo "  Install with: sudo apt-get install latexmk texlive-latex-extra"
fi

echo ""
echo "=========================================="
echo "Documentation Build Complete!"
echo "=========================================="
echo ""
echo "View HTML documentation:"
echo "  Open: file://$DOCS_DIR/build/html/index.html"
echo "  Or run: python -m http.server -d build/html"
echo ""

if [ -f "build/latex/ARIA.pdf" ]; then
    echo "PDF documentation:"
    echo "  Location: $DOCS_DIR/build/latex/ARIA.pdf"
    echo ""
fi
