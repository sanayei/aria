# ARIA Documentation Guide

## Overview

ARIA now has a **professional, comprehensive documentation system** similar to pandas, numpy, and other major Python projects.

**Features:**
- ✅ Auto-generated API documentation from docstrings
- ✅ User guides and tutorials
- ✅ Beautiful Read the Docs theme
- ✅ Multiple output formats (HTML, PDF, ePub)
- ✅ Searchable documentation
- ✅ Can be hosted online (ReadTheDocs) or browsed offline
- ✅ Markdown + ReStructuredText support

## Quick Start

### View Documentation Locally

```bash
# Build the documentation
bash scripts/build_docs.sh

# Open in browser (Linux)
xdg-open docs/build/html/index.html

# Or start a web server
cd docs/build/html
python -m http.server 8080
# Visit: http://localhost:8080
```

### Build Documentation

```bash
# HTML (most common)
cd docs
uv run sphinx-build -b html source build/html

# PDF (requires LaTeX)
uv run sphinx-build -b latex source build/latex
cd build/latex && latexmk -pdf ARIA.tex

# ePub (eBook format)
uv run sphinx-build -b epub source build/epub
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Generator** | Sphinx 8.2 | Industry standard for Python documentation |
| **Theme** | sphinx-rtd-theme | Read the Docs responsive theme |
| **Markdown** | MyST-Parser | Write docs in Markdown instead of RST |
| **API Docs** | autodoc + napoleon | Auto-generate from Python docstrings |
| **Type Hints** | sphinx-autodoc-typehints | Beautiful type annotations |
| **Hosting** | ReadTheDocs | Free hosting with auto-builds |

## Documentation Structure

```
docs/
├── source/
│   ├── index.rst                # Main landing page
│   ├── installation.md          # Installation instructions
│   ├── quickstart.md            # 5-minute quickstart
│   ├── configuration.md         # Configuration reference
│   │
│   ├── api/                     # Auto-generated API docs
│   │   ├── agent.rst           # Agent module
│   │   ├── tools.rst           # Tools module
│   │   ├── memory.rst          # Memory module
│   │   └── web.rst             # Web application
│   │
│   ├── user_guide/              # How-to guides
│   │   ├── overview.md
│   │   ├── file_management.md
│   │   ├── document_analysis.md
│   │   ├── email_management.md
│   │   └── web_interface.md
│   │
│   ├── tutorials/               # Step-by-step tutorials
│   │   ├── scanning_documents.md
│   │   ├── setting_up_gmail.md
│   │   ├── using_web_interface.md
│   │   └── custom_workflows.md
│   │
│   └── development/             # Developer documentation
│       ├── architecture.md
│       ├── contributing.md
│       ├── testing.md
│       └── extending.md
│
└── build/                       # Generated output (gitignored)
    ├── html/                    # HTML documentation
    ├── latex/                   # LaTeX source for PDF
    └── epub/                    # ePub eBook
```

## Hosting on ReadTheDocs (Free!)

### Setup Steps:

1. **Create ReadTheDocs Account**
   - Go to https://readthedocs.org
   - Sign in with GitHub

2. **Import Repository**
   - Click "Import a Project"
   - Select your ARIA repository
   - Click "Import"

3. **Configuration**
   - Configuration is automatic via `.readthedocs.yaml`
   - Builds on every commit to main/master
   - Provides URL like: `https://aria.readthedocs.io`

4. **Multiple Versions**
   - `latest`: Latest commit
   - `stable`: Latest release
   - Per-version docs (e.g., `v0.1.0`)

### Features:
- Automatic rebuilds on git push
- PR previews
- Version management
- Custom domain support
- Search functionality
- Download as PDF/ePub

## Writing Documentation

### Markdown Pages

Create `.md` files using MyST-Parser enhanced Markdown:

```markdown
# Page Title

Regular markdown text with **bold** and *italic*.

## Code Examples

```python
from aria.config import get_settings

settings = get_settings()
print(settings.ollama_model)
```

## Admonitions

:::{note}
This is a note box
:::

:::{warning}
This is a warning box
:::

:::{tip}
This is a tip box
:::

## Lists

- Item 1
- Item 2
  - Nested item

## Links

[Link text](https://example.com)
[Internal link](configuration.md)
```

### API Documentation

API docs are auto-generated from docstrings:

```python
def process_document(file_path: Path, category: str) -> ProcessedDocument:
    """Process a scanned document.

    This function performs OCR, classification, and archiving.

    Args:
        file_path: Path to the PDF document
        category: Document category (medical, financial, etc.)

    Returns:
        ProcessedDocument containing extracted information

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If category is invalid

    Example:
        >>> result = process_document(
        ...     Path("scan.pdf"),
        ...     category="medical"
        ... )
        >>> print(result.person)
        'john'

    Note:
        Requires Tesseract OCR to be installed.

    See Also:
        - :func:`classify_document`
        - :class:`ProcessedDocument`
    """
    ...
```

**Supported Styles:**
- Google-style (shown above) ✅
- NumPy-style ✅
- Sphinx-style ✅

### Adding New Pages

1. **Create the file:**
   ```bash
   # For user guide
   touch docs/source/user_guide/new_feature.md

   # For tutorial
   touch docs/source/tutorials/my_tutorial.md
   ```

2. **Add to table of contents:**

   Edit the parent index file (e.g., `docs/source/index.rst`):

   ```rst
   .. toctree::
      :maxdepth: 2
      :caption: User Guide

      user_guide/overview
      user_guide/new_feature  # Add this line
   ```

3. **Write content:**
   Use Markdown or ReStructuredText

4. **Rebuild:**
   ```bash
   bash scripts/build_docs.sh
   ```

## Live Editing

For real-time preview while editing:

```bash
# Install autobuild
uv add --dev sphinx-autobuild

# Start live server
uv run sphinx-autobuild docs/source docs/build/html

# Visit: http://localhost:8000
# Automatically refreshes on file changes!
```

## Output Formats

### HTML (Default)
- Responsive design
- Searchable
- Sidebar navigation
- Code highlighting
- Mobile-friendly

**Use case:** Online hosting, local browsing

### PDF
- Professional layout
- Table of contents
- Index
- Cross-references

**Use case:** Printable manual, offline distribution

### ePub
- eBook format
- Readable on Kindle, Apple Books, etc.
- Reflowable text

**Use case:** Reading on e-readers, tablets

## Configuration

Main configuration: `docs/source/conf.py`

### Key Settings:

```python
# Project info
project = "ARIA"
release = "0.1.0"

# Theme
html_theme = "sphinx_rtd_theme"

# Extensions
extensions = [
    "sphinx.ext.autodoc",        # Auto API docs
    "sphinx.ext.napoleon",       # Google/NumPy docstrings
    "sphinx.ext.viewcode",       # Source code links
    "sphinx.ext.intersphinx",    # Link to other docs
    "sphinx_autodoc_typehints",  # Type hints
    "myst_parser",               # Markdown support
]

# Autodoc options
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
}
```

### Customization:

**Theme colors:**
```python
html_theme_options = {
    "style_external_links": True,
    "navigation_depth": 4,
}
```

**Custom CSS:**
```python
html_css_files = ["custom.css"]
```

**Logo:**
```python
html_logo = "_static/logo.png"
```

## Best Practices

### 1. Write Good Docstrings

```python
# ❌ Bad
def func(x):
    return x + 1

# ✅ Good
def increment(value: int) -> int:
    """Add one to the input value.

    Args:
        value: The number to increment

    Returns:
        The incremented value
    """
    return value + 1
```

### 2. Organize by Audience

- **Installation/Quickstart**: For new users
- **User Guide**: Task-oriented how-tos
- **Tutorials**: Learning-oriented lessons
- **API Reference**: Information-oriented details
- **Development**: For contributors

### 3. Use Examples

```python
"""
Example:
    Basic usage::

        >>> from aria import Agent
        >>> agent = Agent()
        >>> agent.process("hello")
        'Hello! How can I help?'

    Advanced usage::

        >>> agent = Agent(model="qwen3:32b")
        >>> result = agent.process(
        ...     "Search for Python tutorials",
        ...     tools=["web_search"]
        ... )
"""
```

### 4. Cross-Reference

```rst
See :class:`aria.agent.core.Agent` for details.
See :func:`aria.tools.documents.process_document`.
See :doc:`configuration` for settings.
```

### 5. Keep It Updated

- Update docs with code changes
- Review docs in PR reviews
- Set up CI to build docs on commits

## CI/CD Integration

### GitHub Actions

Create `.github/workflows/docs.yml`:

```yaml
name: Documentation

on:
  push:
    branches: [main, master]
  pull_request:

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install uv
          uv sync
      - name: Build docs
        run: |
          cd docs
          uv run sphinx-build -W -b html source build/html
      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: documentation
          path: docs/build/html/
```

## Comparison with Other Systems

| Feature | Sphinx | MkDocs | Docusaurus |
|---------|--------|--------|------------|
| Python-focused | ✅ Excellent | ⚠️ Good | ❌ No |
| Auto API docs | ✅ Excellent | ⚠️ Plugins | ⚠️ Manual |
| Themes | ✅ Many | ✅ Many | ✅ React |
| Markdown | ✅ Via MyST | ✅ Native | ✅ MDX |
| PDF output | ✅ Built-in | ⚠️ Plugin | ❌ No |
| ReadTheDocs | ✅ Native | ✅ Supported | ⚠️ Manual |
| Maturity | ✅ 15+ years | ⚠️ Newer | ⚠️ Newer |

**Why Sphinx for ARIA:**
- ✅ Python ecosystem standard
- ✅ Best autodoc capabilities
- ✅ Multiple output formats
- ✅ Used by pandas, numpy, Django, etc.
- ✅ ReadTheDocs first-class support

## Next Steps

1. **Expand Content**
   - Fill in tutorial pages
   - Add more user guide sections
   - Write development documentation

2. **Add Screenshots**
   - Web interface screenshots
   - CLI examples
   - Workflow diagrams

3. **Set Up ReadTheDocs**
   - Import repository
   - Configure webhooks
   - Set up custom domain

4. **Write More Examples**
   - Add code examples to docstrings
   - Create example scripts
   - Add Jupyter notebooks

5. **Improve API Docs**
   - Review auto-generated docs
   - Add missing docstrings
   - Improve type hints

## Resources

- **Sphinx:** https://www.sphinx-doc.org/
- **MyST Parser:** https://myst-parser.readthedocs.io/
- **Read the Docs:** https://docs.readthedocs.io/
- **Good Examples:**
  - Pandas: https://pandas.pydata.org/docs/
  - Requests: https://requests.readthedocs.io/
  - FastAPI: https://fastapi.tiangolo.com/

## Support

For documentation issues:
- Check [docs/README.md](docs/README.md)
- Review Sphinx documentation
- Ask in GitHub discussions
