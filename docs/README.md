# ARIA Documentation

Comprehensive documentation for ARIA - AI Research & Intelligence Assistant.

## Building Documentation Locally

### Prerequisites

Documentation dependencies are included in the dev dependencies:

```bash
uv sync
```

### Build HTML Documentation

```bash
# Using the build script
bash scripts/build_docs.sh

# Or manually
cd docs
uv run sphinx-build -b html source build/html
```

View the documentation:

```bash
# Option 1: Direct file access
open build/html/index.html  # macOS
xdg-open build/html/index.html  # Linux

# Option 2: Local web server
cd build/html
python -m http.server 8080
# Visit http://localhost:8080
```

### Build PDF Documentation

Requires LaTeX:

```bash
# Install LaTeX (Ubuntu/Debian)
sudo apt-get install latexmk texlive-latex-extra

# Build PDF
cd docs
uv run sphinx-build -b latex source build/latex
cd build/latex
latexmk -pdf ARIA.tex
```

Output: `docs/build/latex/ARIA.pdf`

### Build ePub (eBook)

```bash
cd docs
uv run sphinx-build -b epub source build/epub
```

Output: `docs/build/epub/ARIA.epub`

## Online Documentation

The documentation is automatically built and hosted on Read the Docs:

**URL:** https://aria.readthedocs.io (once configured)

### Setting Up Read the Docs

1. Go to https://readthedocs.org
2. Sign in with GitHub
3. Import the ARIA repository
4. Configuration is automatic (uses `.readthedocs.yaml`)

Documentation will auto-build on every commit to `main`/`master`.

## Documentation Structure

```
docs/
├── README.md                    # This file
├── Makefile                     # Build automation (Unix)
├── make.bat                     # Build automation (Windows)
├── source/                      # Source files
│   ├── conf.py                  # Sphinx configuration
│   ├── index.rst                # Main index
│   ├── installation.md          # Installation guide
│   ├── quickstart.md            # Quick start guide
│   ├── configuration.md         # Configuration reference
│   ├── api/                     # API reference
│   │   ├── agent.rst
│   │   ├── tools.rst
│   │   ├── memory.rst
│   │   └── web.rst
│   ├── user_guide/              # User guides
│   │   ├── overview.md
│   │   ├── file_management.md
│   │   ├── document_analysis.md
│   │   ├── email_management.md
│   │   └── web_interface.md
│   ├── tutorials/               # Step-by-step tutorials
│   │   ├── scanning_documents.md
│   │   ├── setting_up_gmail.md
│   │   ├── using_web_interface.md
│   │   └── custom_workflows.md
│   └── development/             # Developer guides
│       ├── architecture.md
│       ├── contributing.md
│       ├── testing.md
│       └── extending.md
└── build/                       # Build output (gitignored)
    ├── html/
    ├── latex/
    └── epub/
```

## Writing Documentation

### Format

- Configuration files: ReStructuredText (.rst)
- Content pages: Markdown (.md) - easier to write
- API docs: Auto-generated from Python docstrings

### Markdown Features

MyST-Parser enables enhanced Markdown:

````markdown
# Headings work as expected

## Code Blocks

```python
def example():
    return "Hello"
```

## Admonitions

:::{note}
This is a note
:::

:::{warning}
This is a warning
:::

## Task Lists

- [x] Completed task
- [ ] Pending task
````

### Adding New Pages

1. Create the `.md` or `.rst` file in the appropriate directory
2. Add it to a `toctree` in a parent document
3. Rebuild the documentation

Example - Adding a new tutorial:

```rst
.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/scanning_documents
   tutorials/my_new_tutorial
```

### API Documentation

API docs are auto-generated from docstrings:

```python
def my_function(param: str) -> int:
    """Short description.

    Longer description with more details.

    Args:
        param: Description of parameter

    Returns:
        Description of return value

    Raises:
        ValueError: When something goes wrong

    Example:
        >>> my_function("test")
        42
    """
    return 42
```

Supports both Google-style and NumPy-style docstrings.

## Configuration

Main configuration in `source/conf.py`:

- **Theme**: `sphinx_rtd_theme` (Read the Docs)
- **Extensions**: autodoc, napoleon, viewcode, intersphinx, myst-parser
- **Autodoc**: Auto-generate API docs from code
- **Napoleon**: Parse Google/NumPy docstrings
- **MyST**: Markdown support
- **Intersphinx**: Link to other projects (Python, pandas, etc.)

## Continuous Documentation

### On Every Commit

- Read the Docs automatically rebuilds
- HTML, PDF, and ePub are generated
- Multiple versions (stable, latest, releases)

### Local Development

Use auto-build for live editing:

```bash
# Install sphinx-autobuild
uv add --dev sphinx-autobuild

# Start auto-rebuild server
uv run sphinx-autobuild docs/source docs/build/html
```

Visit http://localhost:8000 - refreshes automatically on save!

## Troubleshooting

### Import Errors

Ensure the project is importable:

```python
# In conf.py
sys.path.insert(0, os.path.abspath("../../src"))
```

### Missing Dependencies

```bash
uv sync  # Reinstall all dependencies
```

### Build Warnings

Review warnings in build output:

```bash
uv run sphinx-build -b html source build/html -W
# -W treats warnings as errors
```

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [MyST Parser](https://myst-parser.readthedocs.io/)
- [Read the Docs](https://docs.readthedocs.io/)
- [ReStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)

## Contributing

See [Development > Contributing](source/development/contributing.md) for guidelines on improving documentation.
