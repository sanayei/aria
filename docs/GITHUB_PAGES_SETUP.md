# GitHub Pages Setup for ARIA Documentation

The documentation is now included in the git repository and can be hosted on GitHub Pages.

## Quick Setup

### 1. Push Documentation to GitHub

```bash
# Stage all documentation files
git add docs/

# Commit
git commit -m "docs: Add built documentation for GitHub Pages"

# Push
git push origin master
```

### 2. Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** → **Pages**
3. Under **Source**, select:
   - **Branch**: `master` (or `main`)
   - **Folder**: `/docs/build/html`
4. Click **Save**

### 3. Access Your Documentation

After a few minutes, your documentation will be available at:

```
https://<username>.github.io/<repository>/
```

For example: `https://yourusername.github.io/aria/`

## Alternative: Custom Domain

### Set Up Custom Domain

1. In **Settings** → **Pages**:
   - Enter your custom domain (e.g., `docs.aria.example.com`)
   - Click **Save**

2. Configure DNS:
   - Add a CNAME record pointing to `<username>.github.io`

3. Enable HTTPS:
   - Check **Enforce HTTPS** in Settings → Pages

## Updating Documentation

Every time you rebuild the documentation, commit and push:

```bash
# Rebuild documentation
bash scripts/build_docs.sh

# Add changes
git add docs/build/html/

# Commit
git commit -m "docs: Update documentation"

# Push
git push origin master
```

GitHub Pages will automatically update within a few minutes.

## Automation with GitHub Actions

For automatic documentation builds on every commit, create `.github/workflows/docs.yml`:

```yaml
name: Build and Deploy Documentation

on:
  push:
    branches: [ master, main ]

jobs:
  build-docs:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install uv
          uv sync

      - name: Build documentation
        run: |
          cd docs
          uv run sphinx-build -b html source build/html
          touch build/html/.nojekyll

      - name: Commit documentation
        run: |
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add docs/build/html/
          git diff --quiet && git diff --staged --quiet || git commit -m "docs: Auto-update documentation [skip ci]"
          git push
```

This will:
1. Build docs on every push
2. Commit the updated HTML
3. Deploy automatically to GitHub Pages

## Directory Structure

```
docs/
├── build/
│   └── html/               # Served by GitHub Pages
│       ├── .nojekyll       # Required for GitHub Pages
│       ├── index.html      # Landing page
│       ├── _static/        # CSS, JS, images
│       ├── _modules/       # Source code links
│       └── api/            # API documentation
└── source/                 # Source files (not served)
```

## Important Files

- **`.nojekyll`**: Tells GitHub Pages not to use Jekyll
  - Required for serving `_static` and `_modules` folders

- **`CNAME`** (optional): For custom domain
  ```bash
  echo "docs.aria.example.com" > docs/build/html/CNAME
  git add docs/build/html/CNAME
  ```

## Troubleshooting

### Pages Not Updating

1. Check GitHub Actions tab for errors
2. Verify branch and folder settings in Settings → Pages
3. Clear browser cache
4. Wait a few minutes for deployment

### 404 Errors

1. Ensure `.nojekyll` exists in `docs/build/html/`
2. Check that `index.html` exists
3. Verify folder path is `/docs/build/html`

### Styling Not Loading

1. Verify `.nojekyll` is present
2. Check that `_static/` folder is committed
3. Rebuild documentation: `bash scripts/build_docs.sh`

### Custom Domain Not Working

1. Check DNS propagation: `dig docs.aria.example.com`
2. Verify CNAME file contains correct domain
3. Wait up to 24 hours for DNS propagation

## Comparison: GitHub Pages vs ReadTheDocs

| Feature | GitHub Pages | ReadTheDocs |
|---------|--------------|-------------|
| Setup | Manual enable | Import repo |
| Auto-build | Requires Actions | Built-in |
| Versioning | Manual | Automatic |
| Search | Basic | Advanced |
| PDF/ePub | No | Yes |
| Custom domain | Free | Free |
| Speed | Fast | Fast |
| Cost | Free | Free |

**Recommendation:**
- **GitHub Pages**: For simple hosting, already using GitHub
- **ReadTheDocs**: For versioning, PDF/ePub, advanced features

## Best Practices

1. **Keep docs/build/ in git**
   - Ensures immediate availability
   - No build step needed on server

2. **Rebuild before committing code changes**
   ```bash
   bash scripts/build_docs.sh
   git add docs/build/html/
   ```

3. **Use GitHub Actions for automation**
   - Prevents forgetting to rebuild
   - Ensures docs stay in sync with code

4. **Add `.nojekyll` to every build**
   ```bash
   # In scripts/build_docs.sh
   touch docs/build/html/.nojekyll
   ```

## Switching to ReadTheDocs Later

If you want to switch to ReadTheDocs:

1. **Stop tracking docs/build/**:
   ```bash
   # Edit .gitignore
   echo "docs/build/" >> .gitignore

   # Remove from git
   git rm -r --cached docs/build/
   git commit -m "docs: Remove built docs, switching to ReadTheDocs"
   ```

2. **Set up ReadTheDocs**:
   - Import repository at https://readthedocs.org
   - Uses `.readthedocs.yaml` (already configured)
   - Auto-builds on every commit

3. **Update documentation**:
   - No need to commit builds anymore
   - ReadTheDocs handles everything

## Resources

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [GitHub Actions for Sphinx](https://github.com/marketplace/actions/sphinx-build)
- [Custom Domains on GitHub Pages](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site)
