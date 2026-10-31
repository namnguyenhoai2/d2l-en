# Local HTML build

From Git Bash, run:

```bash
bash scripts/build-html.sh
```

The script does not install or update anything. It requires `d2lbook` to be in
the active environment or at `.venv/Scripts/d2lbook.exe`. It builds every
framework tab, merges them, then writes the preview to `_build/html/index.html`.
By default it converts Markdown to notebooks without running notebook cells.
`config.ini` is restored automatically.
The script sets `PYTHONUTF8=1`, which is required on Windows for Vietnamese
text in generated Sphinx files. It also loads a repository-local compatibility
shim for this legacy `d2lbook` release's Windows path handling; it does not
modify the installed package in `.venv`.

Pandoc is also required by `nbconvert`. Install it manually if it is not
available, then restart Git Bash so its executable is on `PATH`:

```bash
winget install --id JohnMacFarlane.Pandoc -e
pandoc --version
```

This historical `d2lbook` release depends on `notedown`, which imports
`pkg_resources`. Setuptools 82 removed that module. If the script reports this
compatibility error, run the following once yourself (the script never installs
it automatically):

```bash
uv pip install --system-certs --python .venv/Scripts/python.exe "setuptools<82"
```

Useful options:

```bash
# Delete previous HTML/RST output first
bash scripts/build-html.sh --clean

# Re-evaluate notebooks (slow; requires framework dependencies)
bash scripts/build-html.sh --evaluate
```
