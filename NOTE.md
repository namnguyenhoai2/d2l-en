# Local HTML build

From Git Bash, run:

```bash
bash scripts/build-html.sh
```

The script does not install or update anything. It requires `d2lbook` to be in
the active environment or at `.venv/Scripts/d2lbook.exe`. It builds every
framework tab without evaluating notebooks, then writes the preview to
`_build/html/index.html`. `config.ini` is restored automatically.

Useful options:

```bash
# Delete previous HTML/RST output first
bash scripts/build-html.sh --clean

# Re-evaluate notebooks (slow; requires framework dependencies)
bash scripts/build-html.sh --evaluate
```
