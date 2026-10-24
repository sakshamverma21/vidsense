"""Fail closed when likely credentials are present in source files.

This scanner intentionally reports file names and line numbers only; it never
prints the matched secret. It is suitable for a local pre-commit hook or CI.
Runtime .env files are excluded because they are injected locally/deployed as
secrets and are already protected by .gitignore and .dockerignore.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "env", "node_modules", ".vidsense_cache", "__pycache__"}
SKIP_FILES = {".env", ".env.local", ".env.production"}
PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"gsk_[0-9A-Za-z_-]{20,}"),
    re.compile(r"tvly-[0-9A-Za-z_-]{20,}"),
    re.compile(r"mongodb(?:\+srv)?://[^\s:/]+:[^\s@]+@"),
    re.compile(
        r"(?:GOOGLE_API_KEY|GROQ_API_KEY|TAVILY_API_KEY|SUPADATA_API_KEY|SMTP_PASSWORD)\s*=\s*"
        r"(?!(?:your_|YOUR_|secretref:|os\.getenv\(|\"\"|''))[^\s#]+"
    ),
]


def files_to_scan() -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard"],
            check=True,
            capture_output=True,
            text=True,
        )
        return [ROOT / line for line in result.stdout.splitlines() if line]
    except (OSError, subprocess.CalledProcessError):
        return [p for p in ROOT.rglob("*") if p.is_file()]


def main() -> int:
    findings: list[tuple[str, int]] = []
    for path in files_to_scan():
        if not path.is_file() or path.name in SKIP_FILES or ".env." in path.name and path.name != ".env.example":
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if any(pattern.search(line) for pattern in PATTERNS):
                findings.append((str(path.relative_to(ROOT)), number))

    if findings:
        print("Potential credential(s) found. Values are intentionally redacted:")
        for filename, line in findings:
            print(f"- {filename}:{line}")
        return 1
    print("Secret scan passed; no likely credentials found in scanned files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
