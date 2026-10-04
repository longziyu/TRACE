"""Parse source files without importing packages or executing training/deployment.

Run this from the code directory. No third-party packages are required.
"""
import ast
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    failures = []
    files = sorted((root/"src").rglob("*.py")) + sorted((root/"tools").rglob("*.py"))
    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            failures.append(str(exc))
    if failures:
        print("\n".join(failures))
        return 1
    print(f"Parsed {len(files)} Python files. No modules imported; no pipelines executed.")
    print("This checks syntax only, not runtime compatibility or numerical correctness.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
