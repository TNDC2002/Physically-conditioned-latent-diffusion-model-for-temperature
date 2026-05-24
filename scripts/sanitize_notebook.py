"""Fix nbformat v4 issues in a notebook before nbconvert --execute.

Some Jupyter clients save stream outputs without ``name`` or display_data without
``metadata``, which makes nbconvert reject the file before execution.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def sanitize_notebook(nb: dict, *, clear_outputs: bool) -> dict:
    for cell in nb.get("cells", []):
        if clear_outputs:
            cell["outputs"] = []
            if cell.get("cell_type") == "code":
                cell["execution_count"] = None
            continue

        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream" and "name" not in out:
                out["name"] = "stdout"
            if out.get("output_type") in ("display_data", "execute_result") and "metadata" not in out:
                out["metadata"] = {}
            if out.get("output_type") == "execute_result" and "execution_count" not in out:
                out["execution_count"] = None
    return nb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument(
        "--clear-outputs",
        action="store_true",
        help="Remove all cell outputs (recommended for notebooks run via Slurm nbconvert).",
    )
    args = parser.parse_args()

    path = args.notebook.resolve()
    nb = json.loads(path.read_text(encoding="utf-8"))
    nb = sanitize_notebook(nb, clear_outputs=args.clear_outputs)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    try:
        import nbformat

        nbformat.validate(nbformat.read(path, as_version=4))
    except ImportError:
        pass

    print(f"Sanitized {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
