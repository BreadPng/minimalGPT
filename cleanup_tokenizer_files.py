#!/usr/bin/env python3
import os
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    targets = [
        root / "sp_model.model",
        root / "sp_model.vocab",
        root / "out" / "chargpt" / "tokenized_data.pkl",
        root / "out" / "chargpt" / "tokenizer.model",
        root / "out" / "chargpt" / "tokenizer_metadata.pkl",
    ]

    exit_code = 0
    for path in targets:
        try:
            if path.exists():
                path.unlink()
                print(f"Deleted: {path}")
            else:
                print(f"Not found (skipped): {path}")
        except Exception as e:
            print(f"Failed to delete {path}: {e}")
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


