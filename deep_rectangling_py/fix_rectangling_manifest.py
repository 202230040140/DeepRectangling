from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Rewrite rectangling job manifest with absolute paths.")
    parser.add_argument("--manifest", default="outputs/stitchbench_general/_rectangling_jobs.json")
    args = parser.parse_args()
    path = Path(args.manifest)
    jobs = json.loads(path.read_text(encoding="utf-8-sig"))
    fixed = [
        {
            "input": str(Path(job["input"]).resolve() if Path(job["input"]).is_absolute() else (REPO_ROOT / job["input"]).resolve()),
            "mask": str(Path(job["mask"]).resolve() if Path(job["mask"]).is_absolute() else (REPO_ROOT / job["mask"]).resolve()),
            "output": str(Path(job["output"]).resolve() if Path(job["output"]).is_absolute() else (REPO_ROOT / job["output"]).resolve()),
        }
        for job in jobs
    ]
    path.write_text(json.dumps(fixed, indent=2), encoding="utf-8")
    print(f"Updated {len(fixed)} jobs in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
