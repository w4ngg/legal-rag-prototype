from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .core import AppConfig
from .services import build_service


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legal-ir-smoke",
        description="Run one online hybrid-retrieval smoke query.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Display at most this many results (does not change retrieval).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    config = AppConfig.from_yaml(args.config)
    results = build_service(config).retrieve(args.query)
    if args.limit is not None:
        results = results[: args.limit]
    print(
        json.dumps(
            {
                "query": args.query.strip(),
                "result_count": len(results),
                "results": [result.to_dict() for result in results],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
