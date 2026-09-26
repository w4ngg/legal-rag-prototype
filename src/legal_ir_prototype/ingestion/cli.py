from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import IngestionConfig
from .pipeline import prepare_pipeline, run_pipeline
from .qdrant import ingest_direct
from .snapshot import create_local_snapshot_bundle, restore_remote_snapshot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legal-ir-ingest",
        description="Build/migrate the long-chunk hybrid Qdrant collection.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare", help="raw JSON -> long chunks + corpus BM25 build state"
    )
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--raw-dir", required=True)
    prepare.add_argument("--work-dir", required=True)
    prepare.add_argument("--collection-name")
    prepare.add_argument("--overwrite", action="store_true")

    ingest = commands.add_parser(
        "ingest", help="encode chunks and upsert dense+sparse vectors directly"
    )
    ingest.add_argument("--config", required=True)
    ingest.add_argument("--chunks", required=True)
    ingest.add_argument("--state-dir", required=True)
    ingest.add_argument("--collection-name")
    ingest.add_argument("--rebuild-sparse", action="store_true")

    run = commands.add_parser("run", help="raw JSON -> hybrid Qdrant end to end")
    run.add_argument("--config", required=True)
    run.add_argument("--raw-dir", required=True)
    run.add_argument("--work-dir", required=True)
    run.add_argument("--collection-name")
    run.add_argument("--overwrite-preparation", action="store_true")

    create = commands.add_parser(
        "snapshot-create", help="create/download a local Qdrant snapshot"
    )
    create.add_argument("--config", required=True)
    create.add_argument("--output-dir", required=True)
    create.add_argument("--collection-name")

    restore = commands.add_parser(
        "snapshot-restore", help="restore an HTTP(S)/presigned-S3 snapshot on Qdrant"
    )
    restore.add_argument("--config", required=True)
    restore.add_argument("--snapshot-url", required=True)
    restore.add_argument("--target-collection", required=True)
    restore.add_argument("--checksum")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = IngestionConfig.from_yaml(Path(args.config))
    if args.command == "prepare":
        result = prepare_pipeline(
            args.raw_dir,
            args.work_dir,
            config,
            collection_name=args.collection_name,
            overwrite=args.overwrite,
        )
        print(result)
        return 0
    if args.command == "ingest":
        result = ingest_direct(
            args.chunks,
            args.state_dir,
            config.runtime,
            config.index,
            config.upload,
            collection_name=args.collection_name,
            rebuild_sparse=args.rebuild_sparse,
        )
        print(result)
        return 0
    if args.command == "run":
        result = run_pipeline(
            args.raw_dir,
            args.work_dir,
            config,
            collection_name=args.collection_name,
            overwrite_preparation=args.overwrite_preparation,
        )
        print(result)
        return 0
    if args.command == "snapshot-create":
        result = create_local_snapshot_bundle(
            config.runtime,
            args.output_dir,
            collection_name=args.collection_name,
        )
        print(result)
        return 0
    if args.command == "snapshot-restore":
        result = restore_remote_snapshot(
            config.runtime,
            snapshot_url=args.snapshot_url,
            target_collection=args.target_collection,
            checksum=args.checksum,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
