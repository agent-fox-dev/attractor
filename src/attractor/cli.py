"""Attractor CLI - DOT-based pipeline runner for AI workflows."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from attractor.pipeline.engine import PipelineRunner, PipelineConfig
from attractor.pipeline.graph import Outcome
from attractor.pipeline.interviewer import (
    AutoApproveInterviewer,
    ConsoleInterviewer,
)
from attractor.pipeline.events import PipelineEvent


def _print_event(event: PipelineEvent) -> None:
    kind = event.kind.value
    data_str = ""
    if event.data:
        parts = []
        for k, v in event.data.items():
            if isinstance(v, str) and len(v) > 80:
                v = v[:77] + "..."
            parts.append(f"{k}={v}")
        data_str = " " + " ".join(parts)
    print(f"[{kind}]{data_str}")


def run_pipeline(args: argparse.Namespace) -> int:
    dot_path = Path(args.file)
    if not dot_path.exists():
        print(f"Error: file not found: {dot_path}", file=sys.stderr)
        return 1

    dot_source = dot_path.read_text()

    interviewer = (
        AutoApproveInterviewer()
        if args.auto_approve
        else ConsoleInterviewer()
    )

    config = PipelineConfig(
        logs_root=args.logs_dir or f"runs/{dot_path.stem}",
    )

    runner = PipelineRunner(
        interviewer=interviewer,
        event_callback=_print_event if args.verbose else None,
    )

    try:
        outcome = runner.run(dot_source, config=config)
    except Exception as e:
        print(f"Pipeline error: {e}", file=sys.stderr)
        return 1

    print(f"\nPipeline completed: {outcome.status.value}")
    if outcome.notes:
        print(f"Notes: {outcome.notes}")
    if outcome.failure_reason:
        print(f"Failure: {outcome.failure_reason}")

    return 0 if outcome.status.value in ("success", "partial_success") else 1


def validate_pipeline(args: argparse.Namespace) -> int:
    from attractor.pipeline.parser import parse_dot
    from attractor.pipeline.validation import validate, Severity

    dot_path = Path(args.file)
    if not dot_path.exists():
        print(f"Error: file not found: {dot_path}", file=sys.stderr)
        return 1

    dot_source = dot_path.read_text()

    try:
        graph = parse_dot(dot_source)
    except Exception as e:
        print(f"Parse error: {e}", file=sys.stderr)
        return 1

    diagnostics = validate(graph)
    has_errors = False

    for d in diagnostics:
        prefix = {"error": "ERROR", "warning": "WARN", "info": "INFO"}[
            d.severity.value
        ]
        loc = ""
        if d.node_id:
            loc = f" [{d.node_id}]"
        elif d.edge:
            loc = f" [{d.edge[0]} -> {d.edge[1]}]"
        print(f"  {prefix}: {d.message}{loc}")
        if d.severity == Severity.ERROR:
            has_errors = True

    if not diagnostics:
        print("No issues found.")

    return 1 if has_errors else 0


def checkpoint_info(args: argparse.Namespace) -> int:
    from attractor.pipeline.context import Checkpoint

    cp_path = Path(args.file)
    if not cp_path.exists():
        print(f"Error: checkpoint not found: {cp_path}", file=sys.stderr)
        return 1

    cp = Checkpoint.load(cp_path)
    print(f"Current node:    {cp.current_node}")
    print(f"Completed nodes: {', '.join(cp.completed_nodes)}")
    print(f"Logs:            {len(cp.logs)} entries")
    print(f"Context keys:    {', '.join(cp.context_values.keys())}")
    if cp.node_retries:
        print(f"Retries:         {cp.node_retries}")
    return 0


def resume_pipeline(args: argparse.Namespace) -> int:
    from attractor.pipeline.context import Checkpoint

    dot_path = Path(args.file)
    cp_path = Path(args.checkpoint)

    if not dot_path.exists():
        print(f"Error: file not found: {dot_path}", file=sys.stderr)
        return 1
    if not cp_path.exists():
        print(f"Error: checkpoint not found: {cp_path}", file=sys.stderr)
        return 1

    dot_source = dot_path.read_text()
    checkpoint = Checkpoint.load(cp_path)

    config = PipelineConfig(
        logs_root=args.logs_dir or f"runs/{dot_path.stem}",
        resume_from=checkpoint,
    )

    runner = PipelineRunner(
        interviewer=AutoApproveInterviewer() if args.auto_approve else ConsoleInterviewer(),
        event_callback=_print_event if args.verbose else None,
    )

    try:
        outcome = runner.run(dot_source, config=config)
    except Exception as e:
        print(f"Pipeline error: {e}", file=sys.stderr)
        return 1

    print(f"\nPipeline completed: {outcome.status.value}")
    if outcome.failure_reason:
        print(f"Failure: {outcome.failure_reason}")
    return 0 if outcome.status.value in ("success", "partial_success") else 1


def serve_command(args: argparse.Namespace) -> int:
    from attractor.pipeline.server import serve

    print(f"Starting Attractor pipeline server on {args.host}:{args.port}...")
    serve(host=args.host, port=args.port)
    return 0


def parse_only(args: argparse.Namespace) -> int:
    from attractor.pipeline.parser import parse_dot

    dot_path = Path(args.file)
    if not dot_path.exists():
        print(f"Error: file not found: {dot_path}", file=sys.stderr)
        return 1

    dot_source = dot_path.read_text()

    try:
        graph = parse_dot(dot_source)
    except Exception as e:
        print(f"Parse error: {e}", file=sys.stderr)
        return 1

    print(json.dumps(graph.model_dump(), indent=2, default=str))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="attractor",
        description="DOT-based pipeline runner for AI workflows",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Run a pipeline from a DOT file")
    run_parser.add_argument("file", help="Path to the .dot pipeline file")
    run_parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto-approve all human gates",
    )
    run_parser.add_argument(
        "--logs-dir", help="Directory for run logs (default: runs/<name>)"
    )
    run_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print pipeline events"
    )

    # validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate a DOT pipeline file"
    )
    validate_parser.add_argument("file", help="Path to the .dot pipeline file")

    # parse command
    parse_parser = subparsers.add_parser(
        "parse", help="Parse a DOT file and print the graph as JSON"
    )
    parse_parser.add_argument("file", help="Path to the .dot pipeline file")

    # checkpoint command
    cp_parser = subparsers.add_parser(
        "checkpoint", help="Inspect a checkpoint file"
    )
    cp_parser.add_argument("file", help="Path to the checkpoint JSON file")

    # resume command
    resume_parser = subparsers.add_parser(
        "resume", help="Resume a pipeline from a checkpoint"
    )
    resume_parser.add_argument("file", help="Path to the .dot pipeline file")
    resume_parser.add_argument(
        "--checkpoint", required=True, help="Path to the checkpoint JSON file"
    )
    resume_parser.add_argument(
        "--auto-approve", action="store_true", help="Auto-approve all human gates"
    )
    resume_parser.add_argument(
        "--logs-dir", help="Directory for run logs"
    )
    resume_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print pipeline events"
    )

    # serve command
    serve_parser = subparsers.add_parser(
        "serve", help="Start the HTTP server for remote pipeline execution"
    )
    serve_parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    serve_parser.add_argument(
        "--port", type=int, default=8080, help="Port to listen on (default: 8080)"
    )

    args = parser.parse_args()

    if args.command == "run":
        sys.exit(run_pipeline(args))
    elif args.command == "validate":
        sys.exit(validate_pipeline(args))
    elif args.command == "parse":
        sys.exit(parse_only(args))
    elif args.command == "checkpoint":
        sys.exit(checkpoint_info(args))
    elif args.command == "resume":
        sys.exit(resume_pipeline(args))
    elif args.command == "serve":
        sys.exit(serve_command(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
