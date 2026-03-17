"""ParallelHandler per Section 4.8 of the Attractor spec.

Fans out to branches concurrently using threads, with configurable
join and error policies.
"""

from __future__ import annotations

import concurrent.futures
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..graph import Outcome, StageStatus
from .base import Handler

if TYPE_CHECKING:
    from ..context import Context
    from ..events import EventEmitter
    from ..graph import Graph, Node


class ParallelHandler(Handler):
    """Executes outgoing branches concurrently.

    Attributes on the node control behaviour:
      - ``join_policy``: ``wait_all`` (default) or ``first_success``
      - ``error_policy``: ``fail_fast`` (default), ``continue``, or ``ignore``
    """

    def execute(
        self,
        node: "Node",
        context: "Context",
        graph: "Graph",
        logs_root: Path | None = None,
        emitter: "EventEmitter | None" = None,
    ) -> Outcome:
        outgoing = graph.outgoing_edges(node.id)
        if not outgoing:
            return Outcome(status=StageStatus.SUCCESS, notes="No branches to fan out.")

        join_policy = node.attrs.get("join_policy", "wait_all")
        error_policy = node.attrs.get("error_policy", "fail_fast")
        # k_of_n: how many successes needed (defaults to majority for quorum)
        k_value = int(node.attrs.get("k", 0))

        max_parallel = int(node.attrs.get("max_parallel", 0)) or len(outgoing)
        branch_targets = [e.to_node for e in outgoing]
        results: dict[str, Outcome] = {}
        lock = threading.Lock()

        def _emit(kind: "PipelineEventKind", **data: Any) -> None:
            if emitter is not None:
                from ..events import PipelineEventKind as _PEK  # noqa: F811
                emitter.emit(kind, **data)

        from ..events import PipelineEventKind
        _emit(PipelineEventKind.PARALLEL_STARTED, node_id=node.id, branch_count=len(branch_targets))

        def _run_branch(target_id: str) -> tuple[str, Outcome]:
            _emit(PipelineEventKind.PARALLEL_BRANCH_STARTED, node_id=node.id, branch_target=target_id)
            target_node = graph.nodes.get(target_id)
            if target_node is None:
                outcome = Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=f"Branch target '{target_id}' not found.",
                )
                _emit(PipelineEventKind.PARALLEL_BRANCH_COMPLETED, node_id=node.id, branch_target=target_id, status="fail")
                return target_id, outcome
            try:
                from . import _get_default_registry
                registry = _get_default_registry()
                handler = registry.resolve(target_node)
                branch_ctx = context.clone()
                outcome = handler.execute(target_node, branch_ctx, graph, logs_root)
                if outcome.context_updates:
                    with lock:
                        context.apply_updates(outcome.context_updates)
                _emit(PipelineEventKind.PARALLEL_BRANCH_COMPLETED, node_id=node.id, branch_target=target_id, status=outcome.status.value)
                return target_id, outcome
            except Exception as exc:
                _emit(PipelineEventKind.PARALLEL_BRANCH_COMPLETED, node_id=node.id, branch_target=target_id, status="fail")
                return target_id, Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=str(exc),
                )

        def _result_map() -> dict[str, str]:
            return {k: v.status.value for k, v in results.items()}

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_parallel
        ) as executor:
            futures = {
                executor.submit(_run_branch, tid): tid
                for tid in branch_targets
            }

            if join_policy == "first_success":
                first_success_outcome: Outcome | None = None
                for future in concurrent.futures.as_completed(futures):
                    tid, outcome = future.result()
                    results[tid] = outcome
                    if outcome.status == StageStatus.SUCCESS and first_success_outcome is None:
                        first_success_outcome = outcome
                        for f in futures:
                            f.cancel()
                        break

                context.set("parallel_results", _result_map())
                if first_success_outcome is not None:
                    _emit(PipelineEventKind.PARALLEL_COMPLETED, node_id=node.id, status="success")
                    return Outcome(
                        status=StageStatus.SUCCESS,
                        notes="First success from parallel branches.",
                        context_updates={"parallel_results": _result_map()},
                    )
                _emit(PipelineEventKind.PARALLEL_COMPLETED, node_id=node.id, status="fail")
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason="No branch succeeded in first_success mode.",
                    context_updates={"parallel_results": _result_map()},
                )

            elif join_policy in ("k_of_n", "quorum"):
                total = len(branch_targets)
                required = k_value if (join_policy == "k_of_n" and k_value > 0) else (total // 2 + 1)
                success_count = 0

                for future in concurrent.futures.as_completed(futures):
                    tid, outcome = future.result()
                    results[tid] = outcome
                    if outcome.status == StageStatus.SUCCESS:
                        success_count += 1
                        if success_count >= required:
                            # We have enough, cancel the rest
                            for f in futures:
                                f.cancel()
                            break

                context.set("parallel_results", _result_map())
                if success_count >= required:
                    _emit(PipelineEventKind.PARALLEL_COMPLETED, node_id=node.id, status="success")
                    return Outcome(
                        status=StageStatus.SUCCESS,
                        notes=f"{success_count}/{total} branches succeeded (needed {required}).",
                        context_updates={"parallel_results": _result_map()},
                    )
                _emit(PipelineEventKind.PARALLEL_COMPLETED, node_id=node.id, status="fail")
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=f"Only {success_count}/{total} branches succeeded (needed {required}).",
                    context_updates={"parallel_results": _result_map()},
                )

            else:
                # wait_all (default)
                for future in concurrent.futures.as_completed(futures):
                    tid, outcome = future.result()
                    results[tid] = outcome
                    if outcome.status == StageStatus.FAIL and error_policy == "fail_fast":
                        for f in futures:
                            f.cancel()
                        context.set("parallel_results", _result_map())
                        _emit(PipelineEventKind.PARALLEL_COMPLETED, node_id=node.id, status="fail")
                        return Outcome(
                            status=StageStatus.FAIL,
                            failure_reason=f"Branch '{tid}' failed: {outcome.failure_reason}",
                            context_updates={"parallel_results": _result_map()},
                        )

        result_map = _result_map()
        context.set("parallel_results", result_map)

        failures = [k for k, v in results.items() if v.status == StageStatus.FAIL]
        if failures and error_policy != "ignore":
            _emit(PipelineEventKind.PARALLEL_COMPLETED, node_id=node.id, status="partial_success")
            return Outcome(
                status=StageStatus.PARTIAL_SUCCESS,
                failure_reason=f"Branches failed: {', '.join(failures)}",
                context_updates={"parallel_results": result_map},
            )

        _emit(PipelineEventKind.PARALLEL_COMPLETED, node_id=node.id, status="success")
        return Outcome(
            status=StageStatus.SUCCESS,
            notes=f"All {len(results)} branches completed.",
            context_updates={"parallel_results": result_map},
        )
