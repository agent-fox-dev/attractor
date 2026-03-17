"""Handler package -- provides the default handler registry."""

from __future__ import annotations

from .base import Handler, HandlerRegistry, SHAPE_TO_TYPE, infer_type

_DEFAULT_REGISTRY: HandlerRegistry | None = None


def _get_default_registry() -> HandlerRegistry:
    """Return (and lazily create) the default handler registry."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = _build_default_registry()
    return _DEFAULT_REGISTRY


def _build_default_registry() -> HandlerRegistry:
    """Create a HandlerRegistry with all built-in handlers registered."""
    from .start import StartHandler
    from .exit_handler import ExitHandler
    from .codergen import CodergenHandler
    from .human import WaitForHumanHandler
    from .conditional import ConditionalHandler
    from .parallel import ParallelHandler
    from .fan_in import FanInHandler
    from .tool_handler import ToolHandler
    from .manager import ManagerLoopHandler

    registry = HandlerRegistry()

    # Start
    start = StartHandler()
    registry.register("start", start)

    # Exit
    exit_h = ExitHandler()
    registry.register("exit", exit_h)

    # LLM / Codergen
    codergen = CodergenHandler()
    registry.register("llm", codergen)
    registry.register("codergen", codergen)
    registry.register("coder", codergen)

    # Human
    human = WaitForHumanHandler()
    registry.register("human", human)
    registry.register("wait_for_human", human)
    registry.register("wait.human", human)

    # Conditional
    conditional = ConditionalHandler()
    registry.register("conditional", conditional)
    registry.register("decision", conditional)

    # Parallel
    parallel = ParallelHandler()
    registry.register("parallel", parallel)
    registry.register("fork", parallel)

    # Fan-in
    fan_in = FanInHandler()
    registry.register("fan_in", fan_in)
    registry.register("join", fan_in)
    registry.register("parallel.fan_in", fan_in)

    # Tool
    tool = ToolHandler()
    registry.register("tool", tool)

    # Manager
    manager = ManagerLoopHandler()
    registry.register("manager", manager)
    registry.register("manager_loop", manager)
    registry.register("stack.manager_loop", manager)

    return registry


__all__ = [
    "Handler",
    "HandlerRegistry",
    "SHAPE_TO_TYPE",
    "infer_type",
    "_get_default_registry",
    "_build_default_registry",
]
