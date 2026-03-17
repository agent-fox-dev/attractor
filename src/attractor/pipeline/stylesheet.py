"""Model stylesheet parser and applicator per Section 8 of the Attractor spec.

Stylesheets use a CSS-like syntax to assign model properties to nodes:

    * { llm_model: "claude-sonnet-4-20250514"; }
    .planning { reasoning_effort: "high"; }
    #code_review { llm_model: "claude-opus-4-20250514"; }

Selector specificity: * (0) < .class (1) < #id (2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import Graph


@dataclass
class StyleRule:
    """A single selector -> properties mapping."""
    selector: str
    selector_type: str  # "universal", "class", "id"
    specificity: int  # 0, 1, 2
    properties: dict[str, str] = field(default_factory=dict)


_RULE_PATTERN = re.compile(
    r"([*#.][\w-]*)\s*\{([^}]*)\}",
    re.DOTALL,
)

_PROP_PATTERN = re.compile(
    r"([\w_-]+)\s*:\s*\"?([^;\"}]+)\"?\s*;?",
)


def parse_stylesheet(source: str) -> list[StyleRule]:
    """Parse a stylesheet string into a list of StyleRule objects.

    Rules are returned in source order; specificity is computed per-rule.
    """
    rules: list[StyleRule] = []
    source = source.strip()
    if not source:
        return rules

    for match in _RULE_PATTERN.finditer(source):
        selector = match.group(1).strip()
        body = match.group(2).strip()

        if selector == "*":
            sel_type = "universal"
            specificity = 0
        elif selector.startswith("."):
            sel_type = "class"
            specificity = 1
        elif selector.startswith("#"):
            sel_type = "id"
            specificity = 2
        else:
            continue  # unknown selector

        properties: dict[str, str] = {}
        for prop_match in _PROP_PATTERN.finditer(body):
            prop_name = prop_match.group(1).strip()
            prop_value = prop_match.group(2).strip()
            properties[prop_name] = prop_value

        rules.append(StyleRule(
            selector=selector,
            selector_type=sel_type,
            specificity=specificity,
            properties=properties,
        ))

    return rules


def _matches(rule: StyleRule, node_id: str, css_class: str) -> bool:
    """Check if a rule's selector matches a given node."""
    if rule.selector_type == "universal":
        return True
    if rule.selector_type == "class":
        target_class = rule.selector[1:]  # strip leading '.'
        # A node can have space-separated classes
        return target_class in css_class.split()
    if rule.selector_type == "id":
        target_id = rule.selector[1:]  # strip leading '#'
        return node_id == target_id
    return False


_APPLICABLE_PROPS = frozenset({"llm_model", "llm_provider", "reasoning_effort"})


def apply_stylesheet(graph: "Graph") -> "Graph":
    """Parse the graph's model_stylesheet and apply matching rules to nodes.

    Rules are applied in specificity order (lowest first), so higher-specificity
    rules override lower ones. Within the same specificity, later rules win.
    Modifies nodes in place and returns the graph.
    """
    source = graph.model_stylesheet
    if not source:
        return graph

    rules = parse_stylesheet(source)
    # Sort by specificity (stable sort preserves source order for ties)
    rules.sort(key=lambda r: r.specificity)

    for node in graph.nodes.values():
        css_class = node.css_class
        for rule in rules:
            if _matches(rule, node.id, css_class):
                for prop, value in rule.properties.items():
                    if prop in _APPLICABLE_PROPS:
                        setattr(node, prop, value)

    return graph
