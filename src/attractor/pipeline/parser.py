"""DOT-subset parser for Attractor pipeline definitions.

Parses the supported subset of DOT (digraph only) and returns a Graph model.
Implemented from scratch with no external DOT parsing libraries.

Export: parse_dot(source: str) -> Graph
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from .graph import Edge, Graph, Node


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TokenType(Enum):
    IDENTIFIER = auto()
    STRING = auto()
    NUMBER = auto()
    ARROW = auto()       # ->
    LBRACKET = auto()    # [
    RBRACKET = auto()    # ]
    LBRACE = auto()      # {
    RBRACE = auto()      # }
    EQUALS = auto()      # =
    COMMA = auto()       # ,
    SEMICOLON = auto()   # ;
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: str
    line: int = 0
    col: int = 0


# Regex patterns for tokens (order matters).
_TOKEN_PATTERNS: list[tuple[TokenType | None, re.Pattern[str]]] = [
    (None, re.compile(r"\s+")),                              # whitespace – skip
    (TokenType.ARROW, re.compile(r"->")),
    (TokenType.STRING, re.compile(
        r'"(?:[^"\\]|\\.)*"'                                 # double-quoted
    )),
    (TokenType.NUMBER, re.compile(r"-?(?:\d+\.\d*|\.\d+|\d+)")),
    (TokenType.LBRACKET, re.compile(r"\[")),
    (TokenType.RBRACKET, re.compile(r"\]")),
    (TokenType.LBRACE, re.compile(r"\{")),
    (TokenType.RBRACE, re.compile(r"\}")),
    (TokenType.EQUALS, re.compile(r"=")),
    (TokenType.COMMA, re.compile(r",")),
    (TokenType.SEMICOLON, re.compile(r";")),
    (TokenType.IDENTIFIER, re.compile(r"[A-Za-z_][A-Za-z0-9_]*")),
]


def _strip_comments(source: str) -> str:
    """Remove // line comments and /* ... */ block comments."""
    result: list[str] = []
    i = 0
    n = len(source)
    in_string = False
    while i < n:
        # Track quoted strings so we don't strip inside them.
        if source[i] == '"' and not in_string:
            in_string = True
            j = i + 1
            while j < n:
                if source[j] == '\\':
                    j += 2
                    continue
                if source[j] == '"':
                    j += 1
                    break
                j += 1
            result.append(source[i:j])
            i = j
            in_string = False
            continue
        # Line comment
        if source[i:i + 2] == '//':
            end = source.find('\n', i)
            if end == -1:
                break
            # Preserve the newline so line numbers stay accurate.
            result.append('\n')
            i = end + 1
            continue
        # Block comment
        if source[i:i + 2] == '/*':
            end = source.find('*/', i + 2)
            if end == -1:
                break  # unterminated block comment – drop rest
            # Replace with equivalent whitespace to preserve line numbers.
            block = source[i:end + 2]
            result.append('\n' * block.count('\n'))
            i = end + 2
            continue
        result.append(source[i])
        i += 1
    return ''.join(result)


def _tokenize(source: str) -> list[Token]:
    """Produce a list of tokens from *source* (comments already stripped)."""
    tokens: list[Token] = []
    pos = 0
    n = len(source)
    # Pre-compute line starts for line/col tracking.
    line = 1
    col = 1

    while pos < n:
        matched = False
        for tok_type, pattern in _TOKEN_PATTERNS:
            m = pattern.match(source, pos)
            if m:
                text = m.group(0)
                if tok_type is not None:
                    tokens.append(Token(type=tok_type, value=text, line=line, col=col))
                # Advance line/col counters.
                newlines = text.count('\n')
                if newlines:
                    line += newlines
                    col = len(text) - text.rfind('\n')
                else:
                    col += len(text)
                pos = m.end()
                matched = True
                break
        if not matched:
            # Skip unknown single character.
            if source[pos] == '\n':
                line += 1
                col = 1
            else:
                col += 1
            pos += 1

    tokens.append(Token(type=TokenType.EOF, value="", line=line, col=col))
    return tokens


# ---------------------------------------------------------------------------
# Helpers: attribute value coercion
# ---------------------------------------------------------------------------

_BOOL_TRUE = frozenset({"true", "yes", "1"})
_BOOL_FALSE = frozenset({"false", "no", "0"})


def _unquote(s: str) -> str:
    """Remove surrounding quotes and process escape sequences."""
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        inner = s[1:-1]
        # Process common escape sequences.
        inner = inner.replace('\\"', '"')
        inner = inner.replace('\\\\', '\\')
        inner = inner.replace('\\n', '\n')
        inner = inner.replace('\\t', '\t')
        return inner
    return s


def _coerce_value(raw: str) -> str:
    """Return a cleaned string value suitable for storage in attrs dicts.

    All values are stored as strings in *attrs*; typed fields on Node / Edge /
    Graph are set separately during model construction.
    """
    return _unquote(raw)


# ---------------------------------------------------------------------------
# Recursive-descent parser
# ---------------------------------------------------------------------------

class ParseError(Exception):
    """Raised when the DOT source cannot be parsed."""

    def __init__(self, message: str, token: Token | None = None):
        loc = ""
        if token:
            loc = f" at line {token.line}, col {token.col}"
        super().__init__(f"{message}{loc}")
        self.token = token


@dataclass
class _ParserState:
    tokens: list[Token]
    pos: int = 0

    # Defaults that can be set via `node [...]` / `edge [...]` blocks.
    node_defaults: dict[str, str] = field(default_factory=dict)
    edge_defaults: dict[str, str] = field(default_factory=dict)

    # Accumulated results.
    graph_attrs: dict[str, str] = field(default_factory=dict)
    nodes: dict[str, dict[str, str]] = field(default_factory=dict)
    edges: list[tuple[str, str, dict[str, str]]] = field(default_factory=list)
    graph_name: str = ""

    # Subgraph context: CSS class derived from subgraph label/name.
    subgraph_class: str = ""

    # ---- token helpers ----

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def at(self, tok_type: TokenType) -> bool:
        return self.tokens[self.pos].type == tok_type

    def at_value(self, value: str) -> bool:
        t = self.tokens[self.pos]
        return t.type == TokenType.IDENTIFIER and t.value == value

    def advance(self) -> Token:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def expect(self, tok_type: TokenType) -> Token:
        t = self.tokens[self.pos]
        if t.type != tok_type:
            raise ParseError(
                f"Expected {tok_type.name}, got {t.type.name} ({t.value!r})", t
            )
        self.pos += 1
        return t

    def expect_value(self, value: str) -> Token:
        t = self.tokens[self.pos]
        if not (t.type == TokenType.IDENTIFIER and t.value == value):
            raise ParseError(f"Expected '{value}', got {t.value!r}", t)
        self.pos += 1
        return t

    def maybe(self, tok_type: TokenType) -> Token | None:
        if self.tokens[self.pos].type == tok_type:
            return self.advance()
        return None

    # ---- ensure node registered ----

    def ensure_node(self, node_id: str) -> None:
        if node_id not in self.nodes:
            defaults = dict(self.node_defaults)
            # Inherit CSS class from enclosing subgraph if not already set.
            if self.subgraph_class and "class" not in defaults:
                defaults["class"] = self.subgraph_class
            self.nodes[node_id] = defaults


def _parse_attr_list(state: _ParserState) -> dict[str, str]:
    """Parse ``[ key=value, key=value, ... ]``."""
    attrs: dict[str, str] = {}
    state.expect(TokenType.LBRACKET)
    while not state.at(TokenType.RBRACKET) and not state.at(TokenType.EOF):
        key_tok = state.advance()
        if key_tok.type not in (TokenType.IDENTIFIER, TokenType.STRING):
            raise ParseError(f"Expected attribute key, got {key_tok.value!r}", key_tok)
        key = _unquote(key_tok.value)
        state.expect(TokenType.EQUALS)
        val_tok = state.advance()
        if val_tok.type not in (
            TokenType.IDENTIFIER,
            TokenType.STRING,
            TokenType.NUMBER,
        ):
            raise ParseError(
                f"Expected attribute value, got {val_tok.value!r}", val_tok
            )
        attrs[key] = _coerce_value(val_tok.value)
        # Optional comma / semicolon separator.
        if state.at(TokenType.COMMA):
            state.advance()
        elif state.at(TokenType.SEMICOLON):
            state.advance()
    state.expect(TokenType.RBRACKET)
    return attrs


def _parse_statement(state: _ParserState) -> None:
    """Parse a single statement inside the graph body."""
    tok = state.peek()

    # --- graph attribute block: graph [ ... ] ---
    if tok.type == TokenType.IDENTIFIER and tok.value == "graph":
        state.advance()
        if state.at(TokenType.LBRACKET):
            attrs = _parse_attr_list(state)
            state.graph_attrs.update(attrs)
            state.maybe(TokenType.SEMICOLON)
            return
        # Otherwise fall through – could be graph-level id (unusual, ignore).
        state.maybe(TokenType.SEMICOLON)
        return

    # --- node defaults: node [ ... ] ---
    if tok.type == TokenType.IDENTIFIER and tok.value == "node":
        state.advance()
        if state.at(TokenType.LBRACKET):
            attrs = _parse_attr_list(state)
            state.node_defaults.update(attrs)
        state.maybe(TokenType.SEMICOLON)
        return

    # --- edge defaults: edge [ ... ] ---
    if tok.type == TokenType.IDENTIFIER and tok.value == "edge":
        state.advance()
        if state.at(TokenType.LBRACKET):
            attrs = _parse_attr_list(state)
            state.edge_defaults.update(attrs)
        state.maybe(TokenType.SEMICOLON)
        return

    # --- subgraph ---
    if tok.type == TokenType.IDENTIFIER and tok.value == "subgraph":
        _parse_subgraph(state)
        state.maybe(TokenType.SEMICOLON)
        return

    # --- top-level key = value ---
    # Check if this is ``IDENTIFIER = value`` (graph attribute shorthand).
    if tok.type == TokenType.IDENTIFIER:
        # Look ahead for '='.
        next_tok = state.tokens[state.pos + 1] if state.pos + 1 < len(state.tokens) else None
        if next_tok and next_tok.type == TokenType.EQUALS:
            # But first make sure it's not ``A -> ...`` (edge) or ``A [...]`` (node).
            # We need a 2-token lookahead: IDENT EQUALS ...
            # Only treat as graph attr if the token after '=' is a value (not '>' etc.).
            third = state.tokens[state.pos + 2] if state.pos + 2 < len(state.tokens) else None
            if third and third.type in (
                TokenType.IDENTIFIER,
                TokenType.STRING,
                TokenType.NUMBER,
            ):
                # But we need to also check that the *fourth* token is NOT '->',
                # because ``A = B -> C`` doesn't make sense for attrs but just in case.
                fourth = state.tokens[state.pos + 3] if state.pos + 3 < len(state.tokens) else None
                if not (fourth and fourth.type == TokenType.ARROW):
                    key_tok = state.advance()
                    state.advance()  # consume '='
                    val_tok = state.advance()
                    state.graph_attrs[key_tok.value] = _coerce_value(val_tok.value)
                    state.maybe(TokenType.SEMICOLON)
                    return

    # --- node or edge statement: starts with IDENTIFIER (or STRING for node id) ---
    if tok.type in (TokenType.IDENTIFIER, TokenType.STRING):
        _parse_node_or_edge(state)
        state.maybe(TokenType.SEMICOLON)
        return

    # Unknown – skip.
    state.advance()


def _parse_node_or_edge(state: _ParserState) -> None:
    """Parse a node declaration or an edge chain (``A -> B -> C [attrs]``)."""
    # Collect the chain of node identifiers connected by '->'.
    node_ids: list[str] = []

    first = state.advance()
    node_ids.append(_unquote(first.value))

    while state.at(TokenType.ARROW):
        state.advance()  # consume '->'
        nxt = state.advance()
        if nxt.type not in (TokenType.IDENTIFIER, TokenType.STRING):
            raise ParseError(f"Expected node id after '->', got {nxt.value!r}", nxt)
        node_ids.append(_unquote(nxt.value))

    # Optional attribute list.
    attrs: dict[str, str] = {}
    if state.at(TokenType.LBRACKET):
        attrs = _parse_attr_list(state)

    if len(node_ids) == 1:
        # Node declaration.
        nid = node_ids[0]
        state.ensure_node(nid)
        state.nodes[nid].update(attrs)
    else:
        # Edge chain – create an edge for each consecutive pair.
        merged_attrs = {**state.edge_defaults, **attrs}
        for i in range(len(node_ids) - 1):
            src = node_ids[i]
            dst = node_ids[i + 1]
            state.ensure_node(src)
            state.ensure_node(dst)
            state.edges.append((src, dst, dict(merged_attrs)))


def _derive_subgraph_class(name: str) -> str:
    """Derive a CSS class from a subgraph name.

    ``cluster_planning`` -> ``planning``, ``cluster_code_review`` -> ``code_review``.
    Plain names (without ``cluster_`` prefix) are used as-is.
    """
    if name.startswith("cluster_"):
        return name[8:]
    return name


def _parse_subgraph(state: _ParserState) -> None:
    """Parse ``subgraph name { ... }`` – flattened into the main graph.

    If the subgraph has a name (e.g. ``cluster_planning``), nodes inside
    it inherit a derived CSS class (``planning``) unless they already have
    an explicit ``class`` attribute.
    """
    state.expect_value("subgraph")
    # Optional subgraph name.
    subgraph_name = ""
    if state.at(TokenType.IDENTIFIER) or state.at(TokenType.STRING):
        subgraph_name = state.advance().value
    state.expect(TokenType.LBRACE)

    # Save and scope node defaults and subgraph class.
    saved_node_defaults = dict(state.node_defaults)
    saved_edge_defaults = dict(state.edge_defaults)
    saved_subgraph_class = state.subgraph_class

    if subgraph_name:
        state.subgraph_class = _derive_subgraph_class(subgraph_name)

    while not state.at(TokenType.RBRACE) and not state.at(TokenType.EOF):
        _parse_statement(state)

    state.expect(TokenType.RBRACE)

    # Restore defaults.
    state.node_defaults = saved_node_defaults
    state.edge_defaults = saved_edge_defaults
    state.subgraph_class = saved_subgraph_class


def _parse_body(state: _ParserState) -> None:
    """Parse all statements inside the top-level ``{ ... }``."""
    state.expect(TokenType.LBRACE)
    while not state.at(TokenType.RBRACE) and not state.at(TokenType.EOF):
        _parse_statement(state)
    state.expect(TokenType.RBRACE)


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------

def _to_bool(value: str) -> bool:
    return value.lower() in _BOOL_TRUE


def _to_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _build_graph(state: _ParserState) -> Graph:
    """Convert accumulated parser state into a Graph model."""
    ga = state.graph_attrs

    graph = Graph(
        name=state.graph_name,
        goal=ga.get("goal", ""),
        label=ga.get("label", ""),
        model_stylesheet=ga.get("model_stylesheet", ""),
        default_max_retry=_to_int(ga.get("default_max_retry", ""), 50),
        retry_target=ga.get("retry_target", ""),
        fallback_retry_target=ga.get("fallback_retry_target", ""),
        default_fidelity=ga.get("default_fidelity", ""),
        attrs=dict(ga),
    )

    # Build nodes.
    for nid, raw in state.nodes.items():
        node = Node(
            id=nid,
            label=raw.get("label", ""),
            shape=raw.get("shape", "box"),
            type=raw.get("type", ""),
            prompt=raw.get("prompt", ""),
            max_retries=_to_int(raw.get("max_retries", ""), 0),
            goal_gate=_to_bool(raw.get("goal_gate", "")),
            retry_target=raw.get("retry_target", ""),
            fallback_retry_target=raw.get("fallback_retry_target", ""),
            fidelity=raw.get("fidelity", ""),
            thread_id=raw.get("thread_id", ""),
            timeout=raw.get("timeout", ""),
            llm_model=raw.get("llm_model", ""),
            llm_provider=raw.get("llm_provider", ""),
            reasoning_effort=raw.get("reasoning_effort", "high"),
            auto_status=_to_bool(raw.get("auto_status", "")),
            allow_partial=_to_bool(raw.get("allow_partial", "")),
            attrs=dict(raw),
            **{"class": raw.get("class", "")},
        )
        graph.nodes[nid] = node

    # Build edges.
    for src, dst, raw in state.edges:
        edge = Edge(
            from_node=src,
            to_node=dst,
            label=raw.get("label", ""),
            condition=raw.get("condition", ""),
            weight=_to_int(raw.get("weight", ""), 0),
            fidelity=raw.get("fidelity", ""),
            thread_id=raw.get("thread_id", ""),
            loop_restart=_to_bool(raw.get("loop_restart", "")),
            attrs=dict(raw),
        )
        graph.edges.append(edge)

    return graph


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_dot(source: str) -> Graph:
    """Parse a DOT digraph string and return a ``Graph`` instance.

    Only ``digraph`` is supported; ``graph`` (undirected) will raise
    ``ParseError``.

    Parameters
    ----------
    source:
        The DOT source text.

    Returns
    -------
    Graph
        A fully populated graph model.

    Raises
    ------
    ParseError
        If the source is not valid or is not a digraph.
    """
    cleaned = _strip_comments(source)
    tokens = _tokenize(cleaned)
    state = _ParserState(tokens=tokens)

    # Expect: ``digraph`` or ``strict digraph``
    if state.at_value("strict"):
        state.advance()

    if not state.at_value("digraph"):
        raise ParseError(
            "Only digraph is supported. Expected 'digraph'.", state.peek()
        )
    state.advance()  # consume 'digraph'

    # Optional graph name.
    if state.at(TokenType.IDENTIFIER) or state.at(TokenType.STRING):
        state.graph_name = _unquote(state.advance().value)

    _parse_body(state)

    return _build_graph(state)
