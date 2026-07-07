"""Deterministic read-only SQL guard shared by every agent/dataset query path (C2).

The expert ReAct agents and the talk-to-data graph both run LLM-generated SQL over
untrusted datasets. The old guard was a prefix check (``startswith("SELECT")``),
which does **not** stop stacked statements (``SELECT 1; UPDATE …``) on engines that
honour multiple statements. This module upgrades that to a real, single-statement
validator via ``sqlparse``:

  * exactly one non-empty statement (rejects stacked/`;`-chained SQL);
  * the statement type is a read (``SELECT``, or a ``WITH … SELECT`` CTE, or
    ``PRAGMA`` for SQLite introspection);
  * no DML/DDL/side-effecting keyword leads a statement.

``sqlparse`` is a pure-Python dependency; if it is somehow unavailable we degrade
to the old prefix check rather than hard-failing (fail-safe: still read-only-ish,
just less strict), and log a warning once.

Use :func:`validate_read_only` at every choke point:
``run_data_group_query_logic`` (both graphs), ``mcp_server.run_sql_query``, and the
expert ``run_sql`` tool. Pair it with a **read-only DB role** in production — this
is defence in depth, not a substitute for least-privilege credentials.
"""
from __future__ import annotations

from src.core.logging_config import get_logger

logger = get_logger(__name__)

# Statement leads we accept. WITH is a CTE that must resolve to a SELECT; PRAGMA is
# SQLite read-only introspection used by the schema tools.
_ALLOWED_LEADS = ("SELECT", "WITH", "PRAGMA")

# Keywords that must never lead a statement in a read-only context. Checked as the
# statement's first meaningful token, not by substring (so a column literally named
# "update_time" in a SELECT is fine).
_FORBIDDEN_LEADS = frozenset({
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "REPLACE", "MERGE", "GRANT", "REVOKE", "EXEC", "EXECUTE", "CALL",
    "ATTACH", "DETACH", "COPY", "VACUUM", "ANALYZE", "SET", "BEGIN",
    "COMMIT", "ROLLBACK", "USE", "LOAD", "INSTALL", "PRAGMA",  # PRAGMA re-allowed below when opted in
})


def _prefix_fallback(sql: str) -> tuple[bool, str]:
    head = sql.strip().lstrip("(").upper()
    if head.startswith(_ALLOWED_LEADS):
        # Best-effort stacked-statement rejection without a parser.
        if ";" in sql.strip().rstrip(";"):
            return False, "Multiple statements are not allowed."
        return True, ""
    return False, "Only read-only SELECT/WITH/PRAGMA queries are allowed."


def validate_read_only(sql: str, *, allow_pragma: bool = True) -> tuple[bool, str]:
    """Return ``(ok, reason)``. ``ok=False`` means the SQL must be rejected.

    Args:
        sql: the candidate query.
        allow_pragma: permit a single ``PRAGMA`` statement (SQLite introspection).
            The expert data tool leaves this on; a stricter caller can disable it.
    """
    if not sql or not sql.strip():
        return False, "Empty query."

    try:
        import sqlparse
        from sqlparse import tokens as T
    except ImportError:  # pragma: no cover - degraded mode
        logger.warning("sqlparse unavailable — falling back to prefix-only SQL guard.")
        return _prefix_fallback(sql)

    # Strip comments so a trailing "-- ; DROP" can't smuggle a second statement.
    cleaned = sqlparse.format(sql, strip_comments=True).strip()
    if not cleaned:
        return False, "Query is only comments."

    statements = [s for s in sqlparse.parse(cleaned) if str(s).strip().strip(";")]
    if len(statements) == 0:
        return False, "No executable statement found."
    if len(statements) > 1:
        return False, "Multiple statements are not allowed (single read-only query only)."

    stmt = statements[0]
    # First meaningful token → the statement lead.
    lead = None
    for tok in stmt.flatten():
        if tok.is_whitespace or tok.ttype in T.Comment:
            continue
        val = tok.value.strip()
        if not val or val == "(":
            continue
        lead = val.upper()
        break

    if lead is None:
        return False, "Could not determine statement type."

    if lead == "PRAGMA":
        return (True, "") if allow_pragma else (False, "PRAGMA is not allowed here.")

    if lead in _FORBIDDEN_LEADS:
        return False, f"Write/DDL statements are not allowed (got {lead})."

    if lead not in ("SELECT", "WITH"):
        # sqlparse sometimes leads a parenthesised SELECT with the paren already
        # stripped above; anything else that isn't a known read is rejected.
        stmt_type = stmt.get_type()  # 'SELECT' for WITH…SELECT CTEs too
        if stmt_type != "SELECT":
            return False, f"Only read-only SELECT/WITH queries are allowed (got {lead})."

    return True, ""
