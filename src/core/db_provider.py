"""
Database Provider Module
------------------------
Provides a unified database connection layer that supports multiple providers:
  - SQLite (default)
  - PostgreSQL
  - MySQL
  - SQL Server (MSSQL)

Connection details are read from environment variables via config.py:
  DB_PROVIDER           = sqlite | postgresql | mysql | mssql
  DB_CONNECTION_STRING  = SQLAlchemy-compatible connection string

Usage:
    from src.core.db_provider import get_engine, get_connection, db_manager

    # Get a SQLAlchemy engine for default database
    engine = db_manager.get_provider("default").engine

    # Use as a context manager
    with db_manager.get_provider("default").connect() as conn:
        result = conn.execute(text("SELECT 1"))
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text, Engine, Connection

from src.core.config import settings


# ---------------------------------------------------------------------------
# Supported providers
# ---------------------------------------------------------------------------
SUPPORTED_PROVIDERS = {"sqlite", "postgresql", "mysql", "mssql"}


# ---------------------------------------------------------------------------
# Provider-specific SQL helpers
# ---------------------------------------------------------------------------
class DatabaseProvider:
    """Encapsulates provider-specific SQL dialect differences."""

    def __init__(self, provider: str, connection_string: str):
        provider = provider.strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported DB_PROVIDER: '{provider}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_PROVIDERS))}"
            )
        self.provider = provider
        self.connection_string = connection_string
        self._engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            kwargs: dict = {}
            if self.provider == "sqlite":
                # check_same_thread=False so connections can be used across the
                # threadpool that runs blocking SQL. Default QueuePool gives
                # multiple connections so one wedged/cancelled query cannot
                # block all others (StaticPool's single connection could).
                kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
            self._engine = create_engine(self.connection_string, **kwargs)
        return self._engine

    @contextmanager
    def connect(self) -> Generator[Connection, None, None]:
        """Context manager that yields a SQLAlchemy connection."""
        with self.engine.connect() as conn:
            yield conn

    # -- Schema introspection queries ----------------------------------------

    def get_tables_query(self) -> str:
        """Returns SQL to list all user tables with their DDL or metadata."""
        if self.provider == "sqlite":
            return "SELECT name, sql FROM sqlite_master WHERE type='table'"
        elif self.provider == "postgresql":
            return (
                "SELECT table_name, "
                "'CREATE TABLE ' || table_name AS sql "
                "FROM information_schema.tables "
                "WHERE table_schema = current_schema() AND table_type = 'BASE TABLE'"
            )
        elif self.provider == "mysql":
            return (
                "SELECT table_name, "
                "CONCAT('CREATE TABLE ', table_name) AS sql "
                "FROM information_schema.tables "
                "WHERE table_schema = DATABASE() AND table_type = 'BASE TABLE'"
            )
        elif self.provider == "mssql":
            return (
                "SELECT table_name, "
                "'CREATE TABLE ' + table_name AS sql "
                "FROM information_schema.tables "
                "WHERE table_type = 'BASE TABLE'"
            )
        return ""

    def get_table_names_query(self) -> str:
        """Returns SQL to list just table names."""
        if self.provider == "sqlite":
            return "SELECT name FROM sqlite_master WHERE type='table'"
        elif self.provider == "postgresql":
            return (
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = current_schema() AND table_type = 'BASE TABLE'"
            )
        elif self.provider == "mysql":
            return (
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = DATABASE() AND table_type = 'BASE TABLE'"
            )
        elif self.provider == "mssql":
            return (
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_type = 'BASE TABLE'"
            )
        return ""

    def get_column_info_query(self, table_name: str) -> str:
        """Returns SQL to get column metadata for a given table."""
        if self.provider == "sqlite":
            return f"PRAGMA table_info('{table_name}')"
        elif self.provider == "postgresql":
            return (
                f"SELECT column_name, data_type, is_nullable, column_default "
                f"FROM information_schema.columns "
                f"WHERE table_schema = current_schema() AND table_name = '{table_name}' "
                f"ORDER BY ordinal_position"
            )
        else:
            # Works for mysql, mssql (information_schema)
            return (
                f"SELECT column_name, data_type, is_nullable, column_default "
                f"FROM information_schema.columns "
                f"WHERE table_name = '{table_name}' "
                f"ORDER BY ordinal_position"
            )

    def get_schema_ddl_query(self, table_name: str) -> str:
        """Returns SQL to get table DDL/create statement."""
        if self.provider == "sqlite":
            return f"SELECT sql FROM sqlite_master WHERE name='{table_name}'"
        elif self.provider == "postgresql":
            # PostgreSQL doesn't have a simple DDL query; 
            # we build from information_schema
            return (
                f"SELECT column_name, data_type, is_nullable "
                f"FROM information_schema.columns "
                f"WHERE table_schema = current_schema() AND table_name = '{table_name}' "
                f"ORDER BY ordinal_position"
            )
        elif self.provider == "mysql":
            return f"SHOW CREATE TABLE `{table_name}`"
        elif self.provider == "mssql":
            return (
                f"SELECT column_name, data_type, is_nullable "
                f"FROM information_schema.columns "
                f"WHERE table_name = '{table_name}' "
                f"ORDER BY ordinal_position"
            )
        return ""

    def build_full_schema_text(self) -> str:
        """
        Builds a human-readable schema text for all tables.
        Used by MCP server's get_schema tool.
        """
        schema_info = "Database Schema:\n"
        with self.connect() as conn:
            if self.provider == "sqlite":
                result = conn.execute(text(self.get_tables_query()))
                for row in result:
                    table_name, table_sql = row[0], row[1]
                    schema_info += f"- Table '{table_name}': {table_sql}\n"
            else:
                # Get table list
                tables_result = conn.execute(text(self.get_table_names_query()))
                table_names = [row[0] for row in tables_result]

                for table_name in table_names:
                    col_result = conn.execute(
                        text(self.get_column_info_query(table_name))
                    )
                    columns = col_result.fetchall()

                    if self.provider == "mysql":
                        # MySQL PRAGMA-like: use information_schema results
                        col_defs = ", ".join(
                            f"{c[0]} {c[1]}" + (" NOT NULL" if c[2] == "NO" else "")
                            for c in columns
                        )
                    else:
                        col_defs = ", ".join(
                            f"{c[0]} {c[1]}" + (" NOT NULL" if c[2] == "NO" else "")
                            for c in columns
                        )

                    ddl = f"CREATE TABLE {table_name} ({col_defs})"
                    schema_info += f"- Table '{table_name}': {ddl}\n"

        return schema_info

    def get_column_stats_numeric_query(
        self, table_name: str, col_name: str
    ) -> str:
        """Returns SQL for numeric column stats (min, max, avg, distinct)."""
        return (
            f'SELECT '
            f'MIN("{col_name}"), '
            f'MAX("{col_name}"), '
            f'ROUND(AVG("{col_name}"), 4), '  # ROUND syntax is standard SQL
            f'COUNT(DISTINCT "{col_name}") '
            f'FROM "{table_name}"'
        )

    def get_column_stats_text_top_values_query(
        self, table_name: str, col_name: str, limit: int = 10
    ) -> str:
        """Returns SQL for top N most frequent text values."""
        if self.provider == "mssql":
            return (
                f'SELECT TOP {limit} "{col_name}", COUNT(*) as cnt '
                f'FROM "{table_name}" '
                f'WHERE "{col_name}" IS NOT NULL '
                f'GROUP BY "{col_name}" '
                f'ORDER BY cnt DESC'
            )
        return (
            f'SELECT "{col_name}", COUNT(*) as cnt '
            f'FROM "{table_name}" '
            f'WHERE "{col_name}" IS NOT NULL '
            f'GROUP BY "{col_name}" '
            f'ORDER BY cnt DESC '
            f'LIMIT {limit}'
        )

    def get_count_query(self, table_name: str) -> str:
        """Returns SQL for row count."""
        return f'SELECT COUNT(*) FROM "{table_name}"'

    def get_null_count_query(self, table_name: str, col_name: str) -> str:
        """Returns SQL for null count of a column."""
        return f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col_name}" IS NULL'

    def get_distinct_count_query(self, table_name: str, col_name: str) -> str:
        """Returns SQL for distinct count of a column."""
        return f'SELECT COUNT(DISTINCT "{col_name}") FROM "{table_name}"'

    def get_limit_query(self, table_name: str, limit: int) -> str:
        """Returns SQL to select limited rows from a table."""
        if self.provider == "mssql":
            return f'SELECT TOP {limit} * FROM "{table_name}"'
        return f'SELECT * FROM "{table_name}" LIMIT {limit}'

    def is_numeric_type(self, col_type: str) -> bool:
        """Check if a column type string represents a numeric type."""
        upper = (col_type or "").upper()
        numeric_keywords = (
            "INT", "REAL", "FLOAT", "NUMERIC", "DOUBLE", "DECIMAL",
            "BIGINT", "SMALLINT", "TINYINT", "MONEY", "NUMBER",
            "SERIAL",
        )
        return any(t in upper for t in numeric_keywords)

    @property
    def provider_display_name(self) -> str:
        """Human-readable provider name for prompts and UI."""
        names = {
            "sqlite": "SQLite",
            "postgresql": "PostgreSQL",
            "mysql": "MySQL",
            "mssql": "SQL Server",
        }
        return names.get(self.provider, self.provider)


# ---------------------------------------------------------------------------
# Module-level singletons — main data DB manager
# ---------------------------------------------------------------------------
class DatabaseManager:
    def __init__(self, databases_config: dict):
        self.providers: dict[str, DatabaseProvider] = {}
        for db_name, config in databases_config.items():
            self.providers[db_name] = DatabaseProvider(
                provider=config.get("provider", "sqlite"),
                connection_string=config.get("connection_string", "")
            )

    def get_provider(self, db_name: str = "default") -> DatabaseProvider:
        if db_name not in self.providers:
            raise ValueError(f"Database '{db_name}' not found in configuration.")
        return self.providers[db_name]

    def get_all_providers(self) -> dict[str, DatabaseProvider]:
        return self.providers

db_manager = DatabaseManager(settings.DATABASES)

# Backward compatibility aliases
db_provider = db_manager.get_provider("default")

# ---------------------------------------------------------------------------
# Module-level singletons — history/experiment tracker DB
# ---------------------------------------------------------------------------
history_db_provider = DatabaseProvider(
    provider=settings.HISTORY_DB_PROVIDER,
    connection_string=settings.HISTORY_DB_CONNECTION_STRING,
)

# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------
def get_engine() -> Engine:
    """Get the default database SQLAlchemy engine."""
    return db_manager.get_provider("default").engine


def get_connection():
    """Context manager for default database connection."""
    return db_manager.get_provider("default").connect()


def get_history_engine() -> Engine:
    """Get the history database SQLAlchemy engine."""
    return history_db_provider.engine


def get_history_connection():
    """Context manager for history database connection."""
    return history_db_provider.connect()
