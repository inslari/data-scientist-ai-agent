from __future__ import annotations

import ast
import asyncio
import re
from typing import Any

import numpy as np
import pandas as pd
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.state import AgentState
from src.core.config import settings

MCP_CONFIG = {
    "sqlite": {
        "url": settings.MCP_SERVER_URL,
        "transport": "streamable_http",
    }
}

MAX_ROWS = 10_000


def _resolve_all_related_tables(
    selected_tables: list[str],
    relationships: list[dict],
    all_known_tables: list[str],
) -> list[str]:
    """
    Planner'ın seçtiği tablolardan başlayarak relationship graph'ını
    BFS (Breadth-First Search) ile traverse eder.

    Örnek:
        selected: ['orders']
        relationships: orders→customers, orders→order_items, order_items→products
        sonuç: ['orders', 'customers', 'order_items', 'products']
    """
    graph: dict[str, list[str]] = {t: [] for t in all_known_tables}

    for rel in relationships:
        src = rel.get("source_table")
        tgt = rel.get("target_table")
        if src in graph and tgt in graph:
            graph[src].append(tgt)
            graph[tgt].append(src)

    visited: set[str] = set(selected_tables)
    queue: list[str] = list(selected_tables)

    while queue:
        current = queue.pop(0)
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return list(visited)


def _build_merge_order(
    tables: list[str],
    relationships: list[dict],
    anchor_table: str,
) -> list[dict]:
    """
    Determines the merge order starting from the anchor table.
    Returns a list of merge operations specifying which table merges with which and on which columns.
    """
    ops: list[dict] = []
    merged: set[str] = {anchor_table}
    remaining = set(tables) - merged
    max_iter = len(tables) * 2

    for _ in range(max_iter):
        if not remaining:
            break
        for rel in relationships:
            src = rel.get("source_table")
            tgt = rel.get("target_table")
            src_col = rel.get("source_column")
            tgt_col = rel.get("target_column")

            if src in merged and tgt in remaining:
                ops.append({"left": src, "right": tgt,
                             "left_on": src_col, "right_on": tgt_col})
                merged.add(tgt)
                remaining.discard(tgt)

            elif tgt in merged and src in remaining:
                ops.append({"left": tgt, "right": src,
                             "left_on": tgt_col, "right_on": src_col})
                merged.add(src)
                remaining.discard(src)

    return ops



#  Parse MCP Response to DF
def _parse_mcp_response_to_dataframe(raw: str, table_name: str = "") -> pd.DataFrame:
    """
    Parses the string format SQL output from MCP and converts it into a DataFrame.

    Format:
        Columns: col1, col2, col3
        Returned N rows.
        Results:
        (val1, val2, val3)
        ...
    """
    if not raw:
        raise ValueError(f"[{table_name}] MCP boş yanıt döndürdü.")
    if "Error" in raw or "SQL Error" in raw:
        raise ValueError(f"[{table_name}] MCP SQL hatası: {raw}")

    lines = raw.strip().splitlines()

    columns: list[str] = []
    for line in lines:
        if line.startswith("Columns:"):
            columns = [c.strip() for c in line.replace("Columns:", "").split(",")]
            break

    if not columns:
        raise ValueError(f"[{table_name}] Kolon bilgisi bulunamadı.")

    rows: list[tuple] = []
    for line in lines:
        line = line.strip()
        if line.startswith("(") and line.endswith(")"):
            try:
                parsed = _parse_tuple_line(line)
                rows.append(_pad_or_trim(parsed, len(columns)))
            except Exception:
                continue

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows, columns=columns)
    df = _infer_and_cast_dtypes(df)

    return df.head(MAX_ROWS)


def _pad_or_trim(row: tuple, expected: int) -> tuple:
    lst = list(row)
    if len(lst) < expected:
        lst.extend([None] * (expected - len(lst)))
    return tuple(lst[:expected])


def _parse_tuple_line(line: str) -> tuple:
    try:
        return ast.literal_eval(line)
    except (ValueError, SyntaxError):
        inner = line[1:-1]
        parts = re.split(r",\s*(?=(?:[^'\"]*['\"][^'\"]*['\"])*[^'\"]*$)", inner)
        parsed = []
        for part in parts:
            part = part.strip()
            if part in ("None", "NULL", "null", ""):
                parsed.append(None)
            elif part.startswith(("'", '"')) and part.endswith(("'", '"')):
                parsed.append(part[1:-1])
            else:
                try:
                    parsed.append(int(part))
                except ValueError:
                    try:
                        parsed.append(float(part))
                    except ValueError:
                        parsed.append(part)
        return tuple(parsed)


def _infer_and_cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0 and converted.notna().mean() > 0.8:
            df[col] = converted
            continue
        if df[col].dtype == object:
            try:
                sample = df[col].dropna().head(10).astype(str)
                if sample.str.match(r"\d{4}-\d{2}-\d{2}").mean() > 0.7:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
    return df


# MCP async caller — tables are fetched in parallel
async def _fetch_all_tables(table_names: list[str]) -> dict[str, str]:
    client = MultiServerMCPClient(MCP_CONFIG)
    tools = await client.get_tools()
    tool_map = {t.name: t for t in tools}

    if "run_sql_query" not in tool_map:
        raise RuntimeError("MCP'de 'run_sql_query' tool bulunamadı.")

    async def fetch_one(table: str) -> tuple[str, str]:
        query = f"SELECT * FROM {table} LIMIT {MAX_ROWS}"
        result = await tool_map["run_sql_query"].ainvoke({"query": query})
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict) and item.get("type") == "text":
                    return table, item["text"]
            return table, str(result)
        return table, result if isinstance(result, str) else str(result)

    results = await asyncio.gather(*[fetch_one(t) for t in table_names],
                                   return_exceptions=True)

    output: dict[str, str] = {}
    for res in results:
        if isinstance(res, Exception):
            raise res
        table, raw = res
        output[table] = raw

    return output



# Merger — pandas merge, no SQL JOIN
def _merge_all_dataframes(
    dfs: dict[str, pd.DataFrame],
    merge_ops: list[dict],
    anchor_table: str,
) -> pd.DataFrame:
    if len(dfs) == 1:
        return list(dfs.values())[0]

    result = dfs[anchor_table].copy()

    for op in merge_ops:
        right_table = op["right"]
        if right_table not in dfs:
            print(f"[merge] UYARI: '{right_table}' bulunamadı, atlanıyor.")
            continue

        right_df = dfs[right_table].copy()

        # Resolve overlapping columns
        overlap = (set(result.columns) & set(right_df.columns)) - {op["left_on"], op["right_on"]}
        if overlap:
            right_df = right_df.rename(columns={c: f"{c}_{right_table}" for c in overlap})

        before = len(result)
        result = result.merge(right_df, left_on=op["left_on"],
                              right_on=op["right_on"], how="left")
        print(f"[merge] {op['left']}.{op['left_on']} → "
              f"{right_table}.{op['right_on']} | {before} → {len(result)} satır")

    return result


# Profiler
def _profile_dataframe(df: pd.DataFrame) -> dict:
    """
    Generates statistical profile from actual data.
    Complements intelligence_engine's schema-level analysis.
    feature_engineering_node makes transformation decisions based on this report.
    """
    if df.empty:
        return {"error": "Boş DataFrame, profiling yapılamadı."}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    profile: dict[str, Any] = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "summary": {
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "total_missing": int(df.isnull().sum().sum()),
            "missing_rate": round(df.isnull().mean().mean(), 4),
            "duplicate_rows": int(df.duplicated().sum()),
        },
        "columns": {},
    }

    for col in df.columns:
        col_profile: dict[str, Any] = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "null_rate": round(df[col].isnull().mean(), 4),
            "unique_count": int(df[col].nunique()),
            "unique_rate": round(df[col].nunique() / max(len(df), 1), 4),
        }

        if col in numeric_cols:
            desc = df[col].describe()
            col_profile.update({
                "min": _safe_float(desc.get("min")),
                "max": _safe_float(desc.get("max")),
                "mean": _safe_float(desc.get("mean")),
                "median": _safe_float(df[col].median()),
                "std": _safe_float(desc.get("std")),
                "q25": _safe_float(desc.get("25%")),
                "q75": _safe_float(desc.get("75%")),
                "skewness": _safe_float(df[col].skew()),
                "kurtosis": _safe_float(df[col].kurt()),
                # has_outliers now stores the outlier *rate* (0.0–1.0)
                "has_outliers": _detect_outliers(df[col]),
            })

        elif col in categorical_cols:
            value_counts = df[col].value_counts()
            col_profile.update({
                "top_values": value_counts.head(5).to_dict(),
                "cardinality": (
                    "high" if col_profile["unique_count"] > 50
                    else "medium" if col_profile["unique_count"] > 10
                    else "low"
                ),
            })

        elif col in datetime_cols:
            col_profile.update({
                "min_date": str(df[col].min()),
                "max_date": str(df[col].max()),
                "date_range_days": (
                    (df[col].max() - df[col].min()).days
                    if df[col].notna().any() else None
                ),
            })

        profile["columns"][col] = col_profile

    profile["recommendations"] = _generate_recommendations(profile)
    return profile


def _safe_float(val: Any) -> float | None:
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _detect_outliers(series: pd.Series) -> float:
    """Returns the fraction of outlier rows (IQR method). 0.0 means no outliers."""
    try:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return 0.0
        mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
        return round(float(mask.mean()), 4)
    except Exception:
        return 0.0


def _generate_recommendations(profile: dict) -> list[str]:
    """
    Generates feature engineering recommendations based on profile results.
    feature_engineering_node uses this list to make transformation decisions.

    Rules:
    - missing_rate threshold: >5% (avoids noise from negligible nulls)
    - outlier threshold: >1% of rows (avoids flagging every numeric column)
    - skewness is reported independently even when outliers are present
    """
    recs = []
    summary = profile.get("summary", {})
    numeric_cols = set(summary.get("numeric_columns", []))

    # General quality checks 
    missing_rate = summary.get("missing_rate", 0)
    if missing_rate > 0.05:  # raised from 0.01 → meaningful threshold
        recs.append(
            f"Eksik değer imputation gerekli (missing_rate: %{missing_rate * 100:.1f})"
        )

    if summary.get("duplicate_rows", 0) > 0:
        recs.append(f"Duplicate satır temizlenmeli ({summary['duplicate_rows']} adet)")

    # Date column detection 
    DATE_SUFFIXES = ("_date", "_at", "_time")
    DATE_EXACT = {"created_at", "updated_at", "timestamp", "birth_date", "order_date"}

    OUTLIER_RATE_THRESHOLD = 0.01  # >1% of rows must be outliers to flag

    # Column-level recommendations 
    for col, info in profile.get("columns", {}).items():
        dtype = info.get("dtype", "")
        col_lower = col.lower()
        is_id = col_lower.endswith("_id") or col_lower == "id"
        is_date = (
            "datetime" in dtype
            or col_lower in DATE_EXACT
            or any(col_lower.endswith(s) for s in DATE_SUFFIXES)
        )
        is_monetary = any(k in col_lower for k in ["price", "amount", "cost", "revenue"])

        # outlier_rate is now a float (0.0–1.0), not a bool
        outlier_rate = info.get("has_outliers", 0.0)
        if isinstance(outlier_rate, bool):  # backward-compat guard
            outlier_rate = 0.05 if outlier_rate else 0.0
        has_significant_outliers = outlier_rate > OUTLIER_RATE_THRESHOLD

        skew = info.get("skewness")
        cardinality = info.get("cardinality", "")
        null_rate = info.get("null_rate", 0)
        unique_count = info.get("unique_count", 0)
        unique_rate = info.get("unique_rate", 0)

        # ID columns → drop
        if is_id:
            recs.append(f"'{col}' ID sütun — model eğitiminden çıkarılmalı")
            continue

        # Constant columns → drop
        if unique_count == 1:
            recs.append(f"'{col}' sabit değerli — model eğitiminden çıkarılmalı")
            continue

        # Date columns → feature extraction
        if is_date:
            recs.append(
                f"'{col}' tarih kolonu — yıl/ay/gün/haftanın_günü özelliklerine dönüştür"
            )
            continue

        # High null rate
        if null_rate > 0.3:
            recs.append(
                f"'{col}' çok yüksek null oranı (%{null_rate * 100:.0f}) — drop veya agresif imputation"
            )
            continue

        # Numeric columns
        if col in numeric_cols:
            parts = []

            if is_monetary and has_significant_outliers:
                parts.append(f"parasal + outlier (%{outlier_rate*100:.1f}) → RobustScaler önerilir")
            elif is_monetary:
                parts.append("parasal kolon → StandardScaler önerilir")
            elif has_significant_outliers:
                parts.append(f"outlier oranı %{outlier_rate*100:.1f} → RobustScaler veya IQR capping önerilir")

            # Skewness is reported independently (even alongside outlier warning)
            if skew is not None and abs(skew) > 1:
                parts.append(f"çarpık dağılım (skewness={skew:.2f}) → log/sqrt transform önerilir")

            if parts:
                recs.append(f"'{col}': {' | '.join(parts)}")
            continue

        # Categorical columns
        if dtype == "object":
            if unique_rate > 0.95:
                recs.append(
                    f"'{col}' neredeyse unique — drop veya hash encoding önerilir"
                )
            elif cardinality == "high":
                recs.append(
                    f"'{col}' yüksek kardinaliteli ({unique_count} unique) — target encoding önerilir"
                )
            elif cardinality == "medium":
                recs.append(
                    f"'{col}' orta kardinaliteli ({unique_count} unique) — label encoding önerilir"
                )
            elif cardinality == "low":
                recs.append(
                    f"'{col}' düşük kardinaliteli ({unique_count} unique) — one-hot encoding önerilir"
                )

    return recs


# Main node
def execution_node(state: AgentState) -> AgentState:
    """
    Executes the analysis plan created by the planner.

    Uses intelligence_engine's db_intelligence to:
    1. Start from the tables selected by the planner
    2. Traverse the relationship graph to discover ALL connected tables
    3. Pull tables via MCP in parallel
    4. Merge using Python LEFT JOIN (NOT SQL JOIN)
    5. Generate comprehensive profiling + feature engineering recommendations

    State outputs:
        sql_results      : Merged data (records, max 10K rows)
        profiling_report : Statistical profile + recommendations
    """
    analysis_plan: list[dict] = state.get("analysis_plan", [])
    db_intelligence: dict = state.get("db_intelligence", {})

    if not analysis_plan:
        return {**state, "error": "Analiz planı boş.", "current_step": "execution_node"}

    relationships: list[dict] = db_intelligence.get("relationships", [])
    all_known_tables: list[str] = [
        t["table_name"] for t in db_intelligence.get("tables", [])
    ]

    df: pd.DataFrame | None = None
    sql_results: list[dict] = []
    profiling_report: dict = {}

    for step in analysis_plan:
        step_name = step.get("step", "")

        # SQL_QUERY
        if step_name == "sql_query":
            try:
                selected_tables: list[str] = step.get("table_names", [])

                if not selected_tables:
                    tables = db_intelligence.get("tables", [])
                    if tables:
                        best = max(tables, key=lambda t: t.get("analytics_potential_score", 0))
                        selected_tables = [best["table_name"]]

                if not selected_tables:
                    raise ValueError("SQL için tablo bulunamadı.")

                # Relationship graph traverse
                all_tables = _resolve_all_related_tables(
                    selected_tables, relationships, all_known_tables
                )
                print(f"[execution_node] Planner seçti  : {selected_tables}")
                print(f"[execution_node] Graph traverse  : {all_tables}")

                # Parallel table fetch
                raw_results = asyncio.run(_fetch_all_tables(all_tables))

                # Parse → DataFrame
                dfs: dict[str, pd.DataFrame] = {}
                for table, raw in raw_results.items():
                    dfs[table] = _parse_mcp_response_to_dataframe(raw, table)
                    print(f"[execution_node] '{table}': {len(dfs[table])} satır")

                # Anchor = most rows
                anchor = max(dfs, key=lambda t: len(dfs[t]))
                merge_ops = _build_merge_order(all_tables, relationships, anchor)
                df = _merge_all_dataframes(dfs, merge_ops, anchor)

                print(f"[execution_node] Final: {len(df)} satır x {len(df.columns)} kolon")
                sql_results = df.where(pd.notnull(df), None).to_dict(orient="records")

            except Exception as e:
                return {**state, "error": f"SQL hatası: {str(e)}",
                        "current_step": "execution_node"}

        # PROFILING
        elif step_name == "profiling":
            if df is None or df.empty:
                return {**state, "error": "Profiling için veri yok.",
                        "current_step": "execution_node"}
            try:
                profiling_report = _profile_dataframe(df)
                shape = profiling_report["shape"]
                print(f"[execution_node] Profiling: "
                      f"{shape['rows']} satır x {shape['columns']} kolon | "
                      f"{len(profiling_report.get('recommendations', []))} öneri")
            except Exception as e:
                return {**state, "error": f"Profiling hatası: {str(e)}",
                        "current_step": "execution_node"}

        else:
            continue

    return {
        **state,
        "sql_results": sql_results,
        "profiling_report": profiling_report,
        "current_step": "execution_node",
    }