from __future__ import annotations

"""
Execution Node — BigData Edition
---------------------------------
DuckDB tabanlı execution layer.

Mimari:
  1. MCP'den tablolar paralel çekilir (chunked streaming)
  2. DuckDB in-memory'e yüklenir
  3. Profiling: DuckDB üzerinde full scan (hızlı, RAM dostu)
  4. ML sampling: stratified sample (sınıf dengesi korunur)
  5. State'e veri değil parquet path yazılır (lazy loading)
  6. Feature engineering parquet'ten okur (chunk by chunk)

Neden DuckDB:
  - 10M satır GROUP BY → milisaniyeler (SQLite → dakikalar)
  - Columnar storage → aggregation için optimize
  - Parallel query execution
  - Parquet okuma/yazma native destek
  - In-memory mod → kurulum gerektirmez
"""

import ast
import asyncio
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from mcp.client.streamable_http import streamable_http_client
from langchain_core.runnables import RunnableConfig
from mcp.client.session import ClientSession

from src.core.config import settings

# ── Sabitler ────────────────────────────────────────────────
CHUNK_SIZE        = 5_000   # MCP'den kaç satır çekileceği (per chunk) — küçük chunk = hızlı sorgu, daha az iptal/spin riski
MAX_FETCH_ROWS    = 300_000  # Tek tablodan max çekilecek satır
ML_SAMPLE_ROWS    = 300_000  # ML eğitimi için max satır
PROFILE_SAMPLE    = 300_000  # Profiling için max satır (yoksa full)
PARQUET_DIR       = Path(tempfile.gettempdir()) / "ds_agent_data"
PARQUET_DIR.mkdir(exist_ok=True)



def _coerce_scalar(p: str):
    """
    Repr wrapper'larını gerçek değere indirger:
      np.int64(0)        -> 0
      np.float64(1.5)    -> 1.5
      Decimal('10.5')    -> 10.5
      Timestamp('...')   -> '...' (string, parquet datetime'a çevirir)
      Period('2017-09')  -> '2017-09'
      <NA>/nan/None/null -> None
    Wrapper yoksa int -> float -> string sırasıyla dener.
    """
    s = p.strip()

    # Genel wrapper: AnyName('...') veya AnyName(123) -> içindeki değer
    m = re.match(r"[\w.]+\(\s*['\"]?(.+?)['\"]?\s*\)$", s)
    if m:
        s = m.group(1).strip()

    # Null varyantları
    if s.lower() in ("nan", "na", "<na>", "none", "null", ""):
        return None

    # Kalan tırnakları temizle
    if len(s) >= 2 and s[0] in "'\"" and s[-1] in "'\"":
        s = s[1:-1]

    # Sayıya çevir
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s

# ── Relationship graph traversal ─────────────────────────────
def _resolve_all_related_tables(
    selected: list[str],
    relationships: list[dict],
    all_known: list[str],
) -> list[str]:
    graph: dict[str, list[str]] = {t: [] for t in all_known}
    for rel in relationships:
        src = rel.get("source_table"); tgt = rel.get("target_table")
        if src in graph and tgt in graph:
            graph[src].append(tgt); graph[tgt].append(src)
    visited = set(selected); queue = list(selected)
    while queue:
        cur = queue.pop(0)
        for nb in graph.get(cur, []):
            if nb not in visited:
                visited.add(nb); queue.append(nb)
    return list(visited)


def _build_merge_order(
    tables: list[str],
    relationships: list[dict],
    anchor: str,
) -> list[dict]:
    ops: list[dict] = []
    merged = {anchor}; remaining = set(tables) - merged
    for _ in range(len(tables) * 2):
        if not remaining: break
        for rel in relationships:
            src = str(rel.get("source_table",""))
            tgt = str(rel.get("target_table",""))
            sc  = str(rel.get("source_column",""))
            tc  = str(rel.get("target_column",""))
            if not (src and tgt and sc and tc): continue
            if src in merged and tgt in remaining:
                ops.append({"left":src,"right":tgt,"left_on":sc,"right_on":tc})
                merged.add(tgt); remaining.discard(tgt)
            elif tgt in merged and src in remaining:
                ops.append({"left":tgt,"right":src,"left_on":tc,"right_on":sc})
                merged.add(src); remaining.discard(src)
    return ops

def _build_joined_sql(
    anchor: str,
    merge_ops: list[dict],
    dfs: dict[str, "pd.DataFrame"],
    fetch_limit: int,
) -> str:
    """
    relationships dict'inden dinamik SQL JOIN sorgusu üretir.
    Hiçbir tablo/kolon ismi hardcode değil.

    Duplicate column sorunu: SELECT t.*, a.*, m.* yaparsak
    account_id hem transactions hem accounts'ta var → çakışır.
    Çözüm: anchor.* + dimension kolonlarını suffix'li alias ile seç.
    Join key'i tekrar almayız (zaten anchor'da var).

    Örnek çıktı:
        SELECT t.*, a."customer_name" AS "customer_name_accounts", ...
        FROM "transactions" t
        INNER JOIN "accounts" a ON t."account_id" = a."account_id"
        LIMIT 500000
    """
    anchor_alias = anchor[0].lower()
    join_parts: list[str] = []
    alias_map: dict[str, str] = {anchor: anchor_alias}
    used_aliases: set[str] = {anchor_alias}

    # Anchor kolonları
    anchor_cols = set(dfs[anchor].columns) if anchor in dfs else set()
    select_parts: list[str] = [f'{anchor_alias}.*']

    for op in merge_ops:
        right = op["right"]
        left  = op["left"]
        lo    = op["left_on"]
        ro    = op["right_on"]

        # Benzersiz alias üret
        base_alias = right[0].lower()
        alias = base_alias
        counter = 2
        while alias in used_aliases:
            alias = f"{base_alias}{counter}"
            counter += 1
        used_aliases.add(alias)
        alias_map[right] = alias

        left_alias = alias_map.get(left, anchor_alias)

        join_parts.append(
            f'LEFT JOIN "{right}" {alias} ON {left_alias}."{lo}" = {alias}."{ro}"'
        )

        # Dimension kolonlarını suffix'li alias ile seç (join key hariç)
        if right in dfs:
            for col in dfs[right].columns:
                if col == ro:
                    continue  # join key → zaten anchor'da var
                select_parts.append(f'{alias}."{col}" AS "{col}_{right}"')

    select_clause = ', '.join(select_parts)
    sql = (
        f'SELECT {select_clause} '
        f'FROM "{anchor}" {anchor_alias} '
        + ' '.join(join_parts)
    )
    return sql



def _select_anchor_table(
    dfs: dict[str, "pd.DataFrame"],
    relationships: list[dict],
    selected_tables: list[str],
) -> str:
    """
    Fact table'ı identify ederek anchor seçer.

    Database normalization teorisi:
    ──────────────────────────────────────────────────────────────
    Fact table     = İlişkilerde KAYNAK (source) olur, FK'leri verir.
                     transactions.account_id → accounts.account_id
    Dimension table= İlişkilerde HEDEF (target) olur, FK'leri alır.
                     accounts tablosu transactions'tan FK alır.

    Bu evrensel bir kuraldır — domain, dil, isim bağımsız.

    Skor:
    - Outgoing FK (source_table): +3 puan  → fact table işareti
    - Incoming FK (target_table): -1 puan  → dimension table işareti
    - Tie-breaker: daha fazla satıra sahip olan
    """
    if len(dfs) == 1:
        return list(dfs.keys())[0]

    scores: dict[str, dict] = {}
    for table in dfs:
        outgoing = sum(1 for r in relationships if r.get("source_table") == table)
        incoming = sum(1 for r in relationships if r.get("target_table") == table)
        scores[table] = {
            "outgoing": outgoing,
            "incoming": incoming,
            "score":    outgoing * 3 - incoming,
            "rows":     len(dfs[table]),
        }
        print(f"[execution] '{table}': outgoing_FK={outgoing}, incoming_FK={incoming}, "
              f"score={scores[table]['score']}, rows={len(dfs[table]):,}")

    # En yüksek skor → fact table → anchor
    # Tie varsa daha fazla satıra sahip olan
    anchor = max(scores, key=lambda t: (scores[t]["score"], scores[t]["rows"]))
    print(f"[execution] Anchor (fact table): '{anchor}'")
    return anchor



def _parse_mcp_to_df(raw: str, table_name: str = "") -> pd.DataFrame:

    if not raw:
        raise ValueError(f"[{table_name}] MCP boş yanıt döndürdü.")
    if raw.strip().startswith("Error") or raw.strip().startswith("SQL Error"):
        raise ValueError(f"[{table_name}] MCP SQL hatası: {raw[:200]}")

    lines = raw.strip().splitlines()
    columns: list[str] = []
    for line in lines:
        if line.startswith("Columns:"):
            columns = [c.strip() for c in line.replace("Columns:","").split(",")]
            break
    if not columns:
        raise ValueError(f"[{table_name}] Kolon bilgisi bulunamadı.")

    rows = []
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
    return _infer_dtypes(df)


def _pad_or_trim(row: tuple, n: int) -> tuple:
    lst = list(row)
    lst.extend([None] * max(0, n - len(lst)))
    return tuple(lst[:n])


def _parse_tuple_line(line: str) -> tuple:
    try:
        return ast.literal_eval(line)
    except (ValueError, SyntaxError):
        inner = line[1:-1]
        parts = re.split(r",\s*(?=(?:[^'\"]*['\"][^'\"]*['\"])*[^'\"]*$)", inner)
        parsed = []
        for p in parts:
            p = p.strip()
            if p in ("None","NULL","null",""):
                parsed.append(None)
            elif p.startswith(("'",'"')) and p.endswith(("'",'"')):
                parsed.append(p[1:-1])
            else:
                parsed.append(_coerce_scalar(p))
        return tuple(parsed)


def _infer_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        conv = pd.to_numeric(df[col], errors="coerce")
        if conv.notna().mean() > 0.8:
            df[col] = conv; continue
        if df[col].dtype == object:
            try:
                sample = df[col].dropna().head(10).astype(str)
                if sample.str.match(r"\d{4}-\d{2}-\d{2}").mean() > 0.7:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
    return df


# ── MCP async fetcher (chunked) ──────────────────────────────
async def _fetch_all_tables(table_to_db: dict[str, str], data_group_id: int) -> dict[str, str]:
    from api.core.database import SessionLocal
    from api.schemas.data_group import run_data_group_query_logic, QueryRequest
    from api.models.data_group import DataGroupModel

    db_conn = SessionLocal()
    try:
        dg = db_conn.query(DataGroupModel).filter(DataGroupModel.id == data_group_id).first()
        user_id = dg.user_id if dg else 0
    finally:
        db_conn.close()

    async def fetch_one(table: str, db_name: str) -> tuple[str, str]:
        db = SessionLocal()
        try:
            # Satır sayısını öğren (opsiyonel, log için)
            try:
                req = QueryRequest(query=f'SELECT COUNT(*) FROM "{table}"')
                count_result = await run_data_group_query_logic(data_group_id, req, db, user_id)
                count_text = str(count_result.get("result", "")) if count_result.get("status") == "success" else ""
                total_rows = _extract_count(count_text)
            except Exception:
                total_rows = 0

            print(f"[execution] '{table}': {total_rows:,} satır tespit edildi")

            # TEK SORGU — chunk yok. MAX_FETCH_ROWS kadar LIMIT'le.
            fetch_limit = min(total_rows, MAX_FETCH_ROWS) if total_rows else MAX_FETCH_ROWS
            query = f'SELECT * FROM "{table}" LIMIT {fetch_limit}'
            req = QueryRequest(query=query)
            result = await run_data_group_query_logic(data_group_id, req, db, user_id)

            if result.get("status") == "success":
                print(f"[execution] '{table}': {fetch_limit:,} satır tek sorguda çekildi")
                return table, str(result.get("result", ""))
            return table, ""
        finally:
            db.close()

    results = await asyncio.gather(
        *[fetch_one(t, db) for t, db in table_to_db.items()],
        return_exceptions=True
    )

    output: dict[str, str] = {}
    for res in results:
        if isinstance(res, BaseException):
            raise res
        table, raw = res
        output[table] = raw
    return output


def _extract_count(text: str) -> int:
    """MCP COUNT(*) sonucundan satır sayısını çıkarır."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("(") and line.endswith(")"):
            try:
                val = ast.literal_eval(line)
                if isinstance(val, tuple) and len(val) == 1:
                    return int(val[0])
            except Exception:
                pass
    return 50_000  # fallback


# ── DuckDB engine ────────────────────────────────────────────
class DuckDBEngine:
    """
    DuckDB in-memory analiz motoru.

    Görevleri:
    - DataFrame'leri DuckDB tablosuna yükle
    - Profiling sorgularını çalıştır (full scan, hızlı)
    - Stratified sample üret (ML eğitimi için)
    - Parquet'e yaz (lazy loading)
    """

    def __init__(self):
        self.conn = duckdb.connect(":memory:")
        self._tables: set[str] = set()

    def load(self, name: str, df: pd.DataFrame) -> None:
        """DataFrame'i DuckDB'ye yükle."""
        safe_name = name.replace(".", "_").replace("-", "_")
        self.conn.register(safe_name, df)
        self._tables.add(safe_name)
        print(f"[duckdb] '{safe_name}' yüklendi: {len(df):,} satır × {len(df.columns)} kolon")

    def query(self, sql: str) -> pd.DataFrame:
        """SQL sorgusu çalıştır, DataFrame döndür."""
        return self.conn.execute(sql).df()

    def profile_table(self, name: str) -> dict:
        """
        DuckDB üzerinde hızlı profiling.
        Full scan yapar ama columnar olduğu için hızlı.
        """
        safe = name.replace(".", "_").replace("-", "_")
        if safe not in self._tables:
            raise ValueError(f"'{safe}' DuckDB'de yok.")

        # Genel istatistikler
        shape_df = self.conn.execute(
            f"SELECT COUNT(*) as rows, COUNT(*) as cols FROM {safe} LIMIT 1"
        ).df()

        # Kolon bazlı istatistik
        col_df  = self.conn.execute(f"DESCRIBE {safe}").df()
        columns = col_df["column_name"].tolist()
        dtypes  = dict(zip(col_df["column_name"], col_df["column_type"]))

        col_profiles = {}
        for col in columns:
            safe_col = f'"{col}"'
            dtype    = dtypes.get(col, "VARCHAR").upper()
            is_num   = any(t in dtype for t in ("INT","FLOAT","DOUBLE","DECIMAL","REAL","BIGINT","HUGEINT","UBIGINT"))
            is_date  = any(t in dtype for t in ("DATE","TIME","TIMESTAMP"))

            try:
                null_result = self.conn.execute(
                    f"SELECT COUNT(*) as total, COUNT({safe_col}) as non_null FROM {safe}"
                ).fetchone()
                total     = null_result[0] if null_result else 1
                non_null  = null_result[1] if null_result else 0
                null_cnt  = total - non_null
                null_rate = round(null_cnt / max(total, 1), 4)

                uniq_result = self.conn.execute(
                    f"SELECT COUNT(DISTINCT {safe_col}) FROM {safe}"
                ).fetchone()
                unique_cnt  = uniq_result[0] if uniq_result else 0
                unique_rate = round(unique_cnt / max(total, 1), 4)

                cp: dict[str, Any] = {
                    "dtype":       dtype,
                    "null_count":  null_cnt,
                    "null_rate":   null_rate,
                    "unique_count":unique_cnt,
                    "unique_rate": unique_rate,
                }

                if is_num:
                    stats = self.conn.execute(f"""
                        SELECT
                            MIN({safe_col}), MAX({safe_col}),
                            AVG({safe_col}), MEDIAN({safe_col}),
                            STDDEV({safe_col}),
                            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {safe_col}),
                            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {safe_col})
                        FROM {safe}
                    """).fetchone()
                    if stats:
                        mn,mx,avg,med,std,q1,q3 = stats
                        iqr = (q3 or 0) - (q1 or 0)
                        outlier_rate = 0.0
                        if iqr > 0:
                            out_df = self.conn.execute(f"""
                                SELECT COUNT(*) FROM {safe}
                                WHERE {safe_col} < {q1 - 1.5*iqr}
                                   OR {safe_col} > {q3 + 1.5*iqr}
                            """).fetchone()
                            outlier_rate = round((out_df[0] if out_df else 0) / max(total, 1), 4)
                        cp.update({
                            "min": _sf(mn), "max": _sf(mx),
                            "mean": _sf(avg), "median": _sf(med),
                            "std": _sf(std), "q25": _sf(q1), "q75": _sf(q3),
                            "has_outliers": outlier_rate,
                        })

                elif is_date:
                    date_stats = self.conn.execute(
                        f"SELECT MIN({safe_col}), MAX({safe_col}) FROM {safe}"
                    ).fetchone()
                    if date_stats:
                        cp.update({"min_date": str(date_stats[0]), "max_date": str(date_stats[1])})

                else:
                    top_vals = self.conn.execute(f"""
                        SELECT {safe_col}, COUNT(*) as cnt
                        FROM {safe} WHERE {safe_col} IS NOT NULL
                        GROUP BY {safe_col} ORDER BY cnt DESC LIMIT 5
                    """).fetchall()
                    cp["top_values"] = {str(r[0]): r[1] for r in top_vals}
                    cp["cardinality"] = (
                        "high" if unique_cnt > 50
                        else "medium" if unique_cnt > 10
                        else "low"
                    )

                col_profiles[col] = cp

            except Exception as e:
                col_profiles[col] = {"dtype": dtype, "error": str(e)}

        # Tam satır sayısı
        count_row = self.conn.execute(f"SELECT COUNT(*) FROM {safe}").fetchone()
        total_rows = count_row[0] if count_row else 0
        numeric_cols   = [c for c,d in dtypes.items() if any(t in d.upper() for t in ("INT","FLOAT","DOUBLE","DECIMAL","REAL","BIGINT"))]
        categorical_cols = [c for c,d in dtypes.items() if "VARCHAR" in d.upper() or "TEXT" in d.upper()]
        datetime_cols  = [c for c,d in dtypes.items() if any(t in d.upper() for t in ("DATE","TIME","TIMESTAMP"))]

        from typing import Any
        profile: dict[str, Any] = {
            "shape": {"rows": total_rows, "columns": len(columns)},
            "summary": {
                "numeric_columns":    numeric_cols,
                "categorical_columns":categorical_cols,
                "datetime_columns":   datetime_cols,
                "missing_rate":       round(sum(cp.get("null_rate",0) for cp in col_profiles.values()) / max(len(col_profiles),1), 4),
                "duplicate_rows":     0,  # DuckDB'de pahalı, skip
            },
            "columns": col_profiles,
        }
        profile["recommendations"] = _generate_recommendations(profile)
        return profile

    def stratified_sample(
        self,
        name: str,
        target_col: str | None,
        n: int = ML_SAMPLE_ROWS,
    ) -> pd.DataFrame:
        """
        Stratified sample üretir.
        target_col varsa sınıf dağılımını korur.
        """
        safe = name.replace(".", "_").replace("-", "_")
        total_row = self.conn.execute(f"SELECT COUNT(*) FROM {safe}").fetchone()
        total = total_row[0] if total_row else 0

        if total <= n:
            print(f"[duckdb] '{safe}' zaten küçük ({total:,} satır), tüm veri kullanılıyor.")
            return self.conn.execute(f"SELECT * FROM {safe}").df()

        if target_col and target_col in self._get_columns(safe):
            # Stratified: her sınıftan orantılı örnekle
            classes = self.conn.execute(
                f'SELECT DISTINCT "{target_col}" FROM {safe} WHERE "{target_col}" IS NOT NULL'
            ).fetchall()
            frames = []
            for (cls,) in classes:
                cls_count_row = self.conn.execute(
                    f'SELECT COUNT(*) FROM {safe} WHERE "{target_col}" = $1', [cls]
                ).fetchone()
                cls_count = cls_count_row[0] if cls_count_row else 0
                cls_sample = max(1, int(n * cls_count / total)) if total > 0 else 0
                df_cls = self.conn.execute(
                    f'SELECT * FROM {safe} WHERE "{target_col}" = $1 USING SAMPLE {cls_sample}', [cls]
                ).df()
                frames.append(df_cls)
            result = pd.concat(frames, ignore_index=True)
            print(f"[duckdb] Stratified sample: {len(result):,}/{total:,} satır")
            return result
        else:
            # Random sample
            result = self.conn.execute(
                f"SELECT * FROM {safe} USING SAMPLE {n}"
            ).df()
            print(f"[duckdb] Random sample: {len(result):,}/{total:,} satır")
            return result

    def to_parquet(self, name: str, path: Path) -> None:
        """DuckDB tablosunu parquet'e yaz."""
        safe = name.replace(".", "_").replace("-", "_")
        self.conn.execute(f"COPY {safe} TO '{path}' (FORMAT PARQUET)")
        print(f"[duckdb] '{safe}' → {path} ({path.stat().st_size//1024:,} KB)")

    def _get_columns(self, safe_name: str) -> list[str]:
        try:
            return self.conn.execute(f"DESCRIBE {safe_name}").df()["column_name"].tolist()
        except Exception:
            return []

    def close(self):
        self.conn.close()


def _sf(val: Any) -> float | None:
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return None


# ── Pandas merge (küçük tablolar için fallback) ──────────────
def _merge_dataframes(
    dfs: dict[str, pd.DataFrame],
    merge_ops: list[dict],
    anchor: str,
) -> pd.DataFrame:
    if len(dfs) == 1:
        return list(dfs.values())[0]
    result = dfs[anchor].copy()
    for op in merge_ops:
        rt = op["right"]
        if rt not in dfs: continue
        rdf = dfs[rt].copy()
        overlap = (set(result.columns) & set(rdf.columns)) - {op["left_on"], op["right_on"]}
        if overlap:
            rdf = rdf.rename(columns={c: f"{c}_{rt}" for c in overlap})
        before = len(result)
        result = result.merge(rdf, left_on=op["left_on"], right_on=op["right_on"], how="left")
        print(f"[merge] {op['left']}.{op['left_on']} → {rt}.{op['right_on']} | {before}→{len(result)}")
    return result


# ── Recommendations (DuckDB profil çıktısından) ──────────────
def _generate_recommendations(profile: dict) -> list[str]:
    recs = []
    summary = profile.get("summary", {})
    numeric_cols = set(summary.get("numeric_columns", []))
    missing_rate = summary.get("missing_rate", 0)

    if missing_rate > 0.05:
        recs.append(f"Eksik değer imputation gerekli (missing_rate: %{missing_rate*100:.1f})")
    if summary.get("duplicate_rows", 0) > 0:
        recs.append(f"Duplicate satır temizlenmeli ({summary['duplicate_rows']} adet)")

    DATE_SUFFIXES = ("_date","_at","_time")
    DATE_EXACT = {"created_at","updated_at","timestamp","birth_date","order_date"}
    OUTLIER_THRESHOLD = 0.01

    for col, info in profile.get("columns", {}).items():
        if "error" in info: continue
        dtype       = info.get("dtype","").upper()
        col_lower   = col.lower()
        is_id       = col_lower.endswith("_id") or col_lower == "id"
        is_date_col = ("DATE" in dtype or "TIME" in dtype or
                       col_lower in DATE_EXACT or
                       any(col_lower.endswith(s) for s in DATE_SUFFIXES))
        is_monetary = any(k in col_lower for k in ["price","amount","cost","revenue"])

        outlier_rate = info.get("has_outliers", 0.0)
        if isinstance(outlier_rate, bool):
            outlier_rate = 0.05 if outlier_rate else 0.0
        has_outliers = outlier_rate > OUTLIER_THRESHOLD

        null_rate    = info.get("null_rate", 0)
        unique_cnt   = info.get("unique_count", 0)
        unique_rate  = info.get("unique_rate", 0)
        cardinality  = info.get("cardinality","")

        if is_id:
            recs.append(f"'{col}' ID sütun — model eğitiminden çıkarılmalı"); continue
        if unique_cnt == 1:
            recs.append(f"'{col}' sabit değerli — çıkarılmalı"); continue
        if is_date_col:
            recs.append(f"'{col}' tarih kolonu — yıl/ay/gün özelliklerine dönüştür"); continue
        if null_rate > 0.3:
            recs.append(f"'{col}' çok yüksek null (%{null_rate*100:.0f}) — drop veya agresif imputation"); continue

        if col in numeric_cols:
            parts = []
            if is_monetary and has_outliers:
                parts.append(f"parasal+outlier (%{outlier_rate*100:.1f}) → RobustScaler")
            elif is_monetary:
                parts.append("parasal → StandardScaler")
            elif has_outliers:
                parts.append(f"outlier %{outlier_rate*100:.1f} → RobustScaler/IQR capping")
            skew = info.get("skewness")
            if skew is not None and abs(skew) > 1:
                parts.append(f"çarpık (skew={skew:.2f}) → log/sqrt transform")
            if parts:
                recs.append(f"'{col}': {' | '.join(parts)}")
        elif "VARCHAR" in dtype or "TEXT" in dtype:
            if unique_rate > 0.95:
                recs.append(f"'{col}' neredeyse unique → drop/hash encoding")
            elif cardinality == "high":
                recs.append(f"'{col}' yüksek kardinalite ({unique_cnt}) → target encoding")
            elif cardinality == "medium":
                recs.append(f"'{col}' orta kardinalite ({unique_cnt}) → label encoding")
            elif cardinality == "low":
                recs.append(f"'{col}' düşük kardinalite ({unique_cnt}) → one-hot encoding")
    return recs


# ── Ana node ─────────────────────────────────────────────────
async def execute_data_operations(state: dict, config: RunnableConfig) -> dict:
    data_group_id = config.get("configurable", {}).get("data_group_id")
    if not data_group_id:
        return {"error": "data_group_id is missing in config", "current_step": "execution_node"}

    """
    BigData-ready execution node.

    1. MCP'den chunked parallel fetch
    2. DuckDB'ye yükle
    3. Profiling: DuckDB full scan
    4. Parquet'e yaz → state'e path kaydet
    5. ML için stratified sample
    """
    analysis_plan  = state.get("analysis_plan", [])
    print(f"[DEBUG EXEC] analysis_plan = {analysis_plan}")
    db_intelligence= state.get("db_intelligence", {})

    if not analysis_plan:
        return {"error": "Analiz planı boş.", "current_step": "execution_node"}

    relationships    = db_intelligence.get("relationships", [])
    all_known_tables = [t["table_name"] for t in db_intelligence.get("tables", [])]

    duck = DuckDBEngine()
    sql_results:     list[dict] = []
    profiling_report:dict       = {}
    data_path:       str        = ""
    merged_table:    str        = ""

    try:
        for step in analysis_plan:
            step_name = step.get("step", "").lower()

            # SQL query
            if step_name in ("sql_query",) or "veri" in step_name and ("çek" in step_name or "fetch" in step_name or "extract" in step_name) or step.get("sql_query", "").strip():
                selected = step.get("table_names", [])
                if not selected:
                    tables = db_intelligence.get("tables",[])
                    if tables:
                        best = max(tables, key=lambda t: t.get("analytics_potential_score",0))
                        selected = [best["table_name"]]
                if not selected:
                    raise ValueError("SQL için tablo bulunamadı.")

                all_tables = _resolve_all_related_tables(selected, relationships, all_known_tables)

                # db_name mapping
                tables_info = db_intelligence.get("tables",[])
                table_to_db = {}
                for t in all_tables:
                    db_name = "default"
                    for ti in tables_info:
                        if ti.get("table_name") == t:
                            db_name = str(ti.get("db_name","default")).strip().strip('",\'"')
                            break
                    table_to_db[t] = db_name

                print(f"[execution] Planner: {selected}")
                print(f"[execution] Graph traverse: {all_tables}")

                # Parallel chunked fetch
                raw_results = await _fetch_all_tables(table_to_db, data_group_id)

                # Parse -> DuckDB
                dfs: dict[str, pd.DataFrame] = {}
                for table, raw in raw_results.items():
                    df = _parse_mcp_to_df(raw, table)
                    dfs[table] = df
                    duck.load(table, df)
                    print(f"[execution] '{table}': {len(df):,} satır")

                # Multi-table merge (DuckDB SQL JOIN - fast)
                # Multi-table: tek SQL JOIN ile MCP'den çek
                if len(all_tables) > 1:
                    anchor     = _select_anchor_table(dfs, relationships, all_tables)
                    merge_ops  = _build_merge_order(all_tables, relationships, anchor)

                    if merge_ops:
                        # JOIN SQL'ini dinamik olarak oluştur (LIMIT/OFFSET hariç base SQL)
                        base_join_sql = _build_joined_sql(anchor, merge_ops, dfs, MAX_FETCH_ROWS)
                        print(f"[execution] DB-side JOIN Base SQL:\n  {base_join_sql[:200]}...")

                        tables_info = db_intelligence.get("tables", [])
                        anchor_db = "default"
                        for ti in tables_info:
                            if ti.get("table_name") == anchor:
                                anchor_db = str(ti.get("db_name", "default")).strip().strip('",\'')
                                break

                        # Chunked Fetch
                        chunks_text = []
                        header_written = False
                        
                        # Use a safe chunk size below MAX_DISPLAY_ROWS
                        SAFE_CHUNK_SIZE = 5_000
                        
                        from api.core.database import SessionLocal
                        from api.schemas.data_group import run_data_group_query_logic, QueryRequest
                        from api.models.data_group import DataGroupModel
                        db_conn = SessionLocal()
                        try:
                            dg = db_conn.query(DataGroupModel).filter(DataGroupModel.id == data_group_id).first()
                            user_id = dg.user_id if dg else 0
                            for offset in range(0, MAX_FETCH_ROWS, SAFE_CHUNK_SIZE):
                                limit = min(SAFE_CHUNK_SIZE, MAX_FETCH_ROWS - offset)
                                chunk_sql = f"{base_join_sql} LIMIT {limit} OFFSET {offset}"
                                
                                req = QueryRequest(query=chunk_sql)
                                result = await run_data_group_query_logic(data_group_id, req, db_conn, user_id)
                                chunk_text = str(result.get("result", "")) if result.get("status") == "success" else ""
                                
                                # Eğer data dönmediyse break
                                if "Returned 0 rows" in chunk_text or not chunk_text:
                                    break
                                    
                                if not header_written:
                                    chunks_text.append(chunk_text)
                                    header_written = True
                                else:
                                    data_lines = [l for l in chunk_text.splitlines() if l.strip().startswith("(")]
                                    if not data_lines:
                                        break
                                    chunks_text.append("\n".join(data_lines))
                                    
                                print(f"[execution] JOIN chunk: {offset+limit:,}/{MAX_FETCH_ROWS:,} satır çekildi")
                        finally:
                            db_conn.close()

                        raw_joined = "\n".join(chunks_text)
                        merged_df = _parse_mcp_to_df(raw_joined, "joined")
                        print(f"[execution] DB-side JOIN sonucu: {len(merged_df):,} satır × {len(merged_df.columns)} kolon")
                    else:
                        # merge_ops boş → tablolar arasında bilinen ilişki yok, anchor kullan
                        merged_df = dfs[anchor]

                    merged_name = f"merged_{uuid.uuid4().hex[:8]}"
                    duck.load(merged_name, merged_df)
                    merged_table = merged_name
                else:
                    # Tek tablo — zaten yukarıda duck.load() ile yüklendi
                    single_name = list(dfs.keys())[0]
                    merged_table = single_name.replace(".", "_").replace("-", "_")

                print(f"[execution] Merged table: '{merged_table}'")

                # Parquet'e yaz
                parquet_path = PARQUET_DIR / f"{merged_table}.parquet"
                duck.to_parquet(merged_table, parquet_path)
                data_path = str(parquet_path)
                print(f"[execution] Parquet yazıldı: {data_path}")
                # Preview (1K satır)
                preview = duck.query(f"SELECT * FROM {merged_table} LIMIT 1000")
                sql_results = preview.where(pd.notnull(preview), None).to_dict(orient="records")  # type: ignore[arg-type]

            # ── PROFILING ──────────────────────────────────────
            elif step_name in ("profiling",) or "profil" in step_name:
                if not merged_table:
                    raise ValueError("Profiling için tablo yok.")
                profiling_report = duck.profile_table(merged_table)
                shape = profiling_report["shape"]
                print(f"[execution] Profiling: {shape['rows']:,}×{shape['columns']} | "
                      f"{len(profiling_report.get('recommendations',[]))} öneri")

            else:
                continue

    except Exception as e:
        import traceback
        print(f"\n[execution] HATA: {e}")
        print(traceback.format_exc())
        duck.close()
        return {"error": f"Execution hatası: {str(e)}", "current_step": "execution_node"}

    finally:
        duck.close()

    if not data_path and not sql_results:
        step_names = [s.get("step", "") for s in analysis_plan]
        print(f"[execution] HATA: data_path boş, sql_results boş. Plan adımları: {step_names}")
        return {
            "error": (
                "Veri çekme adımı çalışmadı. Analiz planı bir 'sql_query' adımı içermiyor "
                f"veya tanınmayan adım isimleri var: {step_names}. "
                "Lütfen projeyi yeniden başlatın."
            ),
            "current_step": "execution_node",
        }

    return {
        "sql_results": sql_results,
        "data_path": data_path,
        "profiling_report": profiling_report,
        "error": None
    }


def _merge_with_duckdb(
    duck: DuckDBEngine,
    dfs: dict[str, pd.DataFrame],
    merge_ops: list[dict],
    anchor: str,
) -> pd.DataFrame:
    """
    DuckDB SQL JOIN ile multi-table merge.
    Pandas merge'den çok daha hızlı büyük tablolarda.
    """
    if not merge_ops:
        return dfs[anchor]

    # Tüm tabloları DuckDB'de zaten var
    # SQL JOIN zinciri oluştur
    safe_anchor = anchor.replace(".","_").replace("-","_")
    select_parts = [f"{safe_anchor}.*"]
    join_parts   = []

    for op in merge_ops:
        rt   = op["right"].replace(".","_").replace("-","_")
        lo   = op["left_on"]
        ro   = op["right_on"]
        left = op["left"].replace(".","_").replace("-","_")

        # Overlap kolonları prefix'le
        rdf_cols = duck._get_columns(rt)
        safe_cols = []
        for c in rdf_cols:
            if c == ro:
                continue  # join key'i tekrar alma
            safe_cols.append(f'{rt}."{c}" AS "{c}_{rt}"')
        select_parts.extend(safe_cols)
        join_parts.append(f'LEFT JOIN {rt} ON {left}."{lo}" = {rt}."{ro}"')

    sql = f"SELECT {', '.join(select_parts)} FROM {safe_anchor} {' '.join(join_parts)}"
    try:
        result = duck.query(sql)
        print(f"[duckdb] DuckDB JOIN: {len(result):,} satır")
        return result
    except Exception as e:
        print(f"[duckdb] JOIN hatası, pandas merge'e fallback: {e}")
        return _merge_dataframes(dfs, merge_ops, anchor)