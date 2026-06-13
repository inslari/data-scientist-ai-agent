"""
Entity Resolver Service
========================
Entity-level veri kalıplarını tespit eden ve çözümleyen deterministik servis.

Üç katman:
  1. Entity kolonu tespiti (db_intelligence'dan)
  2. Sinyal toplama (DuckDB ile matematik)
  3. Karar + aggregation (DuckDB GROUP BY)

LLM KULLANMAZ — tüm kararlar matematiksel.

Üç senaryo:
  Senaryo 1 (ING churn): entity_ratio > 1.5, target sabit → AGGREGATE
  Senaryo 2 (fraud):     entity_ratio > 1.5, target değişken → ENTITY_SPLIT
  Senaryo 3 (normal):    entity_ratio ≈ 1 → PASSTHROUGH
"""

from __future__ import annotations

import uuid
import tempfile
from pathlib import Path
from typing import Any

import duckdb

from src.schemas.entity_resolution_schemas import EntitySignals, EntityResolution

# ── Sabitler ────────────────────────────────────────────────
ENTITY_RATIO_THRESHOLD = 1.5   # Bu eşiğin üstünde entity-level kalıp var
PARQUET_DIR = Path(tempfile.gettempdir()) / "ds_agent_data"
PARQUET_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════
#  KATMAN 1 — Entity Kolonu Tespiti
# ═══════════════════════════════════════════════════════════

def _match_target_column(target_col: str, parquet_cols: list[str]) -> str | None:
    """
    Target kolonunu parquet kolonları içinde bulur.
    Önce tam eşleşme, yoksa suffix'li versiyonu ('churn' -> 'churn_referance_data_1_').
    Bulunamazsa None.
    """
    if not target_col:
        return None
    if target_col in parquet_cols:
        return target_col
    # suffix'li ara: 'churn' ile başlayıp '_' ile devam eden
    matches = [c for c in parquet_cols if c == target_col or c.startswith(target_col + "_")]
    return matches[0] if matches else None



def _detect_entity_column(
    db_intelligence: dict,
    target_col: str,
    parquet_path: str,
) -> str | None:
    """
    db_intelligence + parquet profiling'den entity kolonunu tespit eder.

    Strateji (öncelik sırasıyla):
      1. semantic_type == "identifier" olan kolonlar arasından
         unique_rate < 0.5 olanı seç (çok tekrar eden ID = entity)
      2. _id suffix'li kolonlar arasından en düşük unique_rate
      3. Hiçbiri yoksa → None (entity-level kalıp yok)

    Target kolonu ve pure-unique ID'ler (primary key) devre dışı bırakılır.
    """
    columns = db_intelligence.get("columns", [])

    # 1. Parquet'ten gerçek unique_rate hesapla (db_intelligence'daki sample-based olabilir)
    conn = duckdb.connect(":memory:")
    try:
        conn.execute(f"CREATE VIEW data AS SELECT * FROM read_parquet('{parquet_path}')")
        total_rows = conn.execute("SELECT COUNT(*) FROM data").fetchone()[0]

        if total_rows <= 1:
            return None

        # Aday kolonları topla
        candidates: list[dict[str, Any]] = []

        # semantic_type == "identifier" olanları öncelikle kontrol et
        identifier_cols = [
            c["column_name"] for c in columns
            if c.get("semantic_type") == "identifier"
            and c.get("column_name") != target_col
        ]

        # _id suffix'li kolonları da ekle
        parquet_cols = [
            row[0] for row in
            conn.execute("DESCRIBE data").fetchall()
        ]
        id_suffix_cols = [
            c for c in parquet_cols
            if (c.lower().endswith("_id") or c.lower() == "id")
            and c != target_col
        ]

        all_candidate_names = list(set(identifier_cols + id_suffix_cols))

        for col_name in all_candidate_names:
            if col_name == target_col:
                continue
            # Parquet'te var mı?
            if col_name not in parquet_cols:
                continue

            try:
                result = conn.execute(
                    f'SELECT COUNT(DISTINCT "{col_name}") FROM data'
                ).fetchone()
                unique_count = result[0] if result else total_rows
                unique_rate = unique_count / max(total_rows, 1)

                # Pure unique (primary key) → entity değil, atla
                if unique_rate > 0.95:
                    continue

                # Tek değerli kolon → entity değil, atla
                if unique_count <= 1:
                    continue

                candidates.append({
                    "column_name": col_name,
                    "unique_count": unique_count,
                    "unique_rate": unique_rate,
                    "is_semantic_identifier": col_name in identifier_cols,
                    "entity_ratio": total_rows / unique_count,
                })
            except Exception:
                continue

        if not candidates:
            return None

        # Sıralama: semantic identifier önce, sonra en yüksek entity_ratio
        candidates.sort(
            key=lambda c: (
                c["is_semantic_identifier"],  # True önce
                c["entity_ratio"],            # Yüksek ratio önce
            ),
            reverse=True,
        )

        best = candidates[0]
        candidate_summary = [
            (c["column_name"], f"ratio={c['entity_ratio']:.1f}")
            for c in candidates
        ]
        print(f"[entity_resolver] Entity kolon adayları: {candidate_summary}")
        print(f"[entity_resolver] Seçilen entity kolonu: '{best['column_name']}' "
              f"(unique_rate={best['unique_rate']:.4f}, entity_ratio={best['entity_ratio']:.1f})")

        return best["column_name"]

    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════
#  KATMAN 2 — Sinyal Toplama (DuckDB ile matematik)
# ═══════════════════════════════════════════════════════════

def compute_entity_signals(
    parquet_path: str,
    target_col: str,
    db_intelligence: dict,
) -> EntitySignals | None:
    """
    3 deterministik sinyal hesaplar:

      sinyal_1: entity_ratio = total_rows / unique_entity_count
      sinyal_2: target_stable = groupby(entity).target.nunique().max() == 1
      sinyal_3: has_temporal = DATE/TIMESTAMP tipi kolon var mı

    Entity kolonu bulunamazsa None döner (= passthrough).
    """
    # Entity kolonu tespit et
    entity_col = _detect_entity_column(db_intelligence, target_col, parquet_path)
    if not entity_col:
        print("[entity_resolver] Entity kolonu bulunamadı → passthrough")
        return None

    conn = duckdb.connect(":memory:")
    try:
        conn.execute(f"CREATE VIEW data AS SELECT * FROM read_parquet('{parquet_path}')")

        # ── Sinyal 1: Entity Ratio ──
        stats = conn.execute(f"""
            SELECT
                COUNT(*) as total_rows,
                COUNT(DISTINCT "{entity_col}") as unique_entities
            FROM data
        """).fetchone()

        total_rows = stats[0]
        unique_entities = stats[1]
        entity_ratio = total_rows / max(unique_entities, 1)

        print(f"[entity_resolver] Sinyal 1 — entity_ratio: "
              f"{total_rows:,} satır / {unique_entities:,} entity = {entity_ratio:.2f}")

        # ── Sinyal 2: Target Stability ──
        target_stable = False
        parquet_cols = [row[0] for row in conn.execute("DESCRIBE data").fetchall()]

        # Target kolonunu bul: önce tam eşleşme, yoksa suffix'li versiyon
        # (entity_resolver join'den sonra çalışır, kolon adı 'churn_referance_data_1_' olabilir)
        actual_target_col = _match_target_column(target_col, parquet_cols)

        if actual_target_col:
            if actual_target_col != target_col:
                print(f"[entity_resolver] Target '{target_col}' → "
                      f"'{actual_target_col}' (suffix eşleşmesi)")

            target_nunique = conn.execute(f"""
                SELECT MAX(target_nunique) FROM (
                    SELECT COUNT(DISTINCT "{actual_target_col}") as target_nunique
                    FROM data
                    WHERE "{actual_target_col}" IS NOT NULL
                    GROUP BY "{entity_col}"
                )
            """).fetchone()

            max_nunique = target_nunique[0] if target_nunique and target_nunique[0] else 1
            target_stable = (max_nunique == 1)

            print(f"[entity_resolver] Sinyal 2 — target_stable: {target_stable} "
                  f"(max target nunique per entity = {max_nunique})")
        else:
            print(f"[entity_resolver] Sinyal 2 — target '{target_col}' parquet'te bulunamadı "
                  f"(kolonlar: {parquet_cols}), target_stable=False varsayılıyor")

        # ── Sinyal 3: Temporal Kolon ──
        describe_df = conn.execute("DESCRIBE data").fetchall()
        temporal_columns = [
            row[0] for row in describe_df
            if any(t in row[1].upper() for t in ("DATE", "TIMESTAMP", "TIME"))
            and row[0] != actual_target_col
        ]
        has_temporal = len(temporal_columns) > 0

        print(f"[entity_resolver] Sinyal 3 — has_temporal: {has_temporal} "
              f"({temporal_columns})")

        return EntitySignals(
            entity_column=entity_col,
            entity_ratio=round(entity_ratio, 2),
            total_rows=total_rows,
            unique_entities=unique_entities,
            target_stable=target_stable,
            has_temporal=has_temporal,
            temporal_columns=temporal_columns,
        )

    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════
#  KATMAN 3A — Karar Mantığı
# ═══════════════════════════════════════════════════════════

def decide_resolution(signals: EntitySignals | None) -> str:
    """
    Deterministik karar ağacı:

      IF signals is None OR entity_ratio <= threshold:
          → "passthrough"
      ELIF entity_ratio > threshold AND target_stable:
          → "aggregate"
      ELIF entity_ratio > threshold AND NOT target_stable:
          → "entity_split"
    """
    if signals is None:
        return "passthrough"

    if signals.entity_ratio <= ENTITY_RATIO_THRESHOLD:
        print(f"[entity_resolver] KARAR: passthrough "
              f"(entity_ratio={signals.entity_ratio:.2f} <= {ENTITY_RATIO_THRESHOLD})")
        return "passthrough"

    if signals.target_stable:
        print(f"[entity_resolver] KARAR: aggregate "
              f"(entity_ratio={signals.entity_ratio:.2f}, target_stable=True)")
        return "aggregate"
    else:
        print(f"[entity_resolver] KARAR: entity_split "
              f"(entity_ratio={signals.entity_ratio:.2f}, target_stable=False)")
        return "entity_split"


# ═══════════════════════════════════════════════════════════
#  KATMAN 3B — Aggregation Engine (DuckDB GROUP BY)
# ═══════════════════════════════════════════════════════════

def _classify_column(col_name: str, col_type: str, target_col: str, entity_col: str) -> str:
    """Kolon tipini sınıflandır: 'numeric', 'categorical', 'temporal', 'skip'."""
    if col_name == entity_col:
        return "skip"

    # Hedef ve hedefle birebir aynı köke sahip kolonları (churn, churn_binary vb.) atla
    stem = target_col.replace("_binary", "").replace("_label", "").replace("_target", "")
    col_stem = col_name.replace("_binary", "").replace("_label", "").replace("_target", "")
    if col_stem == stem:
        return "skip"

    upper_type = col_type.upper()

    if any(t in upper_type for t in ("DATE", "TIMESTAMP", "TIME")):
        return "temporal"
    if any(t in upper_type for t in ("INT", "FLOAT", "DOUBLE", "DECIMAL", "REAL", "BIGINT", "HUGEINT", "NUMERIC")):
        return "numeric"

    return "categorical"


def aggregate_entity_data(
    parquet_path: str,
    entity_col: str,
    target_col: str,
) -> tuple[str, dict]:
    """
    DuckDB GROUP BY ile tip-bazlı aggregation.

    Stratejiler:
      Sayısal  → mean, std, sum
      Kategorik → mode, nunique
      Temporal  → min, max, range_days
      Target   → FIRST (entity-sabit), temiz isimle yazılır
      Genel    → record_count
    """
    target_col_clean = target_col  # downstream'in beklediği temiz isim ("churn")

    conn = duckdb.connect(":memory:")
    try:
        conn.execute(f"CREATE VIEW data AS SELECT * FROM read_parquet('{parquet_path}')")

        describe = conn.execute("DESCRIBE data").fetchall()
        col_types = {row[0]: row[1] for row in describe}

        # Target'ın gerçek parquet adını ÖNCE bul (suffix'li olabilir: churn_referance_data_1_)
        actual_target_col = _match_target_column(target_col_clean, list(col_types.keys()))
        if actual_target_col:
            print(f"[entity_resolver] Aggregation target: '{actual_target_col}' "
                  f"→ '{target_col_clean}' olarak yazılacak")
        else:
            print(f"[entity_resolver] UYARI: target '{target_col_clean}' "
                  f"aggregation'da bulunamadı! Kolonlar: {list(col_types.keys())}")

        # Aggregation SQL parçaları
        select_parts: list[str] = [f'"{entity_col}"']
        features_created = 0

        # Target → FIRST, gerçek kolondan oku ama TEMİZ isimle yaz
        if actual_target_col:
            select_parts.append(f'FIRST("{actual_target_col}") AS "{target_col_clean}"')

        # Entity başına kayıt sayısı — tek seferlik
        select_parts.append('COUNT(*) AS "record_count"')
        features_created += 1

        for col_name, col_type in col_types.items():
            # Target kolonunu (suffix'li gerçek ismiyle) skip et — yukarıda zaten eklendi
            category = _classify_column(col_name, col_type, actual_target_col or target_col_clean, entity_col)

            if category == "skip":
                continue

            safe_col = f'"{col_name}"'

            if category == "numeric":
                select_parts.extend([
                    f'AVG({safe_col}) AS "{col_name}_mean"',
                    f'STDDEV({safe_col}) AS "{col_name}_std"',
                    f'SUM({safe_col}) AS "{col_name}_sum"',
                ])
                features_created += 3

            elif category == "categorical":
                select_parts.extend([
                    f'MODE({safe_col}) AS "{col_name}_mode"',
                    f'COUNT(DISTINCT {safe_col}) AS "{col_name}_nunique"',
                ])
                features_created += 2

            elif category == "temporal":
                select_parts.extend([
                    f'MIN({safe_col}) AS "{col_name}_min"',
                    f'MAX({safe_col}) AS "{col_name}_max"',
                    f'DATEDIFF(\'day\', MIN({safe_col}), MAX({safe_col})) AS "{col_name}_range_days"',
                ])
                features_created += 3

        select_clause = ",\n    ".join(select_parts)
        agg_sql = f"""
            SELECT
                {select_clause}
            FROM data
            GROUP BY "{entity_col}"
        """

        print(f"[entity_resolver] Aggregation SQL (ilk 500 char):\n{agg_sql[:500]}")

        result_df = conn.execute(agg_sql).df()

        # Target gerçekten çıktıda mı? (güvenlik kontrolü)
        if target_col_clean not in result_df.columns:
            print(f"[entity_resolver] KRİTİK UYARI: target '{target_col_clean}' "
                  f"aggregate çıktısında YOK! Kolonlar: {list(result_df.columns)}")
        else:
            n_classes = result_df[target_col_clean].nunique(dropna=True)
            print(f"[entity_resolver] Target '{target_col_clean}' çıktıda OK "
                  f"({n_classes} benzersiz değer)")

        output_name = f"entity_agg_{uuid.uuid4().hex[:8]}"
        output_path = PARQUET_DIR / f"{output_name}.parquet"
        conn.register("result_view", result_df)
        conn.execute(f"COPY result_view TO '{output_path}' (FORMAT PARQUET)")

        original_rows = conn.execute("SELECT COUNT(*) FROM data").fetchone()[0]

        stats = {
            "original_rows": original_rows,
            "aggregated_rows": len(result_df),
            "features_created": features_created,
            "total_columns": len(result_df.columns),
            "entity_column": entity_col,
            "compression_ratio": round(original_rows / max(len(result_df), 1), 2),
        }

        print(f"[entity_resolver] Aggregation tamamlandı: "
              f"{original_rows:,} → {len(result_df):,} satır "
              f"({features_created} yeni feature, {len(result_df.columns)} kolon)")

        return str(output_path), stats

    finally:
        conn.close()

# ═══════════════════════════════════════════════════════════
#  ANA FONKSİYON — Node tarafından çağrılır
# ═══════════════════════════════════════════════════════════

def resolve_entities(state: dict) -> dict:
    """
    Entity resolution ana fonksiyonu.

    Girdiler (state'ten):
      - data_path: parquet dosya yolu
      - target_definition: {column_name, ...}
      - db_intelligence: {columns, tables, relationships, ...}

    Çıktılar:
      - entity_resolution: EntityResolution dict
      - data_path: güncellenmiş parquet path (aggregate ise yeni, değilse aynı)
    """
    print(f"\n{'='*60}")
    print(f"[entity_resolver] BAŞLIYOR — Entity Resolution")
    print(f"{'='*60}")

    data_path = state.get("data_path", "")
    target_def = state.get("target_definition") or {}
    db_intelligence = state.get("db_intelligence") or {}
    target_col = target_def.get("column_name", "")

    if not data_path:
        print("[entity_resolver] data_path boş, atlaniyor")
        return {
            "entity_resolution": EntityResolution(
                action="passthrough",
                signals=EntitySignals(
                    entity_column="",
                    entity_ratio=1.0,
                    total_rows=0,
                    unique_entities=0,
                    target_stable=False,
                    has_temporal=False,
                    temporal_columns=[],
                ),
                message="Data path bulunamadı, entity resolution atlandı.",
            ).model_dump(),
            "data_path": data_path,
        }

    # 1. Sinyalleri hesapla
    signals = compute_entity_signals(data_path, target_col, db_intelligence)

    # 2. Karar ver
    action = decide_resolution(signals)

    # 3. Aksiyon uygula
    new_data_path = data_path
    agg_stats: dict = {}
    aggregated_path = ""

    if action == "aggregate" and signals:
        aggregated_path, agg_stats = aggregate_entity_data(
            parquet_path=data_path,
            entity_col=signals.entity_column,
            target_col=target_col,
        )
        new_data_path = aggregated_path

    # 4. Mesaj oluştur
    if signals is None:
        message = "Entity kolonu bulunamadı — veri olduğu gibi kullanılacak."
        signals = EntitySignals(
            entity_column="",
            entity_ratio=1.0,
            total_rows=0,
            unique_entities=0,
            target_stable=False,
            has_temporal=False,
            temporal_columns=[],
        )
    elif action == "aggregate":
        message = (
            f"Entity Resolution: {signals.total_rows:,} satır → "
            f"{agg_stats.get('aggregated_rows', signals.unique_entities):,} "
            f"{signals.entity_column} bazında aggregate edildi. "
            f"{agg_stats.get('features_created', 0)} yeni feature oluşturuldu."
        )
    elif action == "entity_split":
        message = (
            f"Entity Resolution: {signals.entity_column} başına "
            f"{signals.entity_ratio:.1f} satır tespit edildi. "
            f"Target satır bazında değişiyor — aggregation YAPILMADI, "
            f"entity-based split önerilecek."
        )
    else:
        message = (
            f"Entity Resolution: {signals.entity_column or 'N/A'} başına "
            f"{signals.entity_ratio:.1f} satır — entity-level kalıp yok, "
            f"veri olduğu gibi kullanılacak."
        )

    print(f"[entity_resolver] SONUÇ: {message}")

    resolution = EntityResolution(
        action=action,
        signals=signals,
        aggregated_data_path=aggregated_path,
        aggregation_stats=agg_stats,
        message=message,
    )

    return {
        "entity_resolution": resolution.model_dump(),
        "data_path": new_data_path,
    }
