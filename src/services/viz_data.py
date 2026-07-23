"""
Data Visualization veri katmanı: bir grafiğin SQL'ini çalıştırmak ve sonucu saklanabilir
hale getirmek.

Neden ayrı servis: aynı iş üç yerden çağrılıyor —
  1. viz_sql_node (grafik ilk üretilirken),
  2. `POST .../refresh` (LLM olmadan veriyi tazelerken),
  3. `POST .../refine` (doğal dille değiştirilen SQL'i çalıştırırken).

Node/service ayrımı (CLAUDE.md): yürütme mantığı burada, node ince adaptör.
"""
from __future__ import annotations

import ast

from src.core.logging_config import get_logger
logger = get_logger(__name__)

# Bir grafiğin kaydında saklanan azami satır. Kayıt JSON kolonuna yazıldığı için
# sınırsız büyümesine izin verilemez; üstü kırpılır (grafik zaten aggregate edilmiş
# veriyi çizer, ham satır yalnızca tablo görünümü / yeniden render içindir).
MAX_STORED_ROWS = 2000


def parse_query_result(raw_txt: str) -> tuple[list[str], list]:
    """
    run_data_group_query_logic'in metin çıktısını (Columns: ... / Results: ...)
    kolon listesi + ham tuple satırlarına çevirir. ttd_sql_node ile aynı ayrıştırma.
    """
    lines = raw_txt.strip().splitlines()
    columns: list[str] = []
    rows: list = []
    for i, line in enumerate(lines):
        if line.startswith("Columns:"):
            columns = [c.strip() for c in line.replace("Columns:", "").split(",")]
        elif line.startswith("Results:") or line.startswith("("):
            data_lines = lines[i + 1:] if line.startswith("Results:") else lines[i:]
            for dl in data_lines:
                dl = dl.strip()
                if not dl or dl.startswith("..."):
                    continue
                if dl.startswith("(") and dl.endswith(")"):
                    try:
                        rows.append(ast.literal_eval(dl))
                    except Exception:
                        pass
            break
    return columns, rows


async def execute_viz_sql(data_group_id: int, sql: str, user_id: int | None) -> tuple[list[str], list]:
    """
    SQL'i data group üzerinde çalıştırır → (kolonlar, ham tuple satırlar).
    Sorgu hata döndürürse Exception yükseltir (çağıran retry/hata yönetimini yapar).
    """
    # Döngüsel import'u önlemek için fonksiyon içinde (api → src → api zinciri)
    from api.core.database import SessionLocal
    from api.schemas.data_group import (
        run_data_group_query_logic as run_data_group_query,
        QueryRequest as DGQueryRequest,
    )

    db = SessionLocal()
    try:
        res = await run_data_group_query(data_group_id, DGQueryRequest(query=sql), db, user_id)
        if res.get("status") == "error":
            raise Exception(res.get("result"))
        raw_txt = str(res.get("result", "")) if res.get("status") == "success" else ""
        return parse_query_result(raw_txt)
    finally:
        db.close()


def to_storable(columns: list, rows: list) -> dict | None:
    """
    Ham satırları galeri kaydına yazılabilir JSON'a çevirir: {"columns", "rows", "truncated"}.

    Tuple'lar listeye çevrilir (JSON), satır sayısı MAX_STORED_ROWS ile sınırlanır.
    Tablo görünümü ve tema/spec değişiminde PNG'yi yeniden üretmek için kullanılır.
    """
    if not columns or not rows:
        return None
    kept = rows[:MAX_STORED_ROWS]
    return {
        "columns":   list(columns),
        "rows":      [list(r) for r in kept],
        "truncated": len(rows) > len(kept),
    }


def from_storable(data: dict | None) -> tuple[list, list]:
    """Saklanan {"columns","rows"} sözlüğünü render_matplotlib'in beklediği ikiliye açar."""
    if not isinstance(data, dict):
        return [], []
    return list(data.get("columns") or []), list(data.get("rows") or [])
