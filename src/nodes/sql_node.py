from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from src.core.llm import get_llm
from src.core.config import settings
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, Field

from ..state import AgentState
from ..core.db_provider import db_manager

# ---------------------------------------------------------------------------
# MCP bağlantı ayarları
# ---------------------------------------------------------------------------
MCP_HOST = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
MCP_PORT = os.getenv("MCP_SERVER_PORT", "9001")
MCP_PATH = os.getenv("MCP_SERVER_PATH", "/mcp")
MCP_URL  = f"http://{MCP_HOST}:{MCP_PORT}{MCP_PATH}"


# ---------------------------------------------------------------------------
# Pydantic şemaları
# ---------------------------------------------------------------------------

class SQLGenerationResult(BaseModel):
    sql: str = Field(description="Çalıştırılacak SELECT sorgusu")
    db_name: str = Field(description="Sorgunun çalıştırılacağı veritabanı adı")
    reasoning: str = Field(description="Sorgunun neden bu şekilde yazıldığını açıklayan tek cümle (Türkçe)")
    data_warning: str = Field(
        default="",
        description="Hiç uygun kolon yoksa buraya Türkçe uyarı yaz. Aksi hâlde boş bırak."
    )
    is_approximation: bool = Field(
        default=False,
        description="True: kullanıcının istedigi kolon tam yoktu, en yakın alternatif kullanıldı. False: birebir eşleşme var."
    )
    approximation_warning: str = Field(
        default="",
        description="is_approximation=True ise: tam istenen verinin neden olmadığını ve hangi alternativin kullanıldığını açıklayan Türkçe cümle."
    )


class ChartConfig(BaseModel):
    chart_type: str = Field(
        description="bar | line | pie | scatter | table — veriye en uygun grafik tipi"
    )
    x_column: str = Field(description="X ekseninde kullanılacak kolon adı")
    y_column: str = Field(description="Y ekseninde kullanılacak kolon adı (yoksa boş string)")
    title: str = Field(description="Grafik başlığı (Türkçe, max 60 karakter)")
    summary: str = Field(description="Sonuçları 1-2 cümleyle anlatan Türkçe özet")


# ---------------------------------------------------------------------------
# Prompt — SQL üretimi (db_intelligence ile zenginleştirilmiş)
# ---------------------------------------------------------------------------

SQL_GEN_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Sen bir SQL veritabanı uzmanısın. Kullanılabilir veritabanları: {db_providers}.
Kullanıcının sorusunu anlayıp ona uygun bir SELECT sorgusu ve bu sorgunun çalıştırılacağı veritabanı adını (db_name) üreteceksin.

Aşağıda veritabanının ham şeması ve her kolon için semantik profil bilgisi verilmiştir.

=== ZORUNLU: Her sorguda şu 3 adımı sırayla uygula ===

ADIM 1 — Kullanıcının tam olarak ne istediğini belirle.
  Kavramın hangi veri tipinde ve hangi birimde olması gerektiğini netleştir.
  Örnek: "kazanç" → sayısal, para birimi | "yaş" → sayısal, yıl | "şehir" → metin

ADIM 2 — Şemada bu kavramı GERÇEKTEN karşılayan bir kolon var mı?
  Sağlanan {schema} ve {column_profiles} üzerinden her kolonu değerlendir.
  Bir kolon ANCAK aşağıdakilerin TÜMÜ sağlanırsa uygun sayılır:
    a) Kolon adı veya açıklaması, kullanıcının istediği kavramla anlamsal olarak DOĞRUDAN eşleşiyor
    b) Kolonun gerçek değerleri (min, max, avg, örnek değerler) kullanıcının beklediği
       anlam ve birimiyle TUTARLI (örn. "kazanç" için gerçek sayısal para değeri beklenir)
    c) Veri tipi uygun (sayısal kavram için sayısal kolon, metin kavram için text kolon)

  GENEL YANILTICI DURUMLAR — bunlar UYGUN KOLON DEĞİLDİR:
    - Birim uyuşmazlığı: Kullanıcı parasal değer istiyorsa saat/adet/oran içeren kolon UYGUN DEĞİL
    - Kategori vs sayı: Kullanıcı gerçek sayısal değer istiyorsa etiket/aralık içeren kolon UYGUN DEĞİL
      (örn. "düşük/orta/yüksek", "<X", ">Y" gibi gruplar sayısal değer değildir)
    - İsim tuzağı: Kolon adında kavramla ilgili bir kelime geçmesi YETMEZ,
      gerçek içerik de eşleşmeli
    - Proxy kolonu: "En yakın" veya "en ilgili" kolonu seçme — doğrudan eşleşme yoksa UYGUN DEĞİL

ADIM 3 — Üç durumdan birini seç:

  [DURUM A] Tam eşleşme var:
    -> Normal SELECT sorgusu yaz
    -> is_approximation = false, approximation_warning = "", data_warning = ""

  [DURUM B] Tam eşleşme yok ama anlamlı bir alternatif yapılabilir:
    -> Mevcut kolonlarla konuyu dolaylı yoldan cevaplayan sorguyu yaz
       (oran, düzüm, kategori ayrımı gibi)
    -> is_approximation = true
    -> approximation_warning: "Bu veri setinde [istenen kavram] için doğrudan kolon yok.
       Bunun yerine [kullanılan kolon/yaklaşım] kullanılarak [ne hesaplandı] gösterilmektedir."
    -> data_warning = ""

  [DURUM C] Hiçbir anlamlı alternatif yok:
    -> sql = SELECT 'NO_VALID_COLUMN' AS warning
    -> data_warning = "Bu veri setinde [kavram] için uygun kolon bulunamadı.
                       Mevcut kolonlar: [...]. Alternatif: [yapılabilecek sorgu önerisi]"
    -> is_approximation = false, approximation_warning = ""

=== Genel Kurallar ===
- Yalnizca SELECT sorgusu yaz
- Aggregation varsa GROUP BY kullan, raw veri istenmisse LIMIT 500 ekle
- Kolon adlarini semadan birebir kullan (buyuk/kucuk harf dahil)
- Nokta, bosluk veya ozel karakter iceren kolon adlarini MUTLAKA cift tirnak ile sar
- Eger soru belirsizse en mantikli yorumu yap ve reasoning alaninda acikla
- Turkce soru gelirse ayni mantikla Ingilizce SQL yaz

--- Ham Sema ---
{schema}

--- Kolon Semantik Profilleri ---
{column_profiles}
""",
    ),
    ("human", "Kullanici sorusu: {user_query}"),
])



# Prompt — Chart konfigürasyonu

CHART_CFG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Sen bir veri görselleştirme uzmanısın. SQL sonuçlarına bakarak en anlamlı grafik tipini seç.

Grafik tipi rehberi:
- Kategorik sütun + sayısal sütun → bar
- Zaman serisi / sıralı veri + sayısal → line
- Oran / pay dağılımı (≤8 kategori) → pie
- İki sayısal sütun arası ilişki → scatter
- Çok kolon / ham tablo → table

Kolon listesi: {columns}
İlk 5 satır: {sample_rows}
Kullanıcı sorusu: {user_query}
""",
    ),
    ("human", "Grafik konfigürasyonunu döndür."),
])


# db_intelligence → kolon profil metni

def _build_column_profiles(db_intelligence: dict) -> str:
    """
    state['db_intelligence'] dict'inden kolon semantik profillerini
    LLM'in okuyabileceği düz metin formatına çevirir.
    """
    columns = db_intelligence.get("columns", [])
    if not columns:
        return "Kolon profili mevcut değil."

    lines = []
    for col in columns:
        table       = col.get("table_name", "?")
        name        = col.get("column_name", "?")
        dtype       = col.get("data_type", "?")
        sem_type    = col.get("semantic_type", "?")
        importance  = col.get("importance_score", 0)
        nullability = col.get("nullability", "?")
        uniqueness  = col.get("uniqueness", "?")
        description = col.get("description", "")
        sample_vals = col.get("sample_values", "")

        line = (
            f'  - {table}."{name}" | type={dtype} | semantic={sem_type} '
            f'| importance={importance:.2f} | null={nullability} | unique={uniqueness}'
        )
        if description:
            line += f'\n      description: {description}'
        if sample_vals:
            line += f'\n      sample_values: {sample_vals}'
        lines.append(line)

    global_insights = db_intelligence.get("global_insights", [])
    header = "Global insights:\n" + "\n".join(f"  * {g}" for g in global_insights) + "\n\n" if global_insights else ""
    return header + "Columns:\n" + "\n".join(lines)



# Async MCP yardımcıları

async def _async_run_query(sql: str, db_name: str) -> tuple[list[str], list[tuple[Any, ...]]]:
    async with streamable_http_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("run_sql_query", {"query": sql, "db_name": db_name})
            raw_text = getattr(result.content[0], "text", "") if result.content else ""

    lines   = raw_text.strip().splitlines()
    columns: list[str]              = []
    rows:    list[tuple[Any, ...]]  = []

    for i, line in enumerate(lines):
        if line.startswith("Columns:"):
            columns = [c.strip() for c in line.replace("Columns:", "").split(",")]
        elif line.startswith("Results:") or line.startswith("("):
            data_lines = lines[i + 1:] if line.startswith("Results:") else lines[i:]
            for dl in data_lines:
                dl = dl.strip()
                if not dl or dl.startswith("..."):
                    continue
                try:
                    rows.append(eval(dl))
                except Exception:
                    pass
            break

    return columns, rows


def _run_async(coro):
    """Streamlit'in sync context'inden async coroutine çalıştır."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _mcp_run_query(sql: str, db_name: str) -> tuple[list[str], list[tuple[Any, ...]]]:
    return _run_async(_async_run_query(sql, db_name))


# Chart payload

def _build_chart_payload(
    columns: list[str],
    rows: list[tuple[Any, ...]],
    chart_cfg: ChartConfig,
) -> dict:
    col_index = {c: i for i, c in enumerate(columns)}
    x_col = chart_cfg.x_column
    y_col = chart_cfg.y_column

    if chart_cfg.chart_type == "table":
        return {
            "type": "table",
            "columns": columns,
            "rows": [list(r) for r in rows],
            "title": chart_cfg.title,
            "summary": chart_cfg.summary,
        }

    if x_col not in col_index:
        x_col = columns[0] if columns else ""
        y_col = columns[1] if len(columns) > 1 else ""

    labels = [str(r[col_index[x_col]]) for r in rows if x_col in col_index]

    if y_col and y_col in col_index:
        values = []
        for r in rows:
            val = r[col_index[y_col]]
            try:
                values.append(float(val))
            except (TypeError, ValueError):
                values.append(0.0)
        datasets = [{"label": y_col, "data": values}]
    else:
        datasets = []

    return {
        "type": chart_cfg.chart_type,
        "title": chart_cfg.title,
        "labels": labels,
        "datasets": datasets,
        "summary": chart_cfg.summary,
        "columns": columns,
        "rows": [list(r) for r in rows],
    }


# Main Node Sync

def sql_node(state: AgentState) -> AgentState:
    """
    SQL Query Node:
      1. db_intelligence'ı state'den okur (db_context_node zaten çalıştı)
      2. LLM ile kolon profillerini kullanarak SQL üretir
      3. MCP'de sorguyu çalıştırır
      4. LLM ile chart konfigürasyonu belirler
      5. State'e yazar
    """
    llm        = get_llm(model_name=settings.INTELLIGENCE_ENGINE_MODEL, temperature=0)
    user_query = state["user_query"]

    # db_intelligence taken from db_context_node
    db_intelligence  = state.get("db_intelligence", {})
    raw_schema       = state.get("raw_schema", "Schema not available.")
    column_profiles  = _build_column_profiles(db_intelligence)

    try:
        db_providers_info = ", ".join([f"{k} ({v.provider_display_name})" for k, v in db_manager.get_all_providers().items()])
        sql_chain  = SQL_GEN_PROMPT | llm.with_structured_output(SQLGenerationResult)
        sql_result: SQLGenerationResult = sql_chain.invoke({  # type: ignore[assignment]
            "schema":          raw_schema,
            "column_profiles": column_profiles,
            "user_query":      user_query,
            "db_providers":    db_providers_info,
        })
        sql                  = sql_result.sql.strip()
        target_db_name       = str(sql_result.db_name).strip().strip('",\'”’ \t\n')
        data_warning         = sql_result.data_warning or ""
        is_approximation     = sql_result.is_approximation
        approximation_warning = sql_result.approximation_warning or ""
    except Exception as e:
        return {**state, "error": f"SQL üretim hatası: {e}", "current_step": "sql_node"}

    # If no valid columns found, return warning
    if "NO_VALID_COLUMN" in sql:
        warning_msg = data_warning or "Bu veri setinde isteğinize uygun kolon bulunamadı."
        return {
            **state,
            "sql_query_raw":     sql,
            "sql_query_columns": [],
            "sql_query_results": [],
            "sql_query_chart": {
                "type":    "warning",
                "title":   "Uygun Kolon Bulunamadı",
                "summary": warning_msg,
            },
            "sql_query_summary": warning_msg,
            "current_step":      "sql_node",
        }
    # 4. Execute the query
    try:
        columns, rows = _mcp_run_query(sql, target_db_name)
    except Exception as e:
        return {**state, "error": f"SQL çalıştırma hatası: {e}", "current_step": "sql_node"}

    if not rows:
        return {
            **state,
            "sql_query_raw":     sql,
            "sql_query_columns": columns,
            "sql_query_results": [],
            "sql_query_chart":   {
                "type": "table", "columns": columns, "rows": [],
                "title": "Sonuç bulunamadı", "summary": "Sorgu sonuç döndürmedi.",
            },
            "sql_query_summary": "Sorgu sonuç döndürmedi.",
            "current_step":      "sql_node",
        }

    # 4. Chart configuration
    sample_rows = rows[:5]
    try:
        chart_chain = CHART_CFG_PROMPT | llm.with_structured_output(ChartConfig)
        chart_cfg: ChartConfig = chart_chain.invoke({  # type: ignore[assignment]
            "columns":     ", ".join(columns),
            "sample_rows": json.dumps([list(r) for r in sample_rows], ensure_ascii=False),
            "user_query":  user_query,
        })
    except Exception as e:
        chart_cfg = ChartConfig(
            chart_type="table",
            x_column=columns[0] if columns else "",
            y_column="",
            title="Sorgu Sonuçları",
            summary=f"Görselleştirme belirlenemedi: {e}",
        )

    # 5. Payload + state
    chart_payload    = _build_chart_payload(columns, rows, chart_cfg)
    results_as_dicts = [dict(zip(columns, row)) for row in rows]

    return {
        **state,
        "sql_query_raw":     sql,
        "sql_query_columns": columns,
        "sql_query_results": results_as_dicts,
        "sql_query_chart":   chart_payload,
        "sql_query_summary": chart_cfg.summary,
        "sql_query_warning": approximation_warning if is_approximation else "",
        "current_step":      "sql_node",
    }