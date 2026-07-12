"""
Data Quality / Validation Layer — Katman 2 (Anlamsal)
======================================================
Yükleme sonrası, modelleme öncesi deterministik veri kalitesi kapısı.

Felsefe (hibrit): TAMAMEN İSTATİSTİKSEL/DETERMİNİSTİK — LLM'e SORMAZ (halüsinasyon
riski yok). Tüm ağır işler DuckDB in-engine (parquet üzerinde full-scan zaten
`execution_node`'da yapıldı); burada yalnız profilden türetme + gerektiğinde birkaç
hedefe yönelik TRY_CAST sorgusu. Veri PANDAS'A YÜKLENMEZ (donanım dostu).

Dinamiklik: hiçbir kolon adı hard-code EDİLMEZ. Sinyaller ad-morfolojisi (prefix
is_/has_/flag_…) + istatistik (nunique, distinct_ratio, TRY_CAST parse oranı).

Tek tutarlı katman, tek durum:
    data_quality_status : "ok" | "degraded" | "failed"
    data_quality_report : {structural, semantic, target_health, quarantine, autofixes, notes, decision}

Karar politikası:
  - Kesin reader-misconfig (sayı-ama-string) → not/öneri (degraded, durdurmaz).
  - Belirsiz bozulma (alan-sayısı kayması) → karantina + DUR (tahmin etme).
  - Target rol-içerik çelişkisi (ad binary ima ediyor ama sürekli) → DUR.
  - Target bütünlüğü kutsal: target varken yapısal karantina → asla degraded, direkt failed.
"""

from __future__ import annotations

from typing import Any

from src.core.config import settings
from src.core.logging_config import get_logger

logger = get_logger(__name__)

# Ad-morfolojisi: bir kolon adının ikili (boolean/flag) rol ima edip etmediği.
# Domain/dil bağımsız olacak şekilde yalın İngilizce boolean morfolojisi + yaygın
# son ekler. (Değerler ayrıca istatistikle teyit edilir; ad tek başına karar vermez.)
_BINARY_PREFIXES = ("is_", "has_", "was_", "can_", "should_", "flag_", "isnt_", "does_")
_BINARY_SUFFIXES = ("_flag", "_ind", "_bool", "_indicator", "_yn", "_is")


def _name_implies_binary(col: str) -> bool:
    c = col.lower().strip()
    # Bare "flag" da morfolojik olarak ikili ima eder (prefix/suffix'e uymaz).
    return c == "flag" or c.startswith(_BINARY_PREFIXES) or c.endswith(_BINARY_SUFFIXES)


def _resolve_target(state: dict) -> str | None:
    """Hedef kolonu dinamik çözer: önce UI manuel seçim, sonra keşif kararı."""
    tc = (state.get("target_col") or "").strip()
    if tc:
        return tc
    td = state.get("target_definition") or {}
    name = (td.get("column_name") or "").strip()
    return name or None


def _is_numeric_dtype(dtype: str) -> bool:
    d = (dtype or "").upper()
    return any(t in d for t in ("INT", "FLOAT", "DOUBLE", "DECIMAL", "REAL", "NUMERIC", "HUGEINT"))


def _numeric_parse_rates(data_path: str, varchar_cols: list[str]) -> dict[str, float]:
    """VARCHAR kolonlar için 'sayısal-parse edilebilir mi' oranını DuckDB in-engine
    hesaplar: AVG(TRY_CAST(col AS DOUBLE) IS NOT NULL) — non-null üzerinde.

    Tek sorguda tüm kolonlar (ucuz, tam veri okunmaz — columnar aggregate).
    Hata / dosya yoksa boş döner (profile-only moda düşer)."""
    if not data_path or not varchar_cols:
        return {}
    try:
        import duckdb
        exprs = []
        for c in varchar_cols:
            safe = c.replace('"', '""')
            # non-null payda; hepsi null ise 0 döndür (NULLIF ile bölme koruması)
            exprs.append(
                f'COALESCE(AVG(CASE WHEN "{safe}" IS NOT NULL '
                f'THEN (TRY_CAST("{safe}" AS DOUBLE) IS NOT NULL)::INT END), 0) AS "{safe}"'
            )
        con = duckdb.connect()
        pq = f"read_parquet('{data_path}')"
        row = con.execute(f"SELECT {', '.join(exprs)} FROM {pq}").fetchone()
        con.close()
        return {c: float(v or 0.0) for c, v in zip(varchar_cols, row)}
    except Exception as e:
        logger.warning(f"[data_quality] numeric-parse in-engine kontrolü atlandı: {e}")
        return {}


def run_data_quality(state: dict) -> dict:
    """Katman 2'yi çalıştırır; data_quality_status + data_quality_report üretir.

    Katman 1a (yapısal) çıktısı state['data_quality_report']['structural'] içinde
    hazır gelir; burada onunla birleştirilip nihai karar verilir.
    """
    profiling = state.get("profiling_report") or {}
    columns: dict[str, dict] = profiling.get("columns", {}) or {}
    shape = profiling.get("shape", {}) or {}
    total_rows = int(shape.get("rows", 0) or 0)
    data_path = state.get("data_path") or ""
    intent = state.get("intent", "")

    # Katman 1a yapısal özeti (execution_node tohumladı)
    seed = state.get("data_quality_report") or {}
    structural = seed.get("structural") or {}
    q_rows = int(structural.get("quarantined_rows", 0) or 0)
    q_total = int(structural.get("total_rows", 0) or 0)
    mismatch_rate = float(structural.get("mismatch_rate", 0.0) or 0.0)

    report: dict[str, Any] = {
        "structural": structural,
        "semantic": {"notes": []},
        "target_health": {},
        "autofixes": [],
        "notes": [],
    }
    notes: list[str] = report["notes"]

    target_col = _resolve_target(state)
    has_target = bool(target_col)

    # ── Anlamsal: sayı-ama-string (kesin reader-misconfig) ──────────────────
    # VARCHAR yüklenmiş ama neredeyse tamamı sayısal-parse edilebilen kolonlar.
    varchar_cols = [c for c, info in columns.items()
                    if not _is_numeric_dtype(info.get("dtype", "")) and "error" not in info]
    parse_rates = _numeric_parse_rates(data_path, varchar_cols)
    for col, rate in parse_rates.items():
        if rate >= settings.DQ_NUMERIC_STRING_RATE:
            report["semantic"]["notes"].append({
                "column": col, "issue": "numeric_loaded_as_string",
                "numeric_parse_rate": round(rate, 4),
            })
            notes.append(f"'{col}': değerlerin %{rate*100:.0f}'i sayısal ama metin (VARCHAR) yüklenmiş.")

    # ── Anlamsal: rol-içerik (ad binary ima ediyor ama sürekli) ─────────────
    # Normal FEATURE'larda yalnız NOT — asla durdurmaz (meşru sürekli olabilir:
    # flag_score, is_amount_ratio…). Sert-durdurma SADECE seçili target'ta olur.
    for col, info in columns.items():
        if col == target_col or "error" in info:
            continue
        nunique = int(info.get("unique_count", 0) or 0)
        ratio = float(info.get("unique_rate", 0.0) or 0.0)
        if (_name_implies_binary(col) and _is_numeric_dtype(info.get("dtype", ""))
                and nunique > settings.DQ_TARGET_MAX_CARD and ratio > settings.DQ_CONTINUOUS_RATIO):
            report["semantic"]["notes"].append({
                "column": col, "issue": "binary_name_continuous_values",
                "nunique": nunique, "distinct_ratio": round(ratio, 4), "severity": "note",
            })
            notes.append(f"'{col}': adı ikili ima ediyor ama {nunique} farklı sürekli değer içeriyor (yalnız uyarı).")

    # ── Target sağlık kontrolü (tek tutarlı katman — ayrı parçalama yok) ──────
    target_conflict = False
    if has_target:
        tinfo = columns.get(target_col, {})
        nunique = int(tinfo.get("unique_count", 0) or 0)
        null_cnt = int(tinfo.get("null_count", 0) or 0)
        non_null = max(total_rows - null_cnt, 0)
        ratio = float(tinfo.get("unique_rate", 0.0) or 0.0)
        dtype = tinfo.get("dtype", "")
        implies_binary = _name_implies_binary(target_col)

        th = {
            "target": target_col, "nunique": nunique, "non_null": non_null,
            "distinct_ratio": round(ratio, 4), "dtype": dtype,
            "name_implies_binary": implies_binary, "intent": intent,
        }

        # nunique BİRİNCİL sinyal: ==2 → KESİN binary, asla 'continuous' deme.
        if nunique == settings.DQ_BINARY_NUNIQUE:
            th["verdict"] = "binary_ok"
        elif non_null < settings.DQ_MIN_ROWS_TARGET_CHECK:
            # Küçük-n koruması: sürekli-bozulma verdiği için kanıt yetersiz.
            th["verdict"] = "insufficient_rows"
        else:
            # Sürekli-şüphe: yüksek kardinalite (birincil) + distinct_ratio (destekleyici).
            continuous_suspect = (
                _is_numeric_dtype(dtype)
                and nunique > settings.DQ_TARGET_MAX_CARD
                and ratio > settings.DQ_CONTINUOUS_RATIO
            )
            if implies_binary and continuous_suspect:
                target_conflict = True
                th["verdict"] = "role_content_conflict"
            else:
                th["verdict"] = "ok"
        report["target_health"] = th

    # ── Nihai karar ──────────────────────────────────────────────────────────
    status = "ok"
    reason = ""
    if target_conflict:
        status, reason = "failed", "target_role_content_conflict"
    elif mismatch_rate > settings.DQ_STRUCT_FAIL_RATE:
        status, reason = "failed", "structural_mismatch_high"
    elif q_rows > 0 and has_target:
        # Target bütünlüğü kutsal: kayma satırın tüm sağ-kolonlarını bozar,
        # etiket hizası kanıtlanamaz → asla degraded, direkt failed.
        status, reason = "failed", "structural_mismatch_target"
    elif q_rows > 0:
        # Target yok + izole karantina → gerçek veri kaybı ama modellenebilir kalan var.
        status, reason = "degraded", "structural_quarantine_isolated"
    # NOT: anlamsal notlar (sayı-ama-string, binary-adlı sürekli FEATURE) yalnız
    # BİLGİLENDİRİCİDİR — status'u DÜŞÜRMEZ (false-positive durdurma yok). Meşru
    # sürekli bir feature (flag_score, is_amount_ratio) pipeline'ı bozmaz; rapora
    # not olarak taşınır. degraded YALNIZ gerçek veri kaybında (karantina) verilir.

    report["quarantine"] = {"rows": q_rows, "total": q_total, "mismatch_rate": mismatch_rate}
    report["decision"] = {"status": status, "reason": reason, "has_target": has_target}

    logger.info(
        f"[data_quality] status={status} reason='{reason or '-'}' "
        f"target={target_col or '-'} "
        f"target_verdict={report.get('target_health', {}).get('verdict', '-')} "
        f"quarantine={q_rows}/{q_total} (mismatch={mismatch_rate:.4f}) "
        f"notes={len(report['semantic']['notes'])}"
    )

    return {"data_quality_status": status, "data_quality_report": report}
