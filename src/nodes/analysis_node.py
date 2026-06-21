from src.core.nodes import node
"""
Analysis Node
=============
execution_node'dan sonra çalışır (intent == "deep_analysis" ise).

Görev:
  1. Parquet'i oku
  2. profile_table çıktısı + run_deep_analysis → korelasyon, histogram,
     hedef ilişkisi, kümeleme
  3. Sonucu state'e yaz (frontend Plotly ile çizecek)

Model EĞİTMEZ. Akış: execution → analysis_node → END

Not: Bu node modeli/leakage/feature hattına HİÇ girmez. Tamamen ayrı.
"""

import pandas as pd

from src.state import AgentState
from src.services.analysis_engine import run_deep_analysis

from src.core.logging_config import get_logger
logger = get_logger(__name__)


@node("analysis_node")
def analysis_node(state: AgentState) -> AgentState:

    data_path = state.get("data_path", "")
    profiling_report = state.get("profiling_report") or {}
    target_def = state.get("target_definition") or {}
    target_col = target_def.get("column_name") or None

    if not data_path:
        logger.info("[analysis_node] data_path boş — analiz yapılamadı")
        return {
            "analysis_report": {"error": "Veri bulunamadı."},
            "current_step": "analysis_node",
        }

    # profiling_report yoksa, en azından boş bir iskelet kullan
    if not profiling_report or "summary" not in profiling_report:
        logger.warning("[analysis_node] UYARI: profiling_report eksik, sınırlı analiz")
        profiling_report = {
            "shape": {"rows": 0, "columns": 0},
            "summary": {"numeric_columns": [], "categorical_columns": [],
                        "datetime_columns": [], "missing_rate": 0.0},
            "columns": {},
        }

    try:
        df = pd.read_parquet(data_path)
        logger.info(f"[analysis_node] Veri okundu: {len(df):,} satır × {len(df.columns)} kolon")

        # Aktif domain pack sektöre özel analiz grubu seçer (örn. ecommerce → RFM,
        # sepet birliktelik). Pack yoksa/"none" ise sadece "general" çalışır.
        from src.domains import get_active_pack
        pack = get_active_pack(state.get("domain_pack"))
        analysis_type = getattr(pack, "analysis_type", "general") or "general"

        analysis = run_deep_analysis(
            df, profiling_report, target_col=target_col, analysis_type=analysis_type
        )

        # Kısa özet log
        corr = analysis.get("correlation_matrix")
        hist = analysis.get("histograms") or {}
        clust = analysis.get("clustering")
        tr = analysis.get("target_relations")

        logger.info(f"[analysis_node] Korelasyon: "
              f"{'var (' + str(len(corr['columns'])) + ' sütun)' if corr else 'yok'}")
        logger.info(f"[analysis_node] Histogram: {len(hist)} sütun")
        logger.info(f"[analysis_node] Kümeleme: "
              f"{'var (' + str(clust['n_clusters']) + ' küme)' if clust else 'yok'}")
        if tr:
            top = tr['relations'][0] if tr['relations'] else None
            logger.info(f"[analysis_node] Hedef '{tr['target']}' ile en ilişkili: "
                  f"{top['feature'] + ' (' + str(top['score']) + ')' if top else 'yok'}")

        return {
            "analysis_report": analysis,
            "current_step": "analysis_node",
        }

    except Exception as e:
        logger.error(f"[analysis_node] HATA: {e}")
        import traceback
        traceback.print_exc()
        return {
            "analysis_report": {"error": str(e)},
            "current_step": "analysis_node",
        }