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


def analysis_node(state: AgentState) -> AgentState:
    print("\n========================================\n"
          "[NODE START] analysis_node başlıyor...\n"
          "========================================")

    data_path = state.get("data_path", "")
    profiling_report = state.get("profiling_report") or {}
    target_def = state.get("target_definition") or {}
    target_col = target_def.get("column_name") or None

    if not data_path:
        print("[analysis_node] data_path boş — analiz yapılamadı")
        return {
            **state,
            "analysis_report": {"error": "Veri bulunamadı."},
            "current_step": "analysis_node",
        }

    # profiling_report yoksa, en azından boş bir iskelet kullan
    if not profiling_report or "summary" not in profiling_report:
        print("[analysis_node] UYARI: profiling_report eksik, sınırlı analiz")
        profiling_report = {
            "shape": {"rows": 0, "columns": 0},
            "summary": {"numeric_columns": [], "categorical_columns": [],
                        "datetime_columns": [], "missing_rate": 0.0},
            "columns": {},
        }

    try:
        df = pd.read_parquet(data_path)
        print(f"[analysis_node] Veri okundu: {len(df):,} satır × {len(df.columns)} kolon")

        analysis = run_deep_analysis(df, profiling_report, target_col=target_col)

        # Kısa özet log
        corr = analysis.get("correlation_matrix")
        hist = analysis.get("histograms") or {}
        clust = analysis.get("clustering")
        tr = analysis.get("target_relations")

        print(f"[analysis_node] Korelasyon: "
              f"{'var (' + str(len(corr['columns'])) + ' sütun)' if corr else 'yok'}")
        print(f"[analysis_node] Histogram: {len(hist)} sütun")
        print(f"[analysis_node] Kümeleme: "
              f"{'var (' + str(clust['n_clusters']) + ' küme)' if clust else 'yok'}")
        if tr:
            top = tr['relations'][0] if tr['relations'] else None
            print(f"[analysis_node] Hedef '{tr['target']}' ile en ilişkili: "
                  f"{top['feature'] + ' (' + str(top['score']) + ')' if top else 'yok'}")

        return {
            **state,
            "analysis_report": analysis,
            "current_step": "analysis_node",
        }

    except Exception as e:
        print(f"[analysis_node] HATA: {e}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "analysis_report": {"error": str(e)},
            "current_step": "analysis_node",
        }