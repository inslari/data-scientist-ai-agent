"""
Text-analysis LangGraph pipeline (yazılı geri bildirim analizi).

Tabular ana hat (src/graph.py) ve Talk-to-Data grafiğinden bağımsız, paralel bir
graftır. Doğrusal akış:

    START → preprocess → detect_timestamp → classify → theme → trend → report → END

- preprocess:        ham metinleri temizler, kolon timestamp'lerini hizalar (lokal)
- detect_timestamp:  veride zaman damgası var mı? (agent / kolon tespiti)
- classify:          her metin için sentiment/kategori/aciliyet/aspect + (varsa) tarih (LLM, partili)
- theme:             TF-IDF/embedding + KMeans kümeleme + LLM tema etiketleme (hibrit)
- trend:             zaman damgalı kayıtlardan duygu trendi (lokal, pandas)
- report:            yönetici özeti + trendler (LLM, runs_on_error=True)

Hata yönetimi @node dekoratörüyle otomatiktir: bir node hata verirse sonraki
node'lar atlanır ve hata report_node'a kadar taşınıp düzgün raporlanır.
"""
from langgraph.graph import END, START, StateGraph

from .state import TextAnalysisState
from .nodes.preprocess_node import ta_preprocess_node
from .nodes.detect_timestamp_node import ta_detect_timestamp_node
from .nodes.classify_node import ta_classify_node
from .nodes.theme_node import ta_theme_node
from .nodes.trend_node import ta_trend_node
from .nodes.report_node import ta_report_node


def build_text_analysis_graph():
    g = StateGraph(TextAnalysisState)  # type: ignore[type-var]

    g.add_node("ta_preprocess_node", ta_preprocess_node)
    g.add_node("ta_detect_timestamp_node", ta_detect_timestamp_node)
    g.add_node("ta_classify_node", ta_classify_node)
    g.add_node("ta_theme_node", ta_theme_node)
    g.add_node("ta_trend_node", ta_trend_node)
    g.add_node("ta_report_node", ta_report_node)

    g.add_edge(START, "ta_preprocess_node")
    g.add_edge("ta_preprocess_node", "ta_detect_timestamp_node")
    g.add_edge("ta_detect_timestamp_node", "ta_classify_node")
    g.add_edge("ta_classify_node", "ta_theme_node")
    g.add_edge("ta_theme_node", "ta_trend_node")
    g.add_edge("ta_trend_node", "ta_report_node")
    g.add_edge("ta_report_node", END)

    return g.compile()
