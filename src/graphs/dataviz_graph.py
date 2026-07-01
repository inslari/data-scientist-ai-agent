"""
dataviz_graph.py
────────────────
Data Visualization için bağımsız LangGraph pipeline (facade).

Talk-to-Data graph'ının görselleştirme muadili: doğal dil / seçilen öneri → şema
analizi → viz planı → SQL → doğrulama → matplotlib render → yorum. Mevcut ML
pipeline (graph.py) ve Talk-to-Data (talk_to_data_graph.py) ile paraleldir; onlara
dokunmaz. Tüm node/state tanımları `dataviz/` klasörü altındadır.
"""

from .dataviz.graph import build_dataviz_graph

__all__ = ["build_dataviz_graph"]
