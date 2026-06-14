"""
talk_to_data_graph.py
─────────────────────
Talk to Data için bağımsız LangGraph pipeline.

Mevcut graph.py (ML pipeline) ile HİÇBİR bağlantısı yoktur.
Bu dosya artık sadece bir yönlendirici (facade) görevi görmektedir.
Tüm Graph node'ları, agent'lar ve state tanımlamaları `talktodata` klasörü altındadır.
"""

from .talktodata.graph import build_ttd_graph

__all__ = ["build_ttd_graph"]
