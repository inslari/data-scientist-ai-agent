from src.graph import build_graph

from src.core.logging_config import get_logger
logger = get_logger(__name__)

# Grafı derleyip oluşturuyoruz
app = build_graph()

# Grafı PNG'ye dönüştürüp kaydediyoruz
png_data = app.get_graph().draw_mermaid_png()

with open("graph_mimarisi.png", "wb") as f:
    f.write(png_data)
    
logger.info("Graf 'graph_mimarisi.png' olarak başarıyla kaydedildi!")
