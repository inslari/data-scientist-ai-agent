import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph import build_graph

# Grafı derleyip oluşturuyoruz
app = build_graph()

# Grafı PNG'ye dönüştürüp kaydediyoruz
png_data = app.get_graph().draw_mermaid_png()

with open("graph_mimarisi.png", "wb") as f:
    f.write(png_data)
    
print("Graf 'graph_mimarisi.png' olarak başarıyla kaydedildi!")
