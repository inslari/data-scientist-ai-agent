import base64

import requests
from langchain_core.runnables.graph import NodeStyles

from src.graph import build_graph

from src.core.logging_config import get_logger
logger = get_logger(__name__)

OUTPUT_PATH = "graph_mimarisi.png"
BACKGROUND = "0B0F19"   # mermaid.ink bgColor'ı '#' olmadan bekler

DARK_NODE_STYLES = NodeStyles(
    default="fill:#111827,stroke:#38BDF8,stroke-width:1.5px,color:#E5E7EB,line-height:1.2",
    first="fill:#0E7490,stroke:#22D3EE,stroke-width:2px,color:#ECFEFF",
    last="fill:#312E81,stroke:#818CF8,stroke-width:2px,color:#EEF2FF",
)

DARK_FRONTMATTER = {
    "config": {
        "theme": "dark",
        "themeVariables": {
            "lineColor": "#64748B",
            "primaryTextColor": "#E5E7EB",
            "fontSize": "16px",
        },
    }
}

# Grafı derleyip mermaid söz dizimini koyu tema ayarlarıyla üretiyoruz
app = build_graph()
mermaid_syntax = app.get_graph().draw_mermaid(
    node_colors=DARK_NODE_STYLES,
    frontmatter_config=DARK_FRONTMATTER,
)

# draw_mermaid_png() arka plan rengini '%23RRGGBB' olarak encode ettiği için
# mermaid.ink bunu yok sayıp beyaza düşüyor; bu yüzden isteği kendimiz kuruyoruz.
encoded = base64.b64encode(mermaid_syntax.encode("utf8")).decode("ascii")
url = f"https://mermaid.ink/img/{encoded}?type=png&bgColor={BACKGROUND}"

response = requests.get(url, timeout=60)
response.raise_for_status()

with open(OUTPUT_PATH, "wb") as f:
    f.write(response.content)

logger.info(f"Graf '{OUTPUT_PATH}' olarak koyu temada kaydedildi!")
