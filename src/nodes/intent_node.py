from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.state import AgentState


# --- Structured output modeli ---
class IntentResult(BaseModel):
    intent: str = Field(
        description="Şunlardan biri: eda, classification, regression, clustering, anomaly_detection, report"
    )
    confidence: float = Field(
        description="0 ile 1 arasında güven skoru"
    )
    reasoning: str = Field(
        description="Bu intent'in neden seçildiğini açıklayan tek cümle (Türkçe)"
    )


INTENT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Sen bir veri bilimi ajanının intent sınıflandırıcısısın. Kullanıcı sorularını Türkçe veya İngilizce alabilirsin.

Görevin: Kullanıcının sorgusunu aşağıdaki kategorilerden BİRİNE sınıflandırmak.

- eda               : Keşifsel veri analizi, istatistik, dağılım inceleme, veri profilleme
- classification    : Kategorik bir hedef değişkeni tahmin etme (evet/hayır, sınıf tahmini)
- regression        : Sayısal/sürekli bir hedef değişkeni tahmin etme (fiyat, miktar, skor)
- clustering        : Etiketsiz veriyi gruplara ayırma (segmentasyon, kümeleme)
- anomaly_detection : Aykırı değer, sahte işlem, anormal davranış tespiti
- report            : Mevcut veriden özet veya iş raporu üretme

Kurallar:
- Yalnızca tek bir intent seç
- Emin değilsen "eda" seç
- "Segment", "group", "cluster" → clustering
- "Anomaly", "fraud", "outlier", "unusual", "aykırı", "sahte" → anomaly_detection
- "Predict" + sayısal değer → regression
- "Predict" + kategori/sınıf → classification
- Yukarıdaki liste dışında intent üretme

Yanıtını yapılandırılmış JSON olarak döndür: intent, confidence (0-1), reasoning (Türkçe).
""",
    ),
    ("human", "Kullanıcı sorusu: {user_query}"),
])


def intent_node(state: AgentState) -> AgentState:
    """
    Node 1: Kullanıcının sorgusunu analiz edip intent'i belirler.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = llm.with_structured_output(IntentResult)

    chain = INTENT_PROMPT | structured_llm

    result: IntentResult = chain.invoke({"user_query": state["user_query"]})

    return {
        **state,
        "intent": result.intent,
        "intent_confidence": result.confidence,
        "current_step": "intent_node",
    }