from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from src.core.llm import get_llm
from src.core.config import settings
from pydantic import BaseModel, Field

from src.state import AgentState

# Structured output model
class IntentResult(BaseModel):
    intent: str = Field(
        description="Şunlardan biri: eda, classification, regression, clustering, anomaly_detection, report, sql_query"
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
- sql_query         : Belirli bir veriyi getirme, filtreleme, sayma, gruplama, sıralama gibi
                      doğrudan sorgu istekleri. Modelleme veya analiz gerektirmeyen,
                      "şunu getir / kaç tane / en yüksek / listele" tarzı talepler.

Kurallar:
- Yalnızca tek bir intent seç
- Emin değilsen "eda" seç
- "Segment", "group", "cluster" → clustering
- "Anomaly", "fraud", "outlier", "unusual", "aykırı", "sahte" → anomaly_detection
- "Predict" + sayısal değer → regression
- "Predict" + kategori/sınıf → classification
- "Getir", "listele", "göster", "kaç tane", "en yüksek", "filtrele", "sırala",
  "select", "fetch", "show me", "how many", "top N", "filter by" → sql_query
- sql_query: model eğitimi veya istatistiksel analiz içermemeli, sadece veri erişimi olmalı
- Karşılaştırma, gruplama veya basit istatistik içeren sorgular da sql_query'dir:
  * "X ile Y arasındaki fark", "X'e göre ortalama", "gruba göre sayı/toplam/ortalama"
  * "erkek vs kadın", "A ile B'yi karşılaştır", "oranı nedir", "dağılımı nedir (basit)"
  * Bu tür sorgular GROUP BY / AVG / COUNT / SUM ile cevaplanabilir → sql_query
- Eğer soru bir MODELİN EĞİTİLMESİNİ veya TAHMİN YAPILMASINI gerektiriyorsa → classification/regression
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
    llm = get_llm(model_name=settings.INTELLIGENCE_ENGINE_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(IntentResult)

    chain = INTENT_PROMPT | structured_llm

    result: IntentResult = chain.invoke({"user_query": state["user_query"]})

    return {
        **state,
        "intent": result.intent,
        "intent_confidence": result.confidence,
        "current_step": "intent_node",
    }