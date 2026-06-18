"""
Target Discovery Node
======================
Hedef değişkeni TEK SEFER tespit eden bağımsız node.

Graph sırası:
  db_context_node → target_discovery_node → planner_node (ML intent)
  db_context_node → sql_node (SQL intent)

Sadece ML intent'lerinde çalışır:
  - classification, regression, anomaly_detection, clustering
  - sql_query, eda → atlar
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Data class
@dataclass
class TargetCandidate:
    column_name: str
    table_name: str
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    is_binary: bool = False
    unique_count: int = 0
    null_rate: float = 0.0
    description: str = ""


def _find_candidates(
    db_intelligence: dict,
    user_query: str,
    intent: str,
) -> list[TargetCandidate]:
    """
    Schema + column metadata'dan hedef adayları bulur.
    Tamamen dinamik — hardcode kolon ismi yok.
    """
    candidates: list[TargetCandidate] = []

    for col_info in db_intelligence.get("columns", []):
        if col_info.get("column_name") == "age":
            print(f"[DEBUG] age col_info: {col_info}")
    
    for col_info in db_intelligence.get("columns", []):
        col_name      = col_info.get("column_name", "")
        semantic_type = col_info.get("semantic_type", "unknown")
        description   = col_info.get("description", "")
        sample_values = col_info.get("sample_values", [])
        importance    = col_info.get("importance_score", 0.5)
        table_name    = col_info.get("table_name", "")
        null_rate     = col_info.get("null_rate", 0.0)

        if not col_name:
            continue

        # Disqualify
        if semantic_type in ("identifier", "timestamp"):
            continue

        cand = TargetCandidate(
            column_name=col_name,
            table_name=table_name,
            description=description,
            null_rate=null_rate,
        )

        # detect binary values
        if sample_values:
            unique_vals = set(str(v) for v in sample_values if v is not None)
            cand.unique_count = len(unique_vals)
            if unique_vals <= {"0", "1"} or \
               unique_vals <= {"True", "False"} or \
               unique_vals <= {"0.0", "1.0"}:
                cand.is_binary = True

        col_lower = col_name.lower()
        # Scoring
        # semantic_type=target → strongest 
        
        # EN GÜÇLÜ SİNYAL: Kullanıcı kolonu adıyla söylüyor mu?
        col_tokens = set(col_lower.replace(".", " ").replace("_", " ").split())
        query_lower = user_query.lower()
        query_tokens = set(query_lower.replace(".", " ").replace("_", " ").split())
        
        if col_lower in query_lower or any(tok in query_tokens for tok in col_tokens if len(tok) > 2):
            cand.score += 6.0
            cand.reasons.append("kullanıcı kolonu adıyla belirtti")
        
        if semantic_type == "target":
            cand.score += 5.0
            cand.reasons.append("semantic_type=target")
        
        if cand.is_binary:
            cand.score += 3.0
            cand.reasons.append("binary (0/1)")
        
        # Boolean naming — col_lower zaten yukarıda tanımlı, tekrar tanımlama
        if col_lower.startswith(("is_", "has_", "flag_")):
            cand.score += 1.5
            cand.reasons.append("boolean naming")

        if intent == "regression":
        # Regression: yüksek kardinaliteli sayısal kolon ister
            if cand.unique_count > 50:
                cand.score += 3.0
                cand.reasons.append(f"high cardinality ({cand.unique_count}) — regression")
            elif cand.unique_count > 20:
                cand.score += 1.5
                cand.reasons.append(f"medium cardinality ({cand.unique_count})")
            elif 0 < cand.unique_count <= 10:
                # Regression için düşük kardinalite kötü — penalty
                cand.score -= 1.0
                cand.reasons.append(f"too few values for regression ({cand.unique_count})")
        else:
            # Classification: düşük kardinalite ister
            if 0 < cand.unique_count <= 10:
                cand.score += 1.0
                cand.reasons.append(f"low cardinality ({cand.unique_count})")
            elif cand.unique_count > 50:
                cand.score -= 0.5
                cand.reasons.append(f"high cardinality ({cand.unique_count}) — not ideal for classification")

        # Ayrıca regression için sayısal kolon bonus'u ekle:
        if intent == "regression":
            # Description veya sample'lara bakarak sayısal mı anla
            dtype = col_info.get("dtype", "").lower()
            if any(t in dtype for t in ("int", "float", "double", "decimal", "numeric")):
                cand.score += 2.0
                cand.reasons.append("numeric dtype — good for regression")
            # Para birimi keywords (price, cost, amount, revenue)
            for kw in ("price", "cost", "amount", "revenue", "value", "fiyat", "tutar"):
                if kw in col_name.lower() or kw in description.lower():
                    cand.score += 2.0
                    cand.reasons.append(f"monetary keyword: {kw}")
                    break
        # User query × description overlap
        if user_query and description:
            stopwords = {
                "bir","bu","ve","ile","için","mi","mı","mu","mü","ne","nedir",
                "nasıl","hangi","tahmin","et","bul","analiz","yap","olan","olduğunu",
                "the","a","an","and","or","for","is","are","to","of","in","what",
                "which","how","predict","detect","find",
            }
            query_words = set(user_query.lower().split()) - stopwords
            desc_words  = set(description.lower().split()) - stopwords
            if query_words and desc_words:
                overlap = len(query_words & desc_words)
                if overlap > 0:
                    cand.score += overlap * 1.5
                    cand.reasons.append(f"query-description overlap ({overlap})")

        # Intelligence engine's importance score
        cand.score += importance * 0.5

        # Negative signals
        if null_rate > 0.3:
            cand.score -= 2.0
            cand.reasons.append(f"high null rate ({null_rate:.0%})")

        if cand.unique_count == 1:
            continue  

        if cand.score > 0:
            candidates.append(cand)

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


# target_discovery_node.py içindeki _ask_llm fonksiyonunu şununla replace et:

def _ask_llm(
    user_query: str,
    intent: str,
    candidates: list[TargetCandidate],
    db_intelligence: dict,
) -> str:
    """LLM'e final kararı sor. Hata varsa boş string döner."""
    import os
    from langchain_core.messages import HumanMessage, SystemMessage
    from src.core.llm import get_llm

    tables_summary = "\n".join(
        f"- {t.get('table_name')}: {t.get('description', '')}"
        for t in db_intelligence.get("tables", [])
    )

    candidates_text = "\n".join(
        f"  - {c.column_name} (tablo: {c.table_name}, skor: {c.score:.2f}, "
        f"binary: {c.is_binary}, açıklama: {c.description})"
        for c in candidates[:8]
    )

    system_prompt = """Sen bir veri bilimci asistanısın.
Kullanıcının sorusu ve aday hedef kolonları verilecek.
En uygun hedef değişkeni seç.

KURALLAR:
- DİKKAT: Kullanıcının sorusunda AÇIKÇA belirttiği veya ima ettiği değişkeni seç. (Örn: "fiyatı tahmin et" diyorsa price, "pozitif oyları tahmin et" diyorsa positivereview seç). Kullanıcının istediği spesifik kolon varsa diğer tüm kuralları yok say!
- Kolon açıklamalarına ve adlarına dikkat et.
- Binary (0/1) kolonlar classification için idealdir.
- Regression için yüksek kardinaliteli sayısal kolonlar idealdir.
- ID, timestamp, identifier kolonları ASLA hedef olamaz.

Sadece seçtiğin kolonun TAM ADINI döndür, başka bir şey yazma."""

    user_prompt = f"""Kullanıcı sorusu: {user_query}
Intent: {intent}

Tablolar:
{tables_summary}

Aday hedef kolonlar:
{candidates_text}

En uygun hedef kolon hangisi? Sadece kolon adını yaz:"""

    try:
        model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o")
        llm = get_llm(model_name=model_name, temperature=0)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        result = response.content.strip()
        result = result.strip("'\"`*. \n")
        if " " in result:
            result = result.split()[0]
        print(f"[target_discovery] LLM onerisi: '{result}'")
        return result
    except Exception as e:
        print(f"[target_discovery] LLM hatasi: {e}")
        return ""

# Final decision (Guardrail)
def _resolve_target(
    candidates: list[TargetCandidate],
    llm_suggestion: str,
) -> TargetCandidate | None:
    """
    Final karar:
    1. LLM önerisi adaylar arasında varsa → onu seç
    2. AMA skor farkı > 3.0 ise matematiksel skor override eder
    3. LLM önerisi adaylar arasında yoksa → en yüksek skorlu
    """
    if not candidates:
        return None

    best = candidates[0]

    if llm_suggestion:
        llm_cand = next((c for c in candidates if c.column_name == llm_suggestion), None)
        if llm_cand:
            score_gap = best.score - llm_cand.score
            if score_gap > 3.0:
                print(f"[target_discovery] GUARDRAIL: LLM '{llm_suggestion}' secti ama "
                      f"'{best.column_name}' cok daha yuksek skorlu "
                      f"({best.score:.2f} vs {llm_cand.score:.2f}). Override.")
            else:
                best = llm_cand
                print(f"[target_discovery] LLM onerisi kabul edildi.")
        else:
            print(f"[target_discovery] LLM onerisi '{llm_suggestion}' adaylar arasinda yok.")

    return best


# Manuel hedef seçimi: görev tipini hedef sütunun verisinden çıkar
def infer_task_type(series) -> str:
    """
    Hedef sütunun verisinden görev tipini (classification/regression) çıkarır.

    Eşikler bu dosyadaki _find_candidates skorlama mantığından BİREBİR alındı
    (yeni eşik uydurulmadı):
      - non-numeric dtype                    → classification
        (regression sayısal kolon ister; ml_trainer satır 925-931 ile tutarlı)
      - binary (0/1, True/False)             → classification
        (_find_candidates is_binary, satır 75-78 / 97)
      - sayısal + unique_count > 50          → regression
        (_find_candidates satır 108: en güçlü regression sinyali)
      - sayısal + düşük kardinalite (≤50)    → classification
        (_find_candidates satır 120-125: düşük kardinalite classification ister)
    """
    import pandas as pd

    s = series.dropna()

    # non-numeric → classification
    if not pd.api.types.is_numeric_dtype(s):
        return "classification"

    # binary (0/1) → classification
    unique_vals = set(str(v) for v in s.unique())
    if unique_vals <= {"0", "1"} or \
       unique_vals <= {"0.0", "1.0"} or \
       unique_vals <= {"True", "False"}:
        return "classification"

    # sayısal + yüksek kardinalite → regression, aksi halde classification
    if s.nunique() > 50:
        return "regression"
    return "classification"


# Main node
def discover_target(state: dict) -> dict:
    """
    Hedef değişkeni tespit eder ve state'e yazar.

    State çıktıları:
        target_definition : dict
          - column_name
          - table_name
          - is_binary
          - description
          - confidence (0.0-1.0)
          - confirmed (False — feature_plan_node onaylar)
    """
    intent          = state.get("intent", "")
    user_query      = state.get("user_query", "")
    db_intelligence = state.get("db_intelligence", {})

    # No target for EDA or SQL query
    if intent in ("sql_query", "eda"):
        print(f"[target_discovery] Intent '{intent}' — atlandi.")
        return {**state, "target_definition": {}, "current_step": "target_discovery_node"}

    print(f"\n{'='*60}")
    print(f"[target_discovery] BASLIYOR")
    print(f"  Intent : {intent}")
    print(f"  Sorgu  : {user_query[:80].encode('ascii', 'replace').decode('ascii')}...")
    print(f"{'='*60}")

    candidates = _find_candidates(db_intelligence, user_query, intent)

    if not candidates:
        print("[target_discovery] Hic aday bulunamadi.")
        return {**state, "target_definition": {}, "current_step": "target_discovery_node"}

    # Log top 5 candidates
    for c in candidates[:5]:
        print(f"[target_discovery] Aday: '{c.column_name}' "
              f"(skor={c.score:.2f}, binary={c.is_binary}) "
              f"reasons={c.reasons}")

    # 2. Ask LLM
    llm_suggestion = _ask_llm(user_query, intent, candidates, db_intelligence)

    # 3. Final decision
    best = _resolve_target(candidates, llm_suggestion)

    if not best:
        print("[target_discovery] Hedef belirlenemedi.")
        return {**state, "target_definition": {}, "current_step": "target_discovery_node"}

    confidence = min(best.score / 10.0, 1.0)
    target_dict = {
        "column_name": best.column_name,
        "table_name":  best.table_name,
        "is_binary":   best.is_binary,  # regresyonda üretilir ama göz ardı edilir (zararsız)
        "description": best.description,
        "confidence":  round(confidence, 2),
        "confirmed":   False, 
    }

    print(f"\n[target_discovery] KARAR: '{best.column_name}' "
          f"(tablo: {best.table_name}, binary: {best.is_binary}, "
          f"confidence: {confidence:.2f})")

    return {
        "target_definition": target_dict,
        "error": None
    }