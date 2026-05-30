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

from src.state import AgentState
from src.core.config import settings


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


        # Scoring
        # semantic_type=target → strongest signal
        if semantic_type == "target":
            cand.score += 5.0
            cand.reasons.append("semantic_type=target")

        # Binary column → classification ideal
        if cand.is_binary:
            cand.score += 3.0
            cand.reasons.append("binary (0/1)")

        # Boolean naming pattern (is_*, has_*, flag_*)
        col_lower = col_name.lower()
        if col_lower.startswith(("is_", "has_", "flag_")):
            cand.score += 1.5
            cand.reasons.append("boolean naming")

        # Low cardinality numerical → classification
        if 0 < cand.unique_count <= 10:
            cand.score += 1.0
            cand.reasons.append(f"low cardinality ({cand.unique_count})")
        elif cand.unique_count > 50 and intent == "regression":
            cand.score += 1.0
            cand.reasons.append("high cardinality, regression candidate")

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


def _ask_llm(
    user_query: str,
    intent: str,
    candidates: list[TargetCandidate],
    db_intelligence: dict,
) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        print("[target_discovery] OpenAI yok, matematiksel skor kullanılıyor.")
        return ""

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
- Kullanıcının sorusunun ANLAMINI dikkate al
- Kolon AÇIKLAMASINA bak, kolon ADINA değil
- Binary (0/1) kolonlar classification için idealdir
- ID, timestamp, identifier kolonları ASLA hedef olamaz

Sadece kolon adını döndür, başka bir şey yazma."""

    user_prompt = f"""Kullanıcı sorusu: {user_query}
Intent: {intent}

Tablolar:
{tables_summary}

Aday hedef kolonlar:
{candidates_text}

En uygun hedef kolon hangisi? Sadece kolon adını yaz:"""

    try:
        client = OpenAI(
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY or "not-needed",
        )
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=50,
        )
        result = response.choices[0].message.content.strip()
        result = result.strip("'\"`*. \n")
        if " " in result:
            result = result.split()[0]
        print(f"[target_discovery] LLM önerisi: '{result}'")
        return result
    except Exception as e:
        print(f"[target_discovery] LLM hatası: {e}")
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
                print(f"[target_discovery] GUARDRAIL: LLM '{llm_suggestion}' seçti ama "
                      f"'{best.column_name}' çok daha yüksek skorlu "
                      f"({best.score:.2f} vs {llm_cand.score:.2f}). Override.")
            else:
                best = llm_cand
                print(f"[target_discovery] LLM önerisi kabul edildi.")
        else:
            print(f"[target_discovery] LLM önerisi '{llm_suggestion}' adaylar arasında yok.")

    return best


# Main node
def target_discovery_node(state: AgentState) -> AgentState:
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
        print(f"[target_discovery] Intent '{intent}' — atlandı.")
        return {**state, "target_definition": {}, "current_step": "target_discovery_node"}

    print(f"\n{'='*60}")
    print(f"[target_discovery] BAŞLIYOR")
    print(f"  Intent : {intent}")
    print(f"  Sorgu  : {user_query[:80]}...")
    print(f"{'='*60}")

    candidates = _find_candidates(db_intelligence, user_query, intent)

    if not candidates:
        print("[target_discovery] Hiç aday bulunamadı.")
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
        "is_binary":   best.is_binary,
        "description": best.description,
        "confidence":  round(confidence, 2),
        "confirmed":   False, 
    }

    print(f"\n[target_discovery] ✓ KARAR: '{best.column_name}' "
          f"(tablo: {best.table_name}, binary: {best.is_binary}, "
          f"confidence: {confidence:.2f})")

    return {
        **state,
        "target_definition": target_dict,
        "current_step": "target_discovery_node",
    }