from src.core.nodes import node
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from ..state import AgentState
from ..core.config import settings
from ..core.llm import get_llm
from ..core.prompt_loader import create_chat_prompt
from ..core.structured import invoke_structured
from ..schemas.feature_plan_schemas import FeaturePlan
from ..services.analysis_engine import extract_plan_hints, format_plan_hints
from ..domains import get_active_pack
from ..domains.base import MerchantContext, trim_doc
from ..domains.schemas import MerchantAnalysis

from src.core.logging_config import get_logger
logger = get_logger(__name__)

load_dotenv(override=True)

FEATURE_PLAN_PROMPT = create_chat_prompt("feature_plan_system.md", "feature_plan_human.md")

@node("feature_plan_node")
def feature_plan_node(state: AgentState) -> AgentState:
    """
    Node 4.5: Kullanıcı sorgusundan feature engineering planı üretir.

    LLM sadece JSON plan üretir — pandas kodu yazmaz.
    Plan feature_engineering_node tarafından execute edilir.

    State girdileri:
        user_query     : Kullanıcının ham sorusu
        intent         : Tespit edilen intent
        db_intelligence: intelligence_engine metadata'sı

    State çıktıları:
        feature_plan   : FeaturePlan dict olarak
    """
    user_query      = state.get("user_query", "")
    intent          = state.get("intent", "classification")
    db_intelligence = state.get("db_intelligence", {})
    critic_feedback = state.get("critic_feedback", [])
    retry_count     = state.get("retry_count", 0)

    target_def_initial = state.get("target_definition") or {}
    if target_def_initial.get("column_name"):
        logger.info(f"[feature_plan] target_discovery_node önerisi: '{target_def_initial['column_name']}' "
              f"(confidence={target_def_initial.get('confidence', 0):.2f})")
    
    if not db_intelligence:
        return {
            "error": "db_intelligence bulunamadı.",
            "current_step": "feature_plan_node",
        }

    tables_summary = _build_db_summary(db_intelligence)

    model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o")
    llm = get_llm(model_name=model_name, temperature=0)

    target_hint = ""
    if target_def_initial.get("column_name"):
        target_hint = f"""

📍 ÖN-TESPİT EDİLEN HEDEF:
target_discovery_node şu kolonu önerdi: '{target_def_initial['column_name']}'
Confidence: {target_def_initial.get('confidence', 0):.2f}
Bu öneriyi dikkate al ama kendi guardrail kontrolünü yap."""

    # Leakage hint: domain leakage suspect'leri math interaction'dan uzak tut
    leakage_hint = ""
    domain_leakage = state.get("domain_leakage_report") or {}
    concerns = domain_leakage.get("concerns", [])
    if concerns:
        suspect_cols = [c["column_name"] for c in concerns]
        leakage_hint = f"""
    
⚠️ LEAKAGE UYARISI:
Şu kolonlar target ile şüpheli ilişki içinde (leakage riski): {', '.join(suspect_cols)}
Bu kolonları derived_features önerilerinde girdi olarak KULLANMA.
Bunlardan türetilmiş yeni feature'lar da leakage taşır."""
    
    # Retry bağlamı: critic feedback'i prompt'a ekle
    retry_context = ""
    if retry_count > 0 and critic_feedback:
        feedback_str = "\n".join(f"  - {fb}" for fb in critic_feedback)
        retry_context = f"""

⚠️ RETRY #{retry_count} — ÖNCEKİ PLAN BAŞARISIZ OLDU:
{feedback_str}

Yukarıdaki sorunları çözmek için FARKLI bir feature engineering stratejisi üret.
Önceki planla aynı yapıyı tekrarlama."""

    # Domain pack (varsa) feature ipuçları — yoksa boş string
    domain_hint = ""
    try:
        pack = get_active_pack(state.get("domain_pack"))
        mc = MerchantContext.from_dict(state.get("merchant_context"))
        analysis = MerchantAnalysis.from_dict(state.get("merchant_analysis"))
        if pack.name != "none" and (mc is not None or analysis is not None):
            hint = pack.feature_hints(mc, db_intelligence, analysis)
            if hint and hint.strip():
                domain_hint = "\n" + hint
                src = "agentic bulgular" if analysis else "statik ipuçları"
                logger.info(f"[feature_plan] Domain pack '{pack.name}' feature ipuçları eklendi ({src}).")
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"[feature_plan] Domain pack feature ipucu başarısız ({e}); atlanıyor.")

    # Analist ipuçları (hedef-kör yapısal bulgular). Hedef kolon ve hedef adayları
    # extract_plan_hints'e HİÇ girmez → target-farkındalıklı ipucu üretilemez.
    analysis_hints = _build_analysis_hints(state)
    analysis_hint = format_plan_hints(analysis_hints) if settings.ENABLE_PLAN_HINTS else ""
    if analysis_hint:
        _counts = {k: len(v) for k, v in analysis_hints.items() if isinstance(v, list) and v}
        logger.info(f"[feature_plan] Analist ipuçları eklendi "
                    f"(kaynak={analysis_hints.get('generated_from')}): {_counts}")

    # Per-data-group domain document (dataset context) — pack'ten bağımsız enjekte edilir
    domain_doc = ""
    doc_text = trim_doc(state.get("domain_doc"), max_chars=4000)
    if doc_text:
        domain_doc = (
            "\n\n--- DATASET DESCRIPTION (provided by the user) ---\n"
            f"{doc_text}\n"
            "Use this to choose meaningful, domain-aware features and the right target; "
            "only reference columns that exist in the schema above."
        )
        logger.info("[feature_plan] Dataset domain_doc feature plan'a eklendi.")

    prompt_value = FEATURE_PLAN_PROMPT.invoke({
        "user_query": user_query,
        "intent": intent,
        "tables_summary": tables_summary,
        "target_hint": target_hint,
        "retry_context": retry_context,
        "leakage_hint": leakage_hint,
        "analysis_hint": analysis_hint,
        "domain_hint": domain_hint,
        "domain_doc": domain_doc,
    })

    # Structured output with a text-parser fallback (C1/H4).
    plan: FeaturePlan = invoke_structured(llm, prompt_value, FeaturePlan)

    # ── Post-processing: Dinamik target validation ────────────────────────────
    # LLM'in hedef kolon seçimini hem target_discovery_node önerisi
    # hem de db_intelligence semantic_type etiketlerine göre validate eder.
    plan_dict = plan.model_dump()
    target_def_initial = state.get("target_definition", {}) or {}
    logger.info(f"[DEBUG FP] target_def_initial = {target_def_initial}")
    _validate_target_selection(plan_dict, user_query, db_intelligence, target_def_initial)
    plan = FeaturePlan(**plan_dict)
    # ──────────────────────────────────────────────────────────────────────────

    logger.info(f"\n[feature_plan] Analiz özeti  : {plan.analysis_summary}")
    logger.info(f"[feature_plan] Hedef değişken: {plan.target.column_name} "
          f"({'türetilecek' if plan.target.is_derived else 'mevcut'})")
    logger.info(f"[feature_plan] Tanım         : {plan.target.positive_class_description}")
    logger.info(f"[feature_plan] Aggregation   : {plan.aggregation.level} "
          f"(group_by: {plan.aggregation.group_by_column})")
    logger.info(f"[feature_plan] Feature grupları: {plan.feature_groups}")
    if plan.derived_features:
        _summ = [f"{d.name}={d.op}{tuple(d.inputs)}" for d in plan.derived_features]
        logger.info(f"[feature_plan] Türetilmiş feature'lar: {_summ}")
    logger.info(f"[feature_plan] Data split    : {plan.use_data_split_for_label}")
    logger.info(f"[feature_plan] Neden         : {plan.reasoning}")

    # Kilitli-holdout ölçüm modunun sınırı: train partisyonunu SATIR düzeyinde yeniden
    # ürettik. Plan agrege ediyorsa (satır → müşteri/ürün) feature_builder'ın gerçek
    # split'i agregasyon SONRASI yapılır; satır düzeyli partisyonumuz onunla birebir
    # örtüşmez. Garanti yaklaşıklaşır — sessiz geçmiyoruz, işaretliyoruz.
    if analysis_hints.get("generated_from") == "train_only" and plan.aggregation.required:
        analysis_hints["generated_from"] = "train_only_approx"
        logger.warning("[feature_plan] KILITLI-HOLDOUT: plan agrege ediyor "
                       f"(level={plan.aggregation.level}) — satır düzeyli train partisyonu "
                       "gerçek split'le birebir örtüşmüyor, train-only garantisi YAKLAŞIK.")

    return {
        "feature_plan": plan.model_dump(),
        "analysis_hints": analysis_hints,
        "current_step": "feature_plan_node",
    }


# Analist ipucu kanalı
def _build_analysis_hints(state: AgentState) -> dict:
    """
    Hedef-kör yapısal ipuçlarını üretir (analysis_engine.extract_plan_hints).

    Buradaki iki iş bilinçli olarak AYRI tutulur:
      1. SATIR SEÇİMİ (bu fonksiyon) — kilitli-holdout modunda train partisyonunu
         yeniden üretmek için hedefi stratify anahtarı olarak kullanabilir.
      2. İSTATİSTİK HESABI (extract_plan_hints) — hedefi HİÇ görmez.
    Hedef kolon df'ten düşürülür ve exclude_columns'a da konur; ipucu üreticisi onu
    hiçbir şekilde göremez.
    """
    profiling_report = state.get("profiling_report") or {}
    if not profiling_report.get("columns"):
        return {}

    exclude = _target_candidates(state)

    df = None
    try:
        data_path = state.get("data_path", "")
        if data_path and Path(data_path).exists():
            df = pd.read_parquet(data_path)
            if settings.ENABLE_LOCKED_HOLDOUT:
                df, carved = _train_partition(df, state)
            else:
                carved = False
            # Hedefi tamamen görüş alanından çıkar (redundant_pairs bir
            # (feature, target) çifti dahi üretemesin).
            df = df.drop(columns=[c for c in exclude if c in df.columns], errors="ignore")
        else:
            carved = False
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"[feature_plan] Analist ipuçları: veri okunamadı ({e}); "
                       "yalnızca profiling'ten üretiliyor.")
        df, carved = None, False

    hints = extract_plan_hints(df, profiling_report, exclude_columns=exclude)
    if hints and carved:
        hints["generated_from"] = "train_only"
    return hints


def _target_candidates(state: AgentState) -> set[str]:
    """Hedef kolon + tüm hedef adayları — ipucu üreticisinin göremeyeceği kolonlar."""
    cands: set[str] = set()

    target_def = state.get("target_definition") or {}
    if target_def.get("column_name"):
        cands.add(target_def["column_name"])
    if state.get("target_col"):
        cands.add(state["target_col"])

    for col in (state.get("db_intelligence") or {}).get("columns", []):
        if col.get("semantic_type") == "target" and col.get("column_name"):
            cands.add(col["column_name"])

    return cands


def _train_partition(df: "pd.DataFrame", state: AgentState) -> tuple["pd.DataFrame", bool]:
    """
    ÖLÇÜM MODU: feature_builder'ın train partisyonunu yeniden üretir (aynı TEST_SIZE,
    aynı seed, sınıflandırmada hedefe göre stratify) ve YALNIZCA train satırlarını
    döndürür. Test satırları analistin görüş alanına kilitlenir.

    Eğitim verisi kaybı YOK — satırlar yalnızca ipucu hesabından çıkarılır, modelden
    değil. Bu yüzden baseline kaymaz ve iki kol (holdout açık/kapalı) AYNI iç test
    setinde doğrudan karşılaştırılabilir.

    Returns: (df, carved) — carved=False ise partisyon üretilemedi, tüm veri döner.
    """
    from sklearn.model_selection import train_test_split
    from src.services.feature_builder import TEST_SIZE

    target_col = (state.get("target_definition") or {}).get("column_name") or state.get("target_col")
    if not target_col or target_col not in df.columns:
        logger.warning("[feature_plan] KILITLI-HOLDOUT: hedef kolon bilinmiyor — "
                       "train partisyonu üretilemedi, ipuçları TÜM veriden.")
        return df, False

    # feature_builder hedefi boş satırlarla eğitmez; partisyonun hizalı olması için
    # aynı satırları burada da düşür.
    df = df[df[target_col].notna()]
    stratify = df[target_col] if state.get("intent") == "classification" else None
    try:
        train_df, _ = train_test_split(
            df, test_size=TEST_SIZE, random_state=settings.RANDOM_SEED, stratify=stratify
        )
    except ValueError:
        train_df, _ = train_test_split(df, test_size=TEST_SIZE, random_state=settings.RANDOM_SEED)

    logger.info(f"[feature_plan] KILITLI-HOLDOUT açık: ipuçları yalnızca train "
                f"partisyonundan ({len(train_df):,}/{len(df):,} satır); test satırları kilitli.")
    return train_df, True


def _build_db_summary(db_intelligence: dict) -> str:
    """
    db_intelligence'ı LLM için okunabilir formata dönüştürür.
    Target adaylarını öne çıkarır ve açıklamaları gösterir.
    """
    lines = []

    # Tablolar
    for table in db_intelligence.get("tables", []):
        lines.append(
            f"Tablo: {table['table_name']} | "
            f"Varlık: {table['inferred_entity']} | "
            f"Potansiyel: {table['analytics_potential_score']:.2f}"
        )

    all_cols = db_intelligence.get("columns", [])

    # ── 1. Hedef kolon adaylarını öne çıkar ──
    target_cols = [c for c in all_cols if c.get("semantic_type") == "target"]
    if target_cols:
        lines.append("\n⚠️  HEDEF KOLON ADAYLARI (semantic_type=target):")
        for col in target_cols:
            desc = col.get("description", "Açıklama yok")
            samples = col.get("sample_values", "")
            lines.append(
                f"  *** {col['table_name']}.{col['column_name']} ***\n"
                f"      Açıklama: {desc}\n"
                f"      Örnek değerler: {samples}"
            )
    else:
        lines.append("\nℹ️  Hedef kolon adayı bulunamadı. LLM kullanıcı sorgusuna göre seçecek.")

    # ── 2. Diğer kolonlar ──
    lines.append("\nDiğer kolonlar (semantic_type | importance | açıklama):")
    for col in all_cols:
        if col.get("semantic_type") == "target":
            continue  # zaten yukarıda gösterildi
        desc = col.get("description", "")
        desc_str = f" | {desc[:80]}" if desc else ""
        lines.append(
            f"  {col['table_name']}.{col['column_name']} → "
            f"{col['semantic_type']} | importance: {col['importance_score']:.2f}{desc_str}"
        )

    rels = db_intelligence.get("relationships", [])
    if rels:
        lines.append("\nİlişkiler:")
        for rel in rels:
            lines.append(
                f"  {rel['source_table']}.{rel['source_column']} → "
                f"{rel['target_table']}.{rel['target_column']}"
            )

    caps = db_intelligence.get("capabilities", [])
    if caps:
        lines.append(f"\nKapasiteler: {', '.join(caps)}")

    return "\n".join(lines)


# Dinamik Target Validation 
def _validate_target_selection(
    plan_dict: dict,
    user_query: str,
    db_intelligence: dict,
    target_definition: dict | None = None,
) -> None:
    target_block   = plan_dict.get("target", {})
    current_target = target_block.get("column_name", "")
    all_cols       = db_intelligence.get("columns", [])
    target_def     = target_definition or {}
    discovery_target = target_def.get("column_name", "")
    confidence       = target_def.get("confidence", 0.0)

    logger.info(f"[DEBUG VALIDATE] target_definition param = {target_definition}")
    logger.info(f"[DEBUG VALIDATE] discovery_target = '{discovery_target}', confidence = {confidence}")

    # ── GUARD: türetilmiş hedef tanım gereği MEVCUT KOLON DEĞİLDİR ──
    # (ör. churn: veride churn kolonu yok, date + cust_id'den türetilecek). Bunu
    # 'halüsinasyon' sanıp override etme/silme — LLM bilinçli olarak türetmeyi seçti.
    if target_block.get("is_derived", False):
        logger.warning(f"[feature_plan] Hedef '{current_target}' türetilmiş "
              f"(derivation_type={target_block.get('derivation_type', '?')}) — "
              f"kolon-varlık doğrulaması atlanıyor.")
        return
    # ── GUARD: target_discovery karar verdiyse, DOKUNMA ──
    if discovery_target and confidence >= 0.4:
        # db'de var mı sadece onu kontrol et
        col_names = [c.get("column_name", "") for c in all_cols]
        if discovery_target in col_names:
            plan_dict["target"]["column_name"] = discovery_target
            logger.info(f"[feature_plan] target_discovery karari korunuyor: '{discovery_target}' (confidence={confidence:.2f})")
            return
        else:
            logger.warning(f"[feature_plan] target_discovery onerisi '{discovery_target}' db'de bulunamadi, validation devam ediyor...")



    # ── 1. HALÜSİNASYON KONTROLÜ ─────────────────────────────
    # LLM uydurma kolon mu seçti?
    all_col_names = {col["column_name"] for col in all_cols}
    if current_target and current_target not in all_col_names:
        logger.warning(f"[feature_plan] ⚠️  LLM'in seçtiği '{current_target}' veritabanında YOK (halüsinasyon)!")
        if discovery_target and discovery_target in all_col_names:
            logger.info(f"[feature_plan] ✓  target_discovery_node'un geçerli '{discovery_target}' kolonu kullanılacak.")
            plan_dict["target"]["column_name"]     = discovery_target
            plan_dict["target"]["is_derived"]      = False
            plan_dict["target"]["derivation_type"] = "none"
            current_target = discovery_target
        else:
            logger.warning("[feature_plan] 🚨 Geçerli bir target bulunamadı!")
            return

    # ── 2. TARGET DISCOVERY OVERRIDE (yüksek confidence) ─────
    if discovery_target and confidence >= 0.5:
        if current_target != discovery_target:
            logger.warning(f"[feature_plan] ⚠️  LLM='{current_target}' ama target_discovery_node "
                  f"='{discovery_target}' (confidence={confidence:.2f}). Override.")
            plan_dict["target"]["column_name"]     = discovery_target
            plan_dict["target"]["is_derived"]      = False
            plan_dict["target"]["derivation_type"] = "none"
            current_target = discovery_target
        else:
            logger.info(f"[feature_plan] ✓  LLM ve target_discovery_node uyumlu: '{current_target}' "
                  f"(confidence={confidence:.2f})")

    # ── 3. SEMANTIC TYPE KONTROLÜ ────────────────────────────
    target_candidates = [
        col for col in all_cols
        if col.get("semantic_type") == "target"
    ]

    if not target_candidates:
        logger.info(f"[feature_plan] ℹ️  IE'de target-etiketli kolon yok. Mevcut seçim korunuyor: '{current_target}'")
        return

    target_col_names = {col["column_name"] for col in target_candidates}
    if current_target in target_col_names:
        logger.info(f"[feature_plan] ✓  Target geçerli: '{current_target}' (semantic_type=target)")
        return

    # ── 4. WORD OVERLAP FALLBACK ─────────────────────────────
    logger.warning(f"[feature_plan] ⚠️  '{current_target}' kolonu semantic_type=target değil. Düzeltiliyor...")
    best = _score_target_by_query_overlap(target_candidates, user_query)
    logger.info(f"[feature_plan] ✓  Target override: '{current_target}' → '{best['column_name']}' "
          f"(skor: {best['_score']:.3f})")

    plan_dict["target"]["column_name"]     = best["column_name"]
    plan_dict["target"]["is_derived"]      = False
    plan_dict["target"]["derivation_type"] = "none"


def _score_target_by_query_overlap(candidates: list[dict], user_query: str) -> dict:
    """
    Aday target kolonları kullanıcı sorgusuna göre skorlar.

    Skor = kullanıcı sorgusu ile (kolon_adı + açıklama + örnek_değerler) arasındaki
           word-overlap. Tamamen dinamik — hardcoded kelime yok.

    Args:
        candidates: db_intelligence'dan gelen, semantic_type='target' olan kolon listesi
        user_query: kullanıcının ham sorusu

    Returns:
        En yüksek skorlu kolon dict'i (+ '_score' anahtarı eklenerek)
    """
    # Normalize the query
    query_words = set(
        w.strip(".,!?;:").lower()
        for w in user_query.split()
        if len(w) > 2
    )

    scored = []
    for col in candidates:
        # Combine all column metadata
        col_text = " ".join([
            col.get("column_name", ""),
            col.get("description", ""),
            col.get("sample_values", ""),
            col.get("table_name", ""),
        ]).lower()

        col_words = set(
            w.strip(".,!?;:").lower()
            for w in col_text.split()
            if len(w) > 2
        )

        overlap  = len(query_words & col_words)
        # Bonus: more words in description give more signals
        desc_len = len(col.get("description", "")) / 200.0
        # Bonus: importance_score (IE's confidence)
        importance = col.get("importance_score", 0.5)

        score = overlap + desc_len + importance
        scored.append({**col, "_score": score})

    # Return the highest scoring candidate
    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[0]