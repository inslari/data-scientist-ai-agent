"""
Text-analysis feature tests (pytest).

Run against a LOCAL sqlite sandbox — never the remote app DB (see CLAUDE.md):
    API_DATABASE_URL=sqlite:////tmp/ta_test.db uv run pytest tests/test_text_analysis.py

LLM/embedding calls are stubbed; no API key required.
"""
import os
# Sandbox DB must be set before importing anything that builds the engine.
os.environ.setdefault("API_DATABASE_URL", "sqlite:////tmp/ta_test.db")

# CRITICAL: bind the application DB engine to the sandbox NOW, before any module
# (e.g. src.core.config) runs load_dotenv(override=True) and clobbers the env var
# with the remote Postgres URL from .env. database.py uses override=False, so the
# engine captures our sqlite URL at this point. (See CLAUDE.md DB-safety note.)
import api.core.database  # noqa: E402,F401

assert api.core.database.SQLALCHEMY_DATABASE_URL.startswith("sqlite"), (
    "Refusing to run: app DB is not a local sqlite sandbox "
    f"({api.core.database.SQLALCHEMY_DATABASE_URL.split('://')[0]})"
)

import asyncio  # noqa: E402

import pytest  # noqa: E402


# ─────────────────────────── engine (deterministic) ───────────────────────────

def test_clean_texts_filters_and_normalizes():
    from src.services.text_analysis_engine import clean_texts
    out = clean_texts(["  çok   boşluk  ", "", "x", None, "tamam"])  # type: ignore[list-item]
    assert out == ["çok boşluk", "tamam"]  # tek karakter 'x' elenir


def test_distribution_counts():
    from src.services.text_analysis_engine import distribution
    assert distribution(["a", "a", "b", None, ""]) == {"a": 2, "b": 1}


def test_cluster_texts_tfidf_fallback(monkeypatch):
    import src.services.text_analysis_engine as eng
    monkeypatch.setattr(eng, "_embed_texts", lambda texts: None)  # embedding kapalı → TF-IDF
    texts = ["kargo gecikti", "teslimat gecikti", "uygulama hata veriyor",
             "login ekranı hata", "ürün harika", "fiyat çok pahalı"]
    clusters = eng.cluster_texts(texts)
    assert clusters
    # Tüm metinler bir kümeye atanmış olmalı (kayıp yok).
    assert sum(len(c["member_indices"]) for c in clusters) == len(texts)
    assert all("top_terms" in c for c in clusters)


def test_build_priority_items_orders_by_urgency():
    from src.services.text_analysis_engine import build_priority_items
    items = [
        {"category": "praise", "urgency": "low", "sentiment_score": 0.9},
        {"category": "complaint", "urgency": "critical", "sentiment_score": -0.9},
        {"category": "complaint", "urgency": "medium", "sentiment_score": -0.3},
    ]
    pr = build_priority_items(items)
    assert pr[0]["urgency"] == "critical"
    # Övgü (praise/low) önceliklendirmeye girmez.
    assert all(p["category"] in ("complaint", "request") or p["urgency"] in ("high", "critical") for p in pr)


# ─────────────────────────── classify node (stubbed LLM) ──────────────────────

def _stub_classify(monkeypatch):
    from src.graphs.textanalysis.nodes import classify_node as cn
    from src.graphs.textanalysis.state import BatchClassification, ItemClassification, Aspect

    def fake(prompt, schema, kwargs):
        lines = [l for l in kwargs["numbered_texts"].splitlines() if l.strip()]
        items = []
        for i, line in enumerate(lines):
            neg = any(w in line.lower() for w in ("kızgın", "gecik", "hata"))
            items.append(ItemClassification(
                index=i,
                sentiment="negative" if neg else "positive",
                sentiment_score=-0.8 if neg else 0.7,
                emotion="anger" if neg else "satisfaction",
                category="complaint" if neg else "praise",
                urgency="high" if neg else "low",
                summary="özet",
                aspects=[Aspect(aspect="teslimat", sentiment="negative")] if neg else [],
            ))
        return BatchClassification(items=items)

    monkeypatch.setattr(cn, "structured_invoke", fake)
    monkeypatch.setattr(cn, "get_cached", lambda kind, model, texts: {})
    monkeypatch.setattr(cn, "set_cached", lambda kind, model, items: None)
    return cn


def test_classify_node_alignment_dedup_and_aspects(monkeypatch):
    cn = _stub_classify(monkeypatch)
    texts = ["kargo gecikti", "ürün harika", "kargo gecikti"]  # 3. = 1. tekrarı
    out = asyncio.run(cn.ta_classify_node({"texts": texts}))

    assert len(out["item_results"]) == 3
    assert [r["index"] for r in out["item_results"]] == [0, 1, 2]
    assert out["sentiment_distribution"]["negative"] == 2
    assert out["sentiment_distribution"]["positive"] == 1
    assert out["category_distribution"]["complaint"] == 2
    assert "teslimat" in out["aspect_distribution"]
    assert out["aspect_distribution"]["teslimat"]["negative"] == 2


# ─────────────────────────── full graph (stubbed) ─────────────────────────────

def test_full_graph_runs(monkeypatch):
    cn = _stub_classify(monkeypatch)
    from src.graphs.textanalysis.nodes import theme_node as tn
    from src.graphs.textanalysis.nodes import report_node as rn
    from src.graphs.textanalysis.nodes import detect_timestamp_node as dn
    from src.graphs.textanalysis.state import ThemeLabel, TimestampDetection

    monkeypatch.setattr(tn, "structured_invoke", lambda p, s, k: ThemeLabel(label="Tema", description="açıklama"))
    monkeypatch.setattr(dn, "structured_invoke", lambda p, s, k: TimestampDetection(has_timestamp=False))
    # Raporu LLM'siz fallback'e zorla.
    monkeypatch.setattr(rn, "_get_llm", lambda temperature=0: (_ for _ in ()).throw(RuntimeError("no-llm")))

    from src.graphs.textanalysis.graph import build_text_analysis_graph
    g = build_text_analysis_graph()
    texts = ["kargo gecikti", "uygulama hata", "ürün harika", "fiyat pahalı", "teslimat gecikti"]
    state = asyncio.run(g.ainvoke({"title": "T", "raw_texts": texts, "language": "tr"}))

    assert state.get("error") is None
    assert len(state["item_results"]) == len(texts)
    assert state["themes"]
    assert state.get("report")
    assert "aspect_distribution" in state


def test_build_time_trend_buckets_by_date():
    from src.services.text_analysis_engine import build_time_trend
    items = [
        {"timestamp": "2026-01-01", "sentiment": "negative", "sentiment_score": -0.8},
        {"timestamp": "2026-01-01", "sentiment": "negative", "sentiment_score": -0.6},
        {"timestamp": "2026-01-15", "sentiment": "positive", "sentiment_score": 0.7},
        {"timestamp": "2026-02-01", "sentiment": "neutral", "sentiment_score": 0.0},
        {"timestamp": "", "sentiment": "positive", "sentiment_score": 0.9},  # tarihsiz → atlanır
    ]
    trend = build_time_trend(items)
    assert len(trend) >= 2
    assert sum(p["count"] for p in trend) == 4  # tarihsiz olan sayılmaz
    assert all({"period", "count", "avg_sentiment", "positive", "negative", "neutral"} <= set(p) for p in trend)


def test_build_time_trend_empty_without_dates():
    from src.services.text_analysis_engine import build_time_trend
    assert build_time_trend([{"timestamp": "", "sentiment": "positive", "sentiment_score": 1.0}]) == []


def test_silhouette_auto_k(monkeypatch):
    import src.services.text_analysis_engine as eng
    monkeypatch.setattr(eng, "_embed_texts", lambda texts: None)  # TF-IDF üstünde silhouette
    texts = [
        "kargo gecikti çok bekledim", "teslimat gecikmesi yaşadım", "paket geç geldi",
        "uygulama açılmıyor hata veriyor", "login olamıyorum hata", "uygulama çöküyor sürekli",
        "ürün kalitesi mükemmel", "çok memnun kaldım harika", "kaliteli ürün teşekkürler",
        "fiyatlar çok pahalı",
    ]
    clusters = eng.cluster_texts(texts)  # n_clusters=None → silhouette yolu
    assert 2 <= len(clusters) <= 8
    assert sum(len(c["member_indices"]) for c in clusters) == len(texts)


def test_graph_empty_input_errors_gracefully():
    from src.graphs.textanalysis.graph import build_text_analysis_graph
    g = build_text_analysis_graph()
    state = asyncio.run(g.ainvoke({"title": "Boş", "raw_texts": ["", "  "], "language": "tr"}))
    # preprocess hata set eder, report_node (runs_on_error) düzgün rapor üretir.
    assert state.get("error")
    assert "Tamamlanamadı" in (state.get("report") or "")


# ─────────────────────────── cache + worker lanes (DB) ────────────────────────

@pytest.fixture()
def db_tables():
    from api.core.database import Base, engine
    import api.models.text_analysis  # noqa: F401
    import api.models.text_cache  # noqa: F401
    import api.models.job  # noqa: F401
    import api.models.project  # noqa: F401
    import api.models.user  # noqa: F401
    import api.models.data_group  # noqa: F401
    import api.models.csv  # noqa: F401
    import api.models.mcp  # noqa: F401
    import api.models.db_intelligence_cache  # noqa: F401
    import api.models.log_entry  # noqa: F401
    Base.metadata.create_all(bind=engine)
    yield


def test_cache_roundtrip(db_tables):
    from src.graphs.textanalysis.cache import get_cached, set_cached
    payloads = {"merhaba": {"sentiment": "positive"}, "kötü": {"sentiment": "negative"}}
    set_cached("classify", "test:model", payloads)
    got = get_cached("classify", "test:model", ["merhaba", "kötü", "yok"])
    assert got["merhaba"]["sentiment"] == "positive"
    assert got["kötü"]["sentiment"] == "negative"
    assert "yok" not in got


def test_worker_lane_recovery_isolation(db_tables):
    import time
    from api.core.database import SessionLocal
    from api.models.job import BackgroundJobModel
    from api.models.project import ProjectModel
    from api.models.text_analysis import TextAnalysisModel
    from api.core.worker import recover_interrupted_jobs, TEXT_ANALYSIS_TASK_TYPES

    db = SessionLocal()
    try:
        proj = ProjectModel(title="p", status="running", created_at=int(time.time()), updated_at=int(time.time()))
        ta = TextAnalysisModel(title="t", source_texts=["a"], status="running",
                               created_at=int(time.time()), updated_at=int(time.time()))
        db.add_all([proj, ta])
        db.commit()
        db.refresh(proj); db.refresh(ta)
        db.add_all([
            BackgroundJobModel(task_type="run_project", project_id=proj.id, status="running"),
            BackgroundJobModel(task_type="run_text_analysis", text_analysis_id=ta.id, status="running"),
        ])
        db.commit()
        proj_id, ta_id = proj.id, ta.id
    finally:
        db.close()

    # Sadece text lane'ini kurtar.
    n = recover_interrupted_jobs(task_types=TEXT_ANALYSIS_TASK_TYPES)
    assert n == 1

    db = SessionLocal()
    try:
        proj = db.query(ProjectModel).get(proj_id)
        ta = db.query(TextAnalysisModel).get(ta_id)
        # Text analizi kurtarıldı (queued), proje DOKUNULMADAN running kaldı.
        assert ta.status == "queued"
        assert proj.status == "running"
    finally:
        db.close()
