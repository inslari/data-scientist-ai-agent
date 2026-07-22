"""Offline deterministik rapor harness'ı — LLM/worker/DB olmadan raporun deterministik
katmanını (headline→explanation→evidence) render eder. 20 fazlık rapor denetimi için
sürdürülebilir doğrulama aracı (ad-hoc harness yerine).

Zincir (GERÇEK servisler — 20 fazın değiştireceği kod yollarını doğrular):
    CSV → build_features → train_model → evaluate_model → run_data_quality
        → build_*_section (deterministik, LLM prose YOK) → render_report (TR + EN)

Kullanım:
    uv run python scripts/offline_report.py --csv PATH --target COL [--intent classification]
        [--full] [--out DIR] [--models m1,m2]

    --full   : tam 14-model + blend (rapora birebir). Varsayılan: hızlı alt-küme (2 model).
    Çıktı    : <out>/report_<lang>.md (TR + EN), fazlar arası `diff` için. Bölüm order
               değerleri başlıklarda görünür (fairness'ın order=50'ye düşmesi gibi bug'lar
               ancak böyle yakalanır).

NOT: LLM narrative (build_section_prose / build_synthesis_prose) ATLANIR — yalnız
deterministik kanıt katmanı. Mutlak metrikler alt-kümede rapordan sapar (model gücü);
YAPI sapmamalı. --full ile rapora birebir yakınsar.
"""
from __future__ import annotations
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
# Remote app DB'ye ASLA dokunma (CLAUDE.md): sandbox sqlite'a sabitle.
os.environ.setdefault("API_DATABASE_URL", "sqlite:///./data/_offline_report_sandbox.sqlite")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.feature_builder import build_features
from src.services.ml_trainer import train_model
from src.services.evaluator import evaluate_model
from src.services.data_quality import run_data_quality
from src.services.report_sections import (
    build_data_quality_section, build_relations_section, build_performance_section,
    build_generalization_section, build_error_analysis_section, build_features_section,
    build_leakage_section, build_fairness_section, build_summary_section,
    build_assessment_section, build_recommendations_section,
)
from src.core.report_schema import render_section

# report_generator.py:713-736 montaj sırasıyla BİREBİR aynı (prose wrapper hariç).
_EVIDENCE_BUILDERS = [
    build_data_quality_section, build_relations_section, build_performance_section,
    build_generalization_section, build_error_analysis_section, build_features_section,
    build_leakage_section, build_fairness_section,
]
_SYNTHESIS_BUILDERS = [build_summary_section, build_assessment_section, build_recommendations_section]


def _infer_semantic_type(df: pd.DataFrame, col: str, target: str) -> str:
    """Dinamik semantic_type çıkarımı — kolon adı hard-code YOK.
    target → target; near-unique (id-benzeri) → identifier; object → category; sayısal → metric.
    """
    if col == target:
        return "target"
    n = len(df)
    nunique = df[col].nunique(dropna=True)
    if n > 0 and nunique / n > 0.95 and not pd.api.types.is_float_dtype(df[col]):
        return "identifier"
    if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
        return "category"
    return "metric"


def _build_profiling(df: pd.DataFrame) -> dict:
    """Minimal profiling_report (DuckDB profiling muadili) — ColumnMetadata + data_quality için."""
    cols = {}
    n = len(df)
    for c in df.columns:
        s = df[c]
        nunique = int(s.nunique(dropna=True))
        null_rate = float(s.isna().mean())
        q1, q3 = (s.quantile(0.25), s.quantile(0.75)) if pd.api.types.is_numeric_dtype(s) else (0, 0)
        iqr = q3 - q1
        has_out = bool(pd.api.types.is_numeric_dtype(s) and iqr and
                       ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).any())
        cols[c] = {
            "dtype": str(s.dtype), "null_rate": round(null_rate, 4),
            "unique_count": nunique, "unique_rate": round(nunique / n, 4) if n else 0.0,
            "has_outliers": has_out,
        }
    return {"columns": cols, "shape": {"rows": n, "cols": len(df.columns)}}


def build_state(csv: str, target: str, intent: str, full: bool, models: list[str] | None,
                nrows: int | None = None) -> dict:
    df = pd.read_csv(csv, nrows=nrows) if nrows else pd.read_csv(csv)
    if target not in df.columns:
        raise SystemExit(f"target '{target}' CSV'de yok. Kolonlar: {list(df.columns)}")
    db_intel = {
        "columns": [{"column_name": c, "semantic_type": _infer_semantic_type(df, c, target),
                     "importance_score": 0.5} for c in df.columns],
        "tables": [],
    }
    state: dict = {
        "intent": intent,
        "user_query": f"offline report: {os.path.basename(csv)}",
        "sql_results": df.to_dict(orient="records"),
        "feature_plan": {"target": {"column_name": target}, "analysis_summary": f"offline {intent}"},
        "target_definition": {"column_name": target, "confirmed": True},
        "target_col": target,
        "db_intelligence": db_intel,
        "profiling_report": _build_profiling(df),
        "report_language": "tr",
        "project_id": "offline_report",
    }
    if not full:
        # Hızlı alt-küme: top_models model setini kısıtlar (ml_trainer:1060).
        state["top_models"] = models or (
            ["logistic_regression", "random_forest"] if intent == "classification"
            else ["logistic_regression", "random_forest"])
    return state


def run_pipeline(state: dict) -> dict:
    for name, fn in [("build_features", build_features), ("data_quality", run_data_quality),
                     ("train_model", train_model), ("evaluate_model", evaluate_model)]:
        out = fn(state) or {}
        if out.get("error"):
            raise SystemExit(f"[{name}] HATA: {out['error']}")
        state.update(out)
    return state


def render(state: dict, lang: str) -> str:
    state = {**state, "report_language": lang}
    lines = [f"# OFFLINE DETERMINISTIC REPORT ({lang.upper()})",
             f"_model={state.get('model_name','?')} · intent={state.get('intent')} · "
             f"n_features={state.get('feature_report',{}).get('n_features','?')}_\n"]
    sections = []
    for b in _EVIDENCE_BUILDERS:
        try:
            sections.append(b(state, lang))
        except Exception as e:
            lines.append(f"\n> ⚠️ {b.__name__} RENDER HATASI: {type(e).__name__}: {e}\n")
    for b in _SYNTHESIS_BUILDERS:
        try:
            sections.append(b(state))
        except Exception as e:
            lines.append(f"\n> ⚠️ {b.__name__} RENDER HATASI: {type(e).__name__}: {e}\n")
    for sec in sorted([s for s in sections if s is not None], key=lambda s: getattr(s, "order", 999)):
        lines.append(f"\n<!-- order={getattr(sec,'order','?')} id={getattr(sec,'id','?')} -->")
        lines.append(render_section(sec, lang))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--intent", default="classification")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--models", default="")
    ap.add_argument("--out", default="data/_offline_reports")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()] or None
    state = build_state(args.csv, args.target, args.intent, args.full, models)
    state = run_pipeline(state)

    for lang in ("tr", "en"):
        text = render(state, lang)
        path = os.path.join(args.out, f"report_{lang}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[{lang}] yazıldı: {path}  ({len(text.splitlines())} satır)")
    print(f"model={state.get('model_name')}  metrics={ {k: round(v,4) for k,v in (state.get('metrics') or {}).items() if isinstance(v,(int,float))} }")

    # #11(a) AUDIT: öneriler sentezine giden DIGEST'in importance'ı GÖRMEDİĞİNİ yapısal
    # olarak kanıtla (LLM'siz — recs prose'u harness'ta üretilmiyor). full digest ile
    # recs-scoped digest'i yan yana yaz; importance full'de VAR, recs'te YOK olmalı.
    from src.services.report_sections import _build_digest
    _secs = []
    for _b in _EVIDENCE_BUILDERS:
        try:
            _secs.append(_b({**state, "report_language": "tr"}, "tr"))
        except Exception:
            pass
    _secs = [s for s in _secs if s is not None]
    _full = _build_digest(_secs, "tr")
    _recs = _build_digest(_secs, "tr", exclude_ids=frozenset({"features"}),
                          causal_note_ids=frozenset({"error_analysis"}))
    with open(os.path.join(args.out, "recs_digest.md"), "w", encoding="utf-8") as f:
        f.write("# #11(a) recs-digest audit\n\n## FULL digest (summary/assessment görür)\n\n"
                + _full + "\n\n## RECS digest (importance KESİLDİ, error_analysis damgalı)\n\n" + _recs)
    print(f"recs-digest dump: {os.path.join(args.out, 'recs_digest.md')}")


if __name__ == "__main__":
    main()
