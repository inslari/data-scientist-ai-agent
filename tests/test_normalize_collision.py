"""Faz 0.5 — _normalize_column_names SESSİZ VERİ KAYBI regresyonu.

Kök bug: fallback auto-detect '_number'/'_score' gibi son ekleri tablo-suffix'i sanıyor;
`products` + `products_number` birlikte olunca `products_number` DEĞER KONTROLÜ OLMADAN
siliniyordu → sessiz veri kaybı (fraud'da en korktuğumuz tip: patlamıyor, yanlış sonuç).

Düzeltme: normalizasyon yalnız DEĞERİ-BİREBİR-AYNI kopyayı düşürür; farklı-değerli
kolonlar aynı ada düşerse ÇARPIŞMA → orijinal ad korunur + rapora taşınır. Fallback'te
spekülatif rename YOK. Sıra-bağımsız/deterministik. Dinamik: kolon adı hard-code yok.

Standalone + pytest uyumlu.
"""
import sys
import numpy as np
import pandas as pd

from src.services.feature_builder import _normalize_column_names

_fail = []


def check(cond, msg):
    print(("  OK  " if cond else "  XX  ") + msg)
    if not cond:
        _fail.append(msg)


def _df(cols_vals: dict) -> pd.DataFrame:
    return pd.DataFrame(cols_vals)


def test_a_fallback_collision_number_both_survive():
    print("\n[a] fallback: x + x_number FARKLI değerler → ikisi de hayatta")
    df = _df({"x": [1, 2, 3], "x_number": [9, 8, 7], "z": [0, 0, 1]})
    out, col = _normalize_column_names(df.copy(), table_names=None)
    check("x" in out.columns and "x_number" in out.columns, "x ve x_number ikisi de korundu (silme yok)")
    check(any(c["normalized_name"] == "x" for c in col), "çarpışma kanıtı üretildi (rapora taşınır)")


def test_b_fallback_collision_score_both_survive():
    print("\n[b] fallback: y + y_score FARKLI değerler → ikisi de hayatta")
    df = _df({"y": [1.0, 2.0, 3.0], "y_score": [0.5, 0.6, 0.7]})
    out, col = _normalize_column_names(df.copy(), table_names=None)
    check("y" in out.columns and "y_score" in out.columns, "y ve y_score ikisi de korundu")
    check(any(set(c["kept_columns"]) == {"y", "y_score"} for c in col), "kept_columns doğru raporlandı")


def test_c_explicit_true_duplicate_dropped():
    print("\n[c] explicit: is_fraud + is_fraud_transactions AYNI değer → suffixli düşer (dedup)")
    vals = [0, 1, 0, 1]
    df = _df({"is_fraud": vals, "is_fraud_transactions": list(vals), "amt": [1, 2, 3, 4]})
    out, col = _normalize_column_names(df.copy(), table_names=["transactions"])
    check("is_fraud" in out.columns and "is_fraud_transactions" not in out.columns,
          "değeri-aynı join-duplicate güvenle düşürüldü")
    check(col == [], "gerçek duplicate → çarpışma DEĞİL (rapora taşınmaz)")


def test_c2_explicit_collision_different_values_kept():
    print("\n[c2] explicit: is_fraud + is_fraud_transactions FARKLI değer → çarpışma, ikisi de kalır")
    df = _df({"is_fraud": [0, 1, 0, 1], "is_fraud_transactions": [1, 1, 1, 1], "amt": [1, 2, 3, 4]})
    out, col = _normalize_column_names(df.copy(), table_names=["transactions"])
    check("is_fraud" in out.columns and "is_fraud_transactions" in out.columns,
          "farklı-değerli kolonlar → sessiz silme YOK, ikisi de korundu")
    check(any(c["normalized_name"] == "is_fraud" for c in col), "explicit branch çarpışması da raporlandı")


def test_c3_dtype_difference_is_not_duplicate():
    print("\n[c3] explicit: aynı sayılar ama int64 vs float64 → 'aynı değil', ikisi de kalır (şüphe→koru)")
    df = pd.DataFrame({"v": pd.Series([1, 2, 3], dtype="int64"),
                       "v_transactions": pd.Series([1.0, 2.0, 3.0], dtype="float64")})
    out, col = _normalize_column_names(df.copy(), table_names=["transactions"])
    check("v" in out.columns and "v_transactions" in out.columns,
          "dtype farkı → duplicate sayılmadı, veri kaybı yok")


def test_d_explicit_lone_suffix_renamed():
    print("\n[d] explicit: lone merchant_name_merchants (merchant_name yok) → rename edilir")
    df = _df({"merchant_name_merchants": ["a", "b"], "amt": [1, 2]})
    out, col = _normalize_column_names(df.copy(), table_names=["merchants"])
    check("merchant_name" in out.columns and "merchant_name_merchants" not in out.columns,
          "explicit tablo-suffix'li lone kolon güvenle rename edildi")
    check(col == [], "rename → çarpışma değil")


def test_e_fallback_lone_no_speculative_rename():
    print("\n[e] fallback: lone products_number (products yok) → KORUNUR (spekülatif rename yok)")
    df = _df({"products_number": [1, 2, 3], "tenure": [4, 5, 6]})
    out, col = _normalize_column_names(df.copy(), table_names=None)
    check("products_number" in out.columns and "products" not in out.columns,
          "fallback lone kolon mangle EDİLMEDİ (products_number korundu)")
    check(col == [], "spekülatif rename yok → çarpışma yok")


def test_f_multi_column_collision():
    print("\n[f] çoklu çarpışma: a + a_number + a_score hepsi FARKLI → üçü de hayatta")
    df = _df({"a": [1, 2, 3], "a_number": [4, 5, 6], "a_score": [7, 8, 9]})
    # explicit iki suffix ile de dene (number, score gerçek 'tablo' gibi)
    out, col = _normalize_column_names(df.copy(), table_names=["number", "score"])
    for c in ("a", "a_number", "a_score"):
        check(c in out.columns, f"{c} korundu (3'lü çarpışmada sessiz kayıp yok)")
    kept = next((c["kept_columns"] for c in col if c["normalized_name"] == "a"), [])
    check(set(kept) == {"a", "a_number", "a_score"}, "3 kolon da çarpışma kanıtında listelendi")


def test_g_order_independence():
    print("\n[g] sıra-bağımsızlık: aynı veri, farklı kolon sırası → aynı çıktı")
    base = {"a": [1, 2, 3], "a_number": [4, 5, 6], "b": [7, 8, 9], "b_transactions": [7, 8, 9]}
    df1 = pd.DataFrame(base)
    df2 = pd.DataFrame({k: base[k] for k in reversed(list(base))})   # ters kolon sırası
    out1, col1 = _normalize_column_names(df1.copy(), table_names=["number", "transactions"])
    out2, col2 = _normalize_column_names(df2.copy(), table_names=["number", "transactions"])
    check(set(out1.columns) == set(out2.columns),
          f"hayatta kalan kolon kümesi sıradan bağımsız: {sorted(out1.columns)}")
    norm = lambda cc: sorted((c["normalized_name"], tuple(c["kept_columns"]), tuple(c["dropped_duplicates"])) for c in cc)
    check(norm(col1) == norm(col2), "çarpışma kayıtları sıradan bağımsız (deterministik)")
    # b_transactions değeri b ile AYNI → dedup; a_number farklı → çarpışma; her iki sırada da böyle
    check("b_transactions" not in out1.columns, "değeri-aynı b_transactions dedup edildi (sıradan bağımsız)")


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("\n" + "=" * 60)
    if _fail:
        print(f"FAIL ({len(_fail)}):")
        for m in _fail:
            print("  - " + m)
        sys.exit(1)
    print("TÜM TESTLER GEÇTİ ✓")
