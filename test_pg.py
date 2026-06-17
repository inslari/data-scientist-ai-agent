"""
FAZ 5 Dogrulama Testi
=====================
Cramer's V (correlation_full) + histogram bin index dogrulamasi.
"""
import sys
sys.path.insert(0, ".")

from src.services.analysis_engine import run_deep_analysis, ANALYSIS_MODULES
import pandas as pd
import numpy as np

np.random.seed(42)
n = 500
df = pd.DataFrame({
    "age": np.random.randint(18, 65, n),
    "income": np.random.normal(50000, 15000, n),
    "tenure": np.random.randint(1, 72, n),
    "monthly_charge": np.random.uniform(20, 100, n),
    "gender": np.random.choice(["Male", "Female"], n),
    "contract": np.random.choice(["Month", "OneYear", "TwoYear"], n, p=[0.5, 0.3, 0.2]),
    "churn": np.random.choice(["Yes", "No"], n, p=[0.3, 0.7]),
})

profile = {
    "shape": {"rows": n, "columns": 7},
    "summary": {
        "numeric_columns": ["age", "income", "tenure", "monthly_charge"],
        "categorical_columns": ["gender", "contract", "churn"],
        "datetime_columns": [],
        "missing_rate": 0.0,
    },
    "columns": {},
}

print("=== run_deep_analysis calistiriliyor ===")
result = run_deep_analysis(df, profile, target_col="churn")

# Geriye donuk uyumluluk: correlation_matrix (sadece sayisal)
print("\n=== FAZ 5a: correlation_matrix (geriye donuk) ===")
cm = result.get("correlation_matrix")
if not cm:
    print("FAIL: correlation_matrix yok!")
else:
    print(f"columns: {cm.get('columns')}")
    print(f"matrix boyutu: {len(cm.get('matrix', []))}x{len(cm.get('matrix', [[]])[0])}")
    print(f"Sadece sayisal: {'OK' if len(cm['columns']) == 4 else 'FAIL'}")

# correlation_full (sayisal + kategorik)
print("\n=== FAZ 5a: correlation_full (birlesik) ===")
cf = result.get("correlation_full")
if not cf:
    print("FAIL: correlation_full yok!")
else:
    all_cols = cf.get("columns", [])
    matrix = cf.get("matrix", [])
    print(f"columns: {all_cols}")
    print(f"matrix boyutu: {len(matrix)}x{len(matrix[0]) if matrix else 0}")
    print(f"Sayisal + kategorik: {'OK' if len(all_cols) == 7 else 'FAIL (expected 7, got ' + str(len(all_cols)) + ')'}")
    
    # Kategorik sutunlar dahil mi?
    has_gender = "gender" in all_cols
    has_contract = "contract" in all_cols
    has_churn = "churn" in all_cols
    print(f"gender dahil: {'OK' if has_gender else 'FAIL'}")
    print(f"contract dahil: {'OK' if has_contract else 'FAIL'}")
    print(f"churn dahil: {'OK' if has_churn else 'FAIL'}")
    
    # Diagonal = 1.0 olmali
    diag_ok = all(abs(matrix[i][i] - 1.0) < 0.01 for i in range(len(all_cols)))
    print(f"Diagonal = 1.0: {'OK' if diag_ok else 'FAIL'}")
    
    # top_pairs
    tp = cf.get("top_pairs", [])
    print(f"top_pairs sayisi: {len(tp)}")
    if tp:
        print(f"  En guclu: {tp[0]}")

# FAZ 5b: histogram bin_indices
print("\n=== FAZ 5b: histogram bin_indices ===")
hist = result.get("histograms", {})
if not hist:
    print("FAIL: histograms yok!")
else:
    first_col = list(hist.keys())[0]
    h = hist[first_col]
    counts = h.get("counts", [])
    indices = h.get("bin_indices", [])
    print(f"counts type: {type(counts[0]) if counts else 'empty'} (beklenen: int)")
    print(f"counts[0]: {counts[0] if counts else 'N/A'}")
    print(f"bin_indices: {indices[:5]}... (len={len(indices)})")
    print(f"Geriye donuk uyumlu (int array): {'OK' if isinstance(counts[0], int) else 'FAIL'}")
    print(f"bin_indices mevcut: {'OK' if indices else 'FAIL'}")

# Diger key'ler saglamlik kontrolu
print("\n=== Diger key'ler ===")
for key in ["target_relations", "clustering", "target_groups", "pairwise", "kde", "ecdf"]:
    print(f"  {key}: {'OK' if result.get(key) else 'FAIL (veya None/bos)'}")

print("\n=== FAZ 5 TESTI TAMAMLANDI ===")
