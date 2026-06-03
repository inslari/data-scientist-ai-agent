import json

def find_np(d, path=""):
    if isinstance(d, dict):
        for k, v in d.items():
            find_np(v, path + f".{k}")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            find_np(v, path + f"[{i}]")
    else:
        import numpy as np
        if isinstance(d, (np.floating, np.integer, np.ndarray)):
            print(f"Found numpy type at {path}: {type(d)}")

print("Done")
