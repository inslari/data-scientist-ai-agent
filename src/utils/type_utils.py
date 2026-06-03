import numpy as np

def clean_np(obj):
    """
    Recursively converts numpy types in dictionaries and lists to standard Python types.
    This prevents LangGraph msgpack/json serialization errors.
    """
    if isinstance(obj, dict):
        return {k: clean_np(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_np(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(clean_np(v) for v in obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return clean_np(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj
