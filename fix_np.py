import os
import glob
import re

def add_cleaner(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    cleaner_func = """
def _clean_np(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: _clean_np(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_np(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_clean_np(v) for v in obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return _clean_np(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj
"""
    if "_clean_np" not in content:
        # Find the last import line to insert after it
        import_pattern = re.compile(r'^(import|from)\s+.*', re.MULTILINE)
        matches = list(import_pattern.finditer(content))
        if matches:
            last_match = matches[-1]
            insert_pos = last_match.end()
            content = content[:insert_pos] + "\n" + cleaner_func + content[insert_pos:]
        
        # Replace return { ... } with return _clean_np({ ... }) at the end of node
        # Actually it's easier to just clean the specific keys we know are problematic
        pass
