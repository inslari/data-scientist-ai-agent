import os
import glob
import re

nodes_dir = '/Users/mustafakemalgordesli/Desktop/ds-agent/data-scientist-ai-agent/src/nodes'

for filepath in glob.glob(os.path.join(nodes_dir, '*_node.py')):
    with open(filepath, 'r') as f:
        content = f.read()
    
    filename = os.path.basename(filepath)
    node_name = filename.replace('.py', '')
    
    # We want to add a start print after the docstring or right after the def node_name(state)
    # Using regex to find the def signature
    pattern = r'(def ' + node_name + r'\(state:?.*?\).*?:)\s*(?:"""[\s\S]*?""")?'
    
    match = re.search(pattern, content)
    if match:
        start_idx = match.end()
        # Find the indentation of the next line
        lines = content[start_idx:].split('\n')
        indent = '    '
        for line in lines:
            if line.strip():
                indent_match = re.match(r'^\s+', line)
                if indent_match:
                    indent = indent_match.group(0)
                break
        
        # Check if already printed
        if f"[NODE START] {node_name}" not in content:
            insert_str = f'\n{indent}print("\\n{'='*40}\\n[NODE START] {node_name} başlıyor...\\n{'='*40}")'
            content = content[:start_idx] + insert_str + content[start_idx:]
            
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Added start print to {filename}")
        else:
            print(f"Already has start print: {filename}")

