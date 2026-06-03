import re

with open("api/routes/project.py", "r") as f:
    lines = f.readlines()

new_lines = []
in_async_for = False
for line in lines:
    if "async for chunk in app.astream" in line:
        in_async_for = True
        new_lines.append(line)
        continue
    
    if in_async_for:
        # Check if we are still inside the run_project_graph_task function
        # The function ends when we hit a line that is dedented back to column 0 or 4
        # Wait, the try block ends at `except Exception as e:` which is 12 spaces in?
        # Actually it's easier to just indent lines 45 to the end of the try block by 4 spaces.
        pass

# Let's just use `black` or something? No, black doesn't fix syntax errors.
