import os
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate

def create_chat_prompt(system_filename: str, human_template: str) -> ChatPromptTemplate:
    prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
    sys_path = prompts_dir / system_filename
    
    if sys_path.exists():
        with open(sys_path, "r", encoding="utf-8") as f:
            system_msg = f.read()
    else:
        system_msg = "You are a helpful data science AI assistant."
    
    # If the user provided a filename for human_template too
    if human_template.endswith(".md"):
        human_path = prompts_dir / human_template
        if human_path.exists():
            with open(human_path, "r", encoding="utf-8") as f:
                human_msg = f.read()
        else:
            human_msg = human_template
    else:
        human_msg = human_template

    return ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", human_msg)
    ])
