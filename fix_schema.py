import re

schema_content = """from pydantic import BaseModel, Field
from typing import Optional

class SQLGenerationResult(BaseModel):
    model_config = {"extra": "forbid"}
    sql: str = Field(description="SQL sorgusu")
    db_name: str = Field(description="Sorgunun çalıştırılacağı veritabanı adı")
    data_warning: Optional[str] = Field(default="", description="Uygun kolon bulunamadığında verilecek uyarı mesajı")
    is_approximation: bool = Field(default=False, description="Sorgu tam eşleşme değilse true, tam eşleşmeyse false")
    approximation_warning: Optional[str] = Field(default="", description="Yaklaşık bir eşleşme yapıldıysa verilecek uyarı mesajı")

class ChartConfig(BaseModel):
    model_config = {"extra": "forbid"}
    type: str = Field(description="Grafik türü")
    x: str = Field(description="X ekseni")
    y: str = Field(description="Y ekseni")
    title: str = Field(description="Grafik başlığı")
"""

with open("src/schemas/sql_schemas.py", "w") as f:
    f.write(schema_content)

with open("src/nodes/sql_node.py", "r") as f:
    sql_node_content = f.read()

# Restore original field access
old_access = """        sql                  = sql_result.query.strip()
        data_warning         = sql_result.explanation or ""
        is_approximation     = False
        approximation_warning = \"\""""
new_access = """        sql                  = sql_result.sql.strip()
        target_db_name       = str(sql_result.db_name).strip().strip('",\\'”’ \\t\\n')
        data_warning         = sql_result.data_warning or ""
        is_approximation     = sql_result.is_approximation
        approximation_warning = sql_result.approximation_warning or \"\""""
sql_node_content = sql_node_content.replace(old_access, new_access)

with open("src/nodes/sql_node.py", "w") as f:
    f.write(sql_node_content)
