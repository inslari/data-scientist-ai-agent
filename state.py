from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):

    user_query: str                 
    intent: str                      
    intent_confidence: float         
    raw_schema: str                  
    db_intelligence: dict            
    relevant_tables: list[str]       
    analysis_plan: list[dict]        
    sql_results: list[dict]          
    profiling_report: dict          
    feature_report: dict             
    model_name: str                  
    model_params: dict               
    model_artifact: Any              
    metrics: dict                    
    insights: list[str]              
    report: str                     
    current_step: str                
    error: Optional[str]             
    messages: list                   