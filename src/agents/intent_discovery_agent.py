from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.agents import create_agent
from langchain_core.tools import tool
from src.core.llm import get_llm
from src.prompts.intent_discovery_prompts import INTENT_DISCOVERY_SYSTEM_PROMPT
import json
from pydantic import BaseModel, Field

class IntentState(TypedDict):
    db_intelligence: dict
    schema_graph: dict
    signals: dict
    domain: str
    intents: list

llm = get_llm(model_name="gpt-4o-mini", temperature=0)

class SignalExtractionInput(BaseModel):
    core_entities: List[str] = Field(description="List of core business entities found (e.g. Users, Payments, Invoices, Products)")
    has_time_series: bool = Field(description="Whether the schema supports time series/temporal analysis")
    has_geospatial: bool = Field(description="Whether the schema supports geospatial/location analysis")
    has_financials: bool = Field(description="Whether the schema contains financial or monetary data")
    complexity_score: int = Field(description="Estimated complexity of the schema from 1 to 10")

@tool(args_schema=SignalExtractionInput)
def record_extracted_signals_tool(core_entities: List[str], has_time_series: bool, has_geospatial: bool, has_financials: bool, complexity_score: int) -> dict:
    """
    Saves the dynamically extracted data signals.
    The LLM MUST call this tool FIRST to record its understanding of the data capabilities.
    """
    return {
        "status": "success",
        "message": "Signals successfully recorded. Please proceed to identify the business domain."
    }


class DomainIdentificationInput(BaseModel):
    primary_domain: str = Field(description="The dynamically identified main business domain (e.g. E-commerce, Fintech, Healthcare, SaaS)")
    sub_domains: List[str] = Field(description="Any identified sub-domains (e.g. B2B, Logistics, Subscription)")
    reasoning: str = Field(description="The logical reasoning behind this domain identification based on the schema")

@tool(args_schema=DomainIdentificationInput)
def record_business_domain_tool(primary_domain: str, sub_domains: List[str], reasoning: str) -> dict:
    """
    Saves the dynamically identified business domain.
    The LLM MUST call this tool SECOND to lock in the business domain context.
    """
    return {
        "status": "success",
        "message": f"Domain '{primary_domain}' recorded. Now proceed to brainstorm and save analytical intents."
    }


class Intent(BaseModel):
    intent_name: str = Field(description="The name of the analytical intent (e.g. Churn Prediction)")
    intent_type: str = Field(description="Type of analytics (e.g. predictive, descriptive, prescriptive)")
    business_value: str = Field(description="Business value impact (High, Medium, Low)")
    description: str = Field(description="Detailed explanation of what this analysis does and why it is useful")
    complexity: str = Field(description="Implementation complexity (High, Medium, Low)")
    required_data: List[str] = Field(description="List of data tables or signals required for this intent")

class SaveIntentsInput(BaseModel):
    intents: List[Intent] = Field(description="The dynamic list of intents discovered by the LLM")

@tool(args_schema=SaveIntentsInput)
def save_discovered_intents_tool(intents: List[Intent]) -> dict:
    """
    Saves the dynamically generated analytical intents. 
    The LLM MUST call this tool as its FINAL STEP to record the discovered intents.
    """
    return {
        "status": "success", 
        "recorded_intents_count": len(intents),
        "recorded_intents": [i.dict() for i in intents]
    }

tools = [
    record_extracted_signals_tool,
    record_business_domain_tool,
    save_discovered_intents_tool
]

tool_node = ToolNode(tools)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=INTENT_DISCOVERY_SYSTEM_PROMPT
)


def intent_discovery_agent_node(state: IntentState) -> dict:
    """
    Full tool-based autonomous intent discovery agent.
    """
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": json.dumps({
                    "schema": state["schema_graph"],
                    "db_intelligence": state["db_intelligence"]
                })
            }
        ]
    })

    return {
        "intents": result["messages"][-1]["content"]
    }

