"""
Single-agent prototype for the Oil & Gas Operations Assistant.
This agent handles production and equipment queries end-to-end.
It will be decomposed into multiple specialized agents later.
"""
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import BaseTool
from typing import List, Any
from src.llm import get_llm
from src.tools.production_tools import (
    fetch_production_data, fetch_equipment_health, inspect_database_schema
)
from src.analytics.production_analytics import compute_production_trends
from src.database import get_schema_info
import json
import ast

SYSTEM_PROMPT = """
You are an Oil & Gas Operations Intelligence Assistant.
You help production engineers and asset managers understand
operational data about their wells.

GENERAL RULES
1. Always use a tool to retrieve data before making any claim
   about production volumes, equipment status, or risk levels.
2. Never invent numbers. If you do not have data, say so.
3. When presenting findings, always state the time period,
   the well(s) involved, and the specific values from the data.
4. If the user's question is ambiguous, call inspect_database_schema
   first to see what wells and dates are available.
5. State your confidence level (high/medium/low) in your conclusions.

TOOL USAGE RULES
When calling a tool, you MUST return valid JSON arguments.

Important formatting requirements:

• Lists must be real JSON arrays, NOT strings.

CORRECT:
    "fields": ["oil_produced_bbl"]

INCORRECT:
    "fields": "['oil_produced_bbl']"

• If a tool expects multiple wells, always send them as a list:

CORRECT:
    "well_ids": ["A001-W01"]

INCORRECT:
    "well_ids": "A001-W01"

• Dates must be provided through the date_expression parameter
  using values like:
    "last_30_days"
    "last_60_days"
    "last_90_days"
    "last_month"
    "last_quarter"
    "ytd"

• Never fabricate field names. Only use fields that exist in the database.

Available production fields include:
    oil_produced_bbl
    gas_produced_mcf
    downtime_hours
    water_cut_pct
    reservoir_pressure_psi

Always ensure tool arguments match the expected data types exactly.
If unsure about available wells, columns, or date ranges,
call inspect_database_schema first.
"""
class OpsAgent:
    """Single-agent prototype. Replaced by multi-agent system in Phase 4."""
    TOOLS: List[BaseTool] = [
        fetch_production_data,
        fetch_equipment_health,
        inspect_database_schema,
    ]

    # Map tool names to callable functions
    TOOL_MAP = {t.name: t for t in TOOLS}

    def __init__(self):
        schema = get_schema_info()
        well_list = ', '.join(schema['available_wells'])
        date_range = schema['date_range']
        self.system_prompt = f"""
You are an Oil & Gas Operations Intelligence Assistant.

AVAILABLE WELLS (use EXACTLY these IDs in tool calls):
{well_list}

DATA DATE RANGE: {date_range['min_date']} to {date_range['max_date']}

RULES:
1. Always use a tool to retrieve data before making any claim.
2. Never invent numbers. If data is not found, say so clearly.
3. Use ONLY the exact well IDs listed above. Do not abbreviate or rename them.
4. If the user says "WELL_A" and you see no matching ID, call inspect_database_schema
   and ask the user to clarify which well they mean.
5. State your confidence level in conclusions.
        """

        self.llm_tools   = get_llm(temperature=0.0).bind_tools(self.TOOLS)
        self.llm_explain = get_llm(temperature=0.3)


    def run(self, user_message: str) -> dict:
        """
        Execute the full agent loop for a single query
        Returns a dictionary with:
            response:       Final natural language answer to the user
            tools_called:   Name the tools that were called (if any)
            tool_result:    Raw tool output (JSON string)
            analytics:      Deterministic analytics results (if run)
            llm_call:       Number of llm calls made
        """
        state = {
            "messages": [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_message)
            ],
            "tools_called": None,
            "tool_result": None,
            "analytics": None,
            "llm_calls": 0
        }

        # ---------- STEP 1: Tool decision ----------
        ai_response = self.llm_tools.invoke(state["messages"])
        state["llm_calls"] += 1
        state["messages"].append(ai_response)

        tool_name = None
        tool_args = {}
        tool_output = None

        # ---------- STEP 2: Execute tool ----------
        if ai_response.tool_calls:

            tool_call = ai_response.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            # Fix LLM formatting mistakes (stringified lists)
            for k, v in tool_args.items():
                if isinstance(v, str) and v.startswith("[") and v.endswith("]"):
                    try:
                        tool_args[k] = json.loads(v)
                    except Exception:
                        try:
                            tool_args[k] = ast.literal_eval(v)
                        except Exception:
                            pass

            tool_fn = self.TOOL_MAP[tool_name]
            tool_output = tool_fn.invoke(tool_args)

            state["tools_called"] = tool_name
            state["tool_result"] = tool_output

            state["messages"].append(
                ToolMessage(content=tool_output, tool_call_id=tool_call_id)
            )

        # ---------- STEP 3: Deterministic analytics ----------
        if tool_name == "fetch_production_data":

            try:
                if len(tool_args.get("well_ids", [])) == 1:

                    analytics = compute_production_trends(
                        well_id=tool_args["well_ids"][0],
                        date_expression=tool_args.get(
                            "date_expression", "last_30_days"
                        )
                    )

                    state["analytics"] = analytics

                    state["messages"].append(
                        HumanMessage(
                            content=(
                                "Additional analytics computed by Python:\n"
                                + json.dumps(analytics, default=str)
                            )
                        )
                    )

            except Exception as e:

                state["messages"].append(
                    HumanMessage(
                        content=f"[Analytics unavailable: {e}]"
                    )
                )

        # ---------- STEP 4: Final explanation ----------
        explanation_messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_message)
        ]

        if state.get('tool_reslut'):
            explanation_messages.append(HumanMessage(
                content = (
                    f"[DATA retrieved from database]/n"
                    f"{state['tool_result']}/n/n"
                    f"Use the above data to answer the user's question. "
                    f"Do not call any tools. Write your answer in plain English."
                )
            ))

        if state.get('analytics'):
            explanation_messages.append(HumanMessage(
                content=(
                    f"[ANALYTICAL RESULT — computed by Python, guaranteed accurate]\n"
                    f"{json.dumps(state['analytics'], default=str)}\n\n"
                    f"Incorporate these computed statistics into your explanation."
                )               
            ))

        # ------------------------------
        final_response = self.llm_explain.invoke(state["messages"])
        state["llm_calls"] += 1
        state["response"] = final_response.content

        return state


    