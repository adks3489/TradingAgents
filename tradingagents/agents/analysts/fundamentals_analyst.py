from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
    calculate_intrinsic_value
)
from tradingagents.dataflows.config import get_config


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
            calculate_intrinsic_value,
        ]

        system_message = (
            "You are a senior fundamental analyst tasked with producing a comprehensive and quantitative fundamental analysis report for a company. "
            "Follow these steps in order:\n\n"
            "**Step 1 – Collect Raw Financial Data**: Use the available tools to retrieve the company's financials: "
            "`get_fundamentals` (overview), `get_income_statement`, `get_balance_sheet`, and `get_cashflow` (use 'annual' frequency for historical trends and 'quarterly' for recent data). "
            "Extract the following key figures for use in valuation: TTM Free Cash Flow (Operating Cash Flow − CapEx), "
            "shares outstanding, net debt (total debt − cash), current stock price, EPS (TTM), EBITDA (TTM), and total revenue (TTM).\n\n"
            "**Step 2 – Compute Intrinsic Value**: Call `calculate_intrinsic_value` providing the company's `ticker` and the `current_date`. "
            "The tool will automatically retrieve accurate financial figures from the system. "
            "You only need to provide `params_json` if you want to override specific values like growth assumptions or beta based on your analysis. "
            "Set growth assumptions based on the company's historical FCF growth CAGR (use a ±10% range around it for bull/bear). "
            "For multiples, use reasonable industry peer averages (e.g., Consumer Staples P/E ~18–22x, EV/EBITDA ~12–15x). "
            "Ensure you include `peer_pe`, `peer_ev_ebitda`, and `peer_ps` in `params_json` to get a full comparative analysis.\n\n"
            "**Step 3 – Write the Final Report**: Combine the qualitative fundamental analysis (financial history, balance sheet health, cash flow dynamics) "
            "with the quantitative valuation output from Step 2. The report MUST include: "
            "(a) a qualitative fundamental section, (b) the full quantitative valuation section as returned by the tool, "
            "(c) a strategic insights section (bull/bear case, catalysts, risks), and "
            "(d) a summary Markdown table of key metrics. "
            "Do not omit or paraphrase the valuation section — include it verbatim, as it contains deterministic calculations."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
