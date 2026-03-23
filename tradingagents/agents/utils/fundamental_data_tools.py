from langchain_core.tools import tool
from typing import Annotated, Optional
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.agents.utils.valuation_logic import DCFInputs, run_valuation
import json


@tool
def get_fundamentals(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Retrieve comprehensive fundamental data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing comprehensive fundamental data
    """
    return route_to_vendor("get_fundamentals", ticker, curr_date)


@tool
def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve balance sheet data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing balance sheet data
    """
    return route_to_vendor("get_balance_sheet", ticker, freq, curr_date)


@tool
def get_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve cash flow statement data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing cash flow statement data
    """
    return route_to_vendor("get_cashflow", ticker, freq, curr_date)


@tool
def get_income_statement(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve income statement data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing income statement data
    """
    return route_to_vendor("get_income_statement", ticker, freq, curr_date)


@tool
def calculate_intrinsic_value(
    ticker: Annotated[Optional[str], "Ticker symbol to auto-fetch financial data."] = None,
    curr_date: Annotated[Optional[str], "Current trading date (yyyy-mm-dd) for auto-fetching data."] = None,
    params_json: Annotated[
        Optional[str],
        """JSON string with financial parameters for valuation. Required fields if ticker/curr_date not provided:
        - free_cash_flow_ttm (float): TTM Free Cash Flow in millions of dollars.
        - shares_outstanding (float): Total shares outstanding in millions.
        - net_debt (float): Total debt minus total cash and equivalents, in millions.
        - current_price (float): Current stock market price per share.
        Optional fields: growth_rate_base, growth_rate_bull, growth_rate_bear, beta, etc.
        Example: {"free_cash_flow_ttm": 64.8, "shares_outstanding": 46.1, "net_debt": -150.6, "current_price": 14.51}""",
    ] = None,
) -> str:
    """
    Calculate the intrinsic value of a stock using DCF and comparable multiples.
    Uses a deterministic Python engine to ensure accuracy.
    Can auto-fetch data if 'ticker' and 'curr_date' are provided, or use 'params_json'.
    Returns a formatted Markdown valuation report.
    """
    params = {}
    if params_json:
        try:
            params = json.loads(params_json)
        except json.JSONDecodeError as e:
            return f"ERROR: Invalid JSON input. Detail: {e}"

    # Auto-fetch logic if ticker is provided
    if ticker:
        import yfinance as yf
        try:
            t = yf.Ticker(ticker.upper())
            info = t.info
            
            def to_m(val):
                return round(float(val) / 1_000_000, 2) if val is not None else None

            # Fetch accurate data from system
            auto_params = {
                "free_cash_flow_ttm": to_m(info.get("freeCashflow")),
                "shares_outstanding": round(float(info.get("sharesOutstanding")) / 1_000_000, 2) if info.get("sharesOutstanding") else None,
                "net_debt": to_m((info.get("totalDebt") or 0) - (info.get("totalCash") or 0)),
                "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "beta": info.get("beta"),
                "eps_ttm": info.get("trailingEps"),
                "ebitda_ttm": to_m(info.get("ebitda")),
                "revenue_ttm": to_m(info.get("totalRevenue")),
            }
            
            # Use auto-fetched data for any field NOT provided in params_json
            for k, v in auto_params.items():
                if v is not None and k not in params:
                    params[k] = v
                    
            # If curr_date is provided and we're doing a historical price check (simplified for now)
            if curr_date and "current_price" not in params:
                hist = t.history(start=curr_date, end=curr_date)
                if not hist.empty:
                    params["current_price"] = float(hist["Close"].iloc[0])

        except Exception as e:
            # Fallback to manual if auto-fetch fails, but log it
            print(f"Warning: Auto-fetch failed for {ticker}: {e}")

    required = ["free_cash_flow_ttm", "shares_outstanding", "net_debt", "current_price"]
    missing = [k for k in required if k not in params]
    if missing:
        return f"ERROR: Missing required fields: {missing}. Provide more data or ensure ticker/curr_date are valid."

    try:
        inputs = DCFInputs(
            free_cash_flow_ttm=float(params["free_cash_flow_ttm"]),
            shares_outstanding=float(params["shares_outstanding"]),
            net_debt=float(params["net_debt"]),
            current_price=float(params["current_price"]),
            growth_rate_base=float(params.get("growth_rate_base", 0.15)),
            growth_rate_bull=float(params.get("growth_rate_bull", 0.25)),
            growth_rate_bear=float(params.get("growth_rate_bear", 0.05)),
            beta=float(params.get("beta", 1.0)),
            risk_free_rate=float(params.get("risk_free_rate", 0.043)),
            equity_risk_premium=float(params.get("equity_risk_premium", 0.055)),
            terminal_growth_rate=float(params.get("terminal_growth_rate", 0.03)),
            projection_years=int(params.get("projection_years", 5)),
            eps_ttm=params.get("eps_ttm"),
            ebitda_ttm=params.get("ebitda_ttm"),
            revenue_ttm=params.get("revenue_ttm"),
            peer_pe=params.get("peer_pe"),
            peer_ev_ebitda=params.get("peer_ev_ebitda"),
            peer_ps=params.get("peer_ps"),
        )
    except (ValueError, TypeError) as e:
        return f"ERROR: Invalid parameter type. Detail: {e}"

    try:
        result = run_valuation(inputs)
    except Exception as e:
        return f"ERROR: Valuation computation failed. Detail: {e}"

    return result.formatted_report
