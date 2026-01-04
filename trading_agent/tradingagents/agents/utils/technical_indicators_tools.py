from langchain_core.tools import tool
from typing import Annotated
from ...dataflows.interface import route_to_vendor

@tool
def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of. MUST be one of: close_50_sma, close_200_sma, close_10_ema, macd, macds, macdh, rsi, boll, boll_ub, boll_lb, atr, vwma, mfi. Do NOT use 'macd_signal' - use 'macds' instead."],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """
    Retrieve technical indicators for a given ticker symbol.
    Uses the configured technical_indicators vendor.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        indicator (str): Technical indicator to get. MUST be EXACTLY one of these valid values:
            - close_50_sma, close_200_sma, close_10_ema (Moving Averages)
            - macd, macds, macdh (MACD indicators - use 'macds' NOT 'macd_signal')
            - rsi (RSI)
            - boll, boll_ub, boll_lb, atr (Volatility)
            - vwma, mfi (Volume-based)
        curr_date (str): The current trading date you are trading on, YYYY-mm-dd
        look_back_days (int): How many days to look back, default is 30
    Returns:
        str: A formatted dataframe containing the technical indicators for the specified ticker symbol and indicator.
    """
    return route_to_vendor("get_indicators", symbol, indicator, curr_date, look_back_days)