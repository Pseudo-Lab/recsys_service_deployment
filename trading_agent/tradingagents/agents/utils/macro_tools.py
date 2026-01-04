from langchain_core.tools import tool  
from typing import Annotated  
from ...dataflows.fred import get_net_liquidity, get_macro_indicators  
  
@tool  
def get_net_liquidity_tool(  
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],  
    lookback_days: Annotated[int, "Number of days to look back"] = 90  
) -> str:  
    """순유동성 지표 조회: Fed Balance Sheet - (TGA + RRP)"""  
    return get_net_liquidity(curr_date, lookback_days)  
  
@tool  
def get_macro_indicators_tool(  
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"]  
) -> str:  
    """CPI, PCE, 실업률, 장단기 금리차 조회"""  
    return get_macro_indicators(curr_date)