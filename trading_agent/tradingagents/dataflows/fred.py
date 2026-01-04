from pandas_datareader import data as web  
from typing import Annotated  
from datetime import datetime, timedelta  
  
def get_net_liquidity(  
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],  
    lookback_days: Annotated[int, "Number of days to look back"] = 90  
) -> str:  
    """  
    Calculate Net Liquidity = Fed Balance Sheet - (TGA + RRP)  
    """  
    end = datetime.strptime(curr_date, "%Y-%m-%d")  
    start = end - timedelta(days=lookback_days)  
      
    # WALCL: 연준 총자산, WTREGEN: 재무부 계정, RRPONTSYD: 역레포  
    df = web.DataReader(['WALCL', 'WTREGEN', 'RRPONTSYD'], 'fred', start, end)  
    df = df.fillna(method='ffill')  
      
    df['Net_Liquidity'] = df['WALCL'] - (df['WTREGEN'] + df['RRPONTSYD'])  
      
    current_liquidity = df['Net_Liquidity'].iloc[-1]  
    ma_20 = df['Net_Liquidity'].rolling(window=20).mean().iloc[-1]  
    trend = "INCREASING" if current_liquidity > ma_20 else "DECREASING"  
      
    return f"Net Liquidity: {current_liquidity:.2f}B, Trend: {trend}, 20-day MA: {ma_20:.2f}B\n{df.tail(10).to_string()}"  
  
def get_macro_indicators(  
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"]  
) -> str:  
    """  
    Get CPI, PCE, Unemployment Rate, and Yield Curve (10Y-2Y)  
    """  
    end = datetime.strptime(curr_date, "%Y-%m-%d")  
    start = end - timedelta(days=365)  
      
    # CPIAUCSL: CPI, PCE: PCE, UNRATE: 실업률, DGS10: 10년물, DGS2: 2년물  
    df = web.DataReader(['CPIAUCSL', 'PCE', 'UNRATE', 'DGS10', 'DGS2'], 'fred', start, end)  
    df = df.fillna(method='ffill')  
      
    df['Yield_Curve'] = df['DGS10'] - df['DGS2']  
      
    return f"Latest Macro Indicators:\n{df.tail(5).to_string()}"
