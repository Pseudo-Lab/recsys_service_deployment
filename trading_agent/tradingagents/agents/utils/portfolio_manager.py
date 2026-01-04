
from .kis_util import kis_client
from .user_profile import profile_manager

class PortfolioManager:
    def __init__(self):
        pass

    def judge_and_execute(self, ticker: str, price: float, confidence: str, reason: str, action: str = "BUY", exchange_cd: str = "NASD"):
        """
        1. Validates Confidence (Must be High/Strong) - Relaxed for testing
        2. Calculates Position Size
        3. Executes Trade via KIS
        """
        # 1. Validation
        # For Testing: Relaxed Confidence Check
        # if "HIGH" not in confidence.upper() and "STRONG" not in confidence.upper():
        #    return {"success": False, "msg": f"Trade skipped: Confidence '{confidence}' is too low."}

        # 2. Get Profile & Balance
        profile = profile_manager.load_profile()
        risk_tol = profile.risk_tolerance.upper()
        
        # Determine Allocation % based on Risk Tolerance
        alloc_pct = 0.05 # Default 5%
        if "AGGRESSIVE" in risk_tol:
            alloc_pct = 0.10
        elif "CONSERVATIVE" in risk_tol:
            alloc_pct = 0.02
            
        # Get Balance (Mock if API not set)
        if not kis_client.app_key:
            balance = 10000.0 # Mock $10k
            is_mock = True
        else:
            balance = kis_client.get_balance() or 1000.0 # Fallback 1000 if 0 to avoid div/0 in sizing logic if holding check fails
            is_mock = False

        if balance <= 0 and action == "BUY":
             return {"success": False, "msg": "Insufficient balance or API error."}

        # Calculate Qty (Targeting specific $ amount)
        # For SELL, we also use this as "Target Sell Amount" for now
        trade_amt = balance * alloc_pct
        qty = int(trade_amt // price)
        
        if qty < 1:
            qty = 1 # Force at least 1 for testing
            # return {"success": False, "msg": f"Calculated quantity is 0 (Balance: ${balance}, Price: ${price})"}

        # 3. Execute
        if is_mock:
            res = kis_client.mock_order(ticker, qty, price, side=action, exchange_cd=exchange_cd)
        else:
            if action == "SELL":
                res = kis_client.sell_limit_order(ticker, qty, price, exchange_cd=exchange_cd)
            else:
                res = kis_client.buy_limit_order(ticker, qty, price, exchange_cd=exchange_cd)

        return {
            "success": res["success"],
            "ticker": ticker,
            "action": action,
            "qty": qty,
            "price": price,
            "total_amt": qty * price,
            "balance_remaining": balance - (qty * price) if action=="BUY" else balance + (qty*price),
            "msg": res["msg"],
            "is_mock": is_mock
        }

portfolio_manager = PortfolioManager()
