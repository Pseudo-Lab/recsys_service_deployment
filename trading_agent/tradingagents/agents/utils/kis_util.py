
import os
import requests
import json
import time

class KisUSClient:
    def __init__(self):
        self.app_key = os.getenv("KIS_APP_KEY")
        self.app_secret = os.getenv("KIS_APP_SECRET")
        self.ano = os.getenv("KIS_CANO") # Account No (Front 8)
        self.ano_prdt = os.getenv("KIS_ACNT_PRDT_CD") # Account Product Code (Back 2)
        self.mode = os.getenv("KIS_MODE", "VIRTUAL") # REAL or VIRTUAL

        if self.mode == "REAL":
            self.base_url = "https://openapi.koreainvestment.com:9443"
        else:
            self.base_url = "https://openapivts.koreainvestment.com:29443"

        self.access_token = None
    
    def _headers(self, tr_id=None):
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.get_access_token()}",
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }
        if tr_id:
            headers["tr_id"] = tr_id
        return headers

    def get_access_token(self):
        # MVP: Simple token fetch (In prod, should cache/refresh)
        if self.access_token:
            return self.access_token
        
        url = f"{self.base_url}/oauth2/tokenP"
        headers = {"content-type": "application/json"}
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }
        
        try:
            res = requests.post(url, headers=headers, data=json.dumps(body))
            if res.status_code == 200:
                self.access_token = res.json()["access_token"]
                return self.access_token
            else:
                print(f"KIS Token Error: {res.text}")
                return None
        except Exception as e:
            print(f"KIS Token Exception: {e}")
            return None

    def get_balance(self):
        """
        Returns (Deposit Balance USD, Exchange Rate)
        TR_ID: VTTT8804U (Virtual), TTTS3012R (Real - Check docs, usually different)
        For MVP, assuming Virtual US Stock Balance TR
        """
        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-present-balance"
        tr_id = "VTTT8804U" if self.mode == "VIRTUAL" else "TTTS3012R" # Check real TR ID

        headers = self._headers(tr_id=tr_id)
        params = {
            "CANO": self.ano,
            "ACNT_PRDT_CD": self.ano_prdt,
            "WCRC_FRCR_DVS_CD": "02", # USD
            "NATN_CD": "840", # US
            "TR_MK": "01",
            "INQR_DVS_CD": "00"
        }
        
        try:
            res = requests.get(url, headers=headers, params=params)
            data = res.json()
            if res.status_code == 200 and data['rt_cd'] == '0':
                # Parse output2 for deposit
                # Note: KIS API response structure varies. Assume 'output2' has 'ovrs_ord_psbl_amt' (Orderable Amount)
                # This is a simplification.
                deposit = float(data.get('output2', {}).get('ovrs_ord_psbl_amt', 0))
                return deposit
            else:
                print(f"KIS Balance Error: {data.get('msg1')}")
                return 0.0
        except Exception as e:
            print(f"KIS Balance Exception: {e}")
            return 0.0

    def buy_limit_order(self, ticker: str, qty: int, price: float, exchange_cd: str = "NASD"):
        """
        Buy US Stock (Limit Order)
        TR_ID: VTTT1002U (Virtual Buy), TTTS1002U (Real Buy)
        """
        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/order"
        tr_id = "VTTT1002U" if self.mode == "VIRTUAL" else "TTTS1002U" 

        headers = self._headers(tr_id=tr_id)
        
        # KIS requires strict formatting
        # KIS requires strict formatting
        # ORD_SVR_DVS_CD: Try empty string for overseas
        body = {
            "CANO": self.ano,
            "ACNT_PRDT_CD": self.ano_prdt,
            "OVRS_EXCG_CD": exchange_cd, 
            "PDNO": ticker.upper(),
            "ORD_QTY": str(qty),
            "OVRS_ORD_UNPR": str(price),
            "ORD_SVR_DVS_CD": "0",
            "ORD_DVS": "00" # Limit
        }
        
        try:
            res = requests.post(url, headers=headers, data=json.dumps(body))
            data = res.json()
            if res.status_code == 200 and data['rt_cd'] == '0':
                return {"success": True, "msg": data['msg1'], "order_no": data['output']['ODNO']}
            else:
                 return {"success": False, "msg": data.get('msg1', 'Unknown Error')}
        except Exception as e:
            return {"success": False, "msg": str(e)}

    def sell_limit_order(self, ticker: str, qty: int, price: float, exchange_cd: str = "NASD"):
        """
        Sell US Stock (Limit Order)
        TR_ID: VTTT1001U (Virtual Sell), TTTS1001U (Real Sell)
        """
        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/order"
        tr_id = "VTTT1001U" if self.mode == "VIRTUAL" else "TTTS1001U"

        headers = self._headers(tr_id=tr_id)
        
        body = {
            "CANO": self.ano,
            "ACNT_PRDT_CD": self.ano_prdt,
            "OVRS_EXCG_CD": exchange_cd, 
            "PDNO": ticker.upper(),
            "ORD_QTY": str(qty),
            "OVRS_ORD_UNPR": str(price),
            "ORD_SVR_DVS_CD": "0", 
            "ORD_DVS": "00" 
        }
        
        try:
            res = requests.post(url, headers=headers, data=json.dumps(body))
            data = res.json()
            if res.status_code == 200 and data['rt_cd'] == '0':
                return {"success": True, "msg": data['msg1'], "order_no": data['output']['ODNO']}
            else:
                 return {"success": False, "msg": data.get('msg1', 'Unknown Error')}
        except Exception as e:
            return {"success": False, "msg": str(e)}

    # Mock Mode for testing without API Keys
    def mock_order(self, ticker, qty, price, side="BUY", exchange_cd="NASD"):
        time.sleep(1)
        return {"success": True, "msg": f"MOCK {side} ORDER SUCCESS ({exchange_cd})", "order_no": "12345678"}

# Global Instance
kis_client = KisUSClient()
