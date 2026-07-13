"""
富邦模組共用設定：subscribe_list.py 和 marketdata_ws.py 都需要的常數放這裡，
避免兩邊互相 import 造成循環匯入。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env", override=True)

# 富邦行情 WebSocket rate limit：200 檔訂閱/連線、最多 5 條連線。
MAX_PER_CONNECTION = int(os.environ.get("FUBON_WS_SYMBOLS_PER_CONN", "200"))
MAX_CONNECTIONS = int(os.environ.get("FUBON_WS_CONNECTIONS", "5"))
