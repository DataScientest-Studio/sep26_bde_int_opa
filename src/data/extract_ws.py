"""BRONZE: live klines from Binance WebSocket -> MongoDB (raw_klines).

Owner: Dennis

TODO:
- connect to {BINANCE_WS_URL}/stream?streams=btcusdt@kline_1m/solusdt@kline_1m
- the candle is in data["k"]; k["x"] == True means the candle is closed
- insert documents in the agreed format (see README), source = "websocket"
- reconnect if the connection drops
"""
