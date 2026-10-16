"""BRONZE: historical klines from Binance REST -> MongoDB (raw_klines).

Owner: Dennis

TODO:
- GET {BINANCE_REST_URL}/api/v3/klines?symbol=...&interval=...&startTime=...&limit=1000
- loop with startTime until now (max 1000 candles per request)
- for each kline, insert one document in the agreed format (see README)
"""
