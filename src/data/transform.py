"""SILVER: MongoDB raw_klines -> PostgreSQL fact_kline.

Owner: Talgat

TODO:
- read documents from raw_klines
- parse payload (array for source="rest", object with "k" for source="websocket")
- convert types (strings -> numbers, ms -> timestamps)
- upsert into fact_kline (ON CONFLICT DO NOTHING)
"""
