-- Trading pairs known to the warehouse.
SELECT symbol, base_asset, quote_asset FROM dim_symbol ORDER BY symbol;
