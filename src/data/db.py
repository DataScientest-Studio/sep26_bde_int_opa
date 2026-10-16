"""Shared database connections. Both extract and transform scripts import from here."""
import os

import psycopg2
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()


def get_mongo_collection():
    """Return the bronze collection (raw_klines) in MongoDB."""
    uri = (
        f"mongodb://{os.getenv('MONGO_USER')}:{os.getenv('MONGO_PASSWORD')}"
        f"@{os.getenv('MONGO_HOST', 'localhost')}:{os.getenv('MONGO_PORT', '27017')}"
        "/?authSource=admin"
    )
    client = MongoClient(uri)
    db = client[os.getenv("MONGO_DB", "opa")]
    return db[os.getenv("MONGO_COLLECTION", "raw_klines")]


def get_pg_connection():
    """Return a connection to PostgreSQL (silver + gold)."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


if __name__ == "__main__":
    # Quick connection check: python src/data/db.py
    col = get_mongo_collection()
    print("Mongo OK, documents in raw_klines:", col.count_documents({}))
    with get_pg_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT symbol FROM dim_symbol ORDER BY symbol;")
        print("Postgres OK, symbols:", [r[0] for r in cur.fetchall()])
