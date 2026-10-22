import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# ==============================================================================
# SUPABASE POSTGRESQL CONFIGURATION
# ==============================================================================
SUPABASE_DB_URL = os.getenv(
    "SUPABASE_DB_URL",
    "postgresql://postgres.snaylmuzcijunycsroto:DeePa%40015SUBASRI@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres?sslmode=require"
)

SUPABASE_HOST = os.getenv("SUPABASE_HOST", "aws-0-ap-southeast-1.pooler.supabase.com")
SUPABASE_PORT = int(os.getenv("SUPABASE_PORT", "6543"))
SUPABASE_USER = os.getenv("SUPABASE_USER", "postgres.snaylmuzcijunycsroto")
SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD", "DeePa@015SUBASRI")
SUPABASE_DB = os.getenv("SUPABASE_DB", "postgres")


class SupabaseConnectionWrapper:
    """
    Wrapper around psycopg2 connection to ensure 100% compatibility
    with MySQL Connector cursor syntax used in app.py (e.g., dictionary=True).
    """
    def __init__(self, raw_conn):
        self._conn = raw_conn

    def cursor(self, dictionary=False, **kwargs):
        if dictionary:
            return self._conn.cursor(cursor_factory=RealDictCursor, **kwargs)
        return self._conn.cursor(**kwargs)

    def commit(self):
        return self._conn.commit()

    def rollback(self):
        return self._conn.rollback()

    def close(self):
        return self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.rollback()
        else:
            self.commit()
        self.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)


def get_connection():
    """
    Establishes and returns a connection to the Supabase database.
    """
    if SUPABASE_DB_URL and SUPABASE_DB_URL.strip():
        db_url = SUPABASE_DB_URL.strip()
        if db_url.startswith("postgres://"):
            db_url = "postgresql://" + db_url[len("postgres://"):]
        
        if "sslmode=" not in db_url:
            separator = "&" if "?" in db_url else "?"
            db_url = f"{db_url}{separator}sslmode=require"
            
        raw_conn = psycopg2.connect(db_url)
    elif SUPABASE_HOST and SUPABASE_HOST.strip():
        raw_conn = psycopg2.connect(
            host=SUPABASE_HOST.strip(),
            port=SUPABASE_PORT,
            user=SUPABASE_USER.strip(),
            password=SUPABASE_PASSWORD,
            dbname=SUPABASE_DB.strip(),
            sslmode="require"
        )
    else:
        raise ValueError(
            "Supabase database connection details are not configured! "
            "Please check SUPABASE_DB_URL in .env or database.py."
        )

    return SupabaseConnectionWrapper(raw_conn)


if __name__ == "__main__":
    try:
        con = get_connection()
        print("[+] Supabase PostgreSQL Connection Successful!")
        cur = con.cursor(dictionary=True)
        cur.execute("SELECT * FROM users")
        rows = cur.fetchall()
        print(f"[+] Total Users in Supabase: {len(rows)}")
        for u in rows:
            print(f"    - User: {u['userid']} ({u['fullname']}) | Email: {u['email']}")
        cur.close()
        con.close()
    except Exception as e:
        print("[-] Connection Failed:", e)
