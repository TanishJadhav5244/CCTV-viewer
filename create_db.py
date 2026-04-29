"""Create the cctv_analytics database if it doesn't exist."""
import psycopg2

conn = psycopg2.connect(dbname='postgres', user='postgres', password='postgres', host='localhost', port=5432)
conn.autocommit = True
cur = conn.cursor()

cur.execute("SELECT 1 FROM pg_database WHERE datname='cctv_analytics'")
exists = cur.fetchone()
if not exists:
    cur.execute("CREATE DATABASE cctv_analytics")
    print("[OK] Created database 'cctv_analytics'")
else:
    print("[OK] Database 'cctv_analytics' already exists")

conn.close()
print("Done.")
