"""Send one small read request to Qdrant Cloud and Neo4j AuraDB so their free tiers don't pause them.

Both suspend a free database after a few days without requests (Qdrant warns by email first). Any read
resets the timer, so this reads one document passage and counts customers. Nothing is changed.

Run from the project root:  python -m scripts.keep_alive
Windows Task Scheduler runs it every 2 days (task "AML assistant keep-alive"; see the README). Each run
appends a line to storage/keep_alive.log, and the exit code is 1 if either database didn't answer.
"""

import sys
from datetime import datetime

from app.config import PROJECT_ROOT, get_settings
from app.graph.db import connect

LOG = PROJECT_ROOT / "storage" / "keep_alive.log"


def ping_qdrant(settings) -> str:
    from qdrant_client import QdrantClient

    if not settings.qdrant_url:
        return "skipped (no QDRANT_URL)"
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=30)
    client.scroll(settings.qdrant_collection, limit=1, with_payload=False)
    return f"ok ({client.get_collection(settings.qdrant_collection).points_count} passages)"


def ping_neo4j(settings) -> str:
    with connect(settings) as driver:
        rows, _, _ = driver.execute_query("MATCH (c:Customer) RETURN count(c) AS n", database_=settings.neo4j_database)
    return f"ok ({rows[0]['n']} customers)"


def main() -> None:
    settings = get_settings()
    results, failed = [], False
    for name, ping in [("Qdrant", ping_qdrant), ("Neo4j", ping_neo4j)]:
        try:
            results.append(f"{name} {ping(settings)}")
        except Exception as exc:  # report every database, even if one is down
            failed = True
            results.append(f"{name} FAILED ({type(exc).__name__}: {str(exc)[:120]})")
    line = f"{datetime.now():%Y-%m-%d %H:%M}  " + " | ".join(results)
    print(line)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
