"""Load data/synthetic/*.csv into Neo4j, replacing whatever is there.

Run from the project root:  python -m scripts.load_graph
"""

import time

from app.config import get_settings
from app.graph.db import connect
from app.graph.loader import load_dataset


def main() -> None:
    settings = get_settings()
    if not (settings.synthetic_dir / "transfers.csv").exists():
        raise SystemExit("No data found. Run first: python -m scripts.generate_data")

    driver = connect(settings)
    started = time.time()
    try:
        counts = load_dataset(driver, settings.synthetic_dir, settings.neo4j_database)
    finally:
        driver.close()
    print(f"Done in {time.time() - started:.0f}s: {counts}")


if __name__ == "__main__":
    main()
