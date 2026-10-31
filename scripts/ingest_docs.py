"""Index everything in data/docs into Qdrant.

Run from the project root:  python -m scripts.ingest_docs

With the local (embedded) Qdrant store, stop the API first: only one process can
open the store at a time. Or use the "Re-index documents" button in the UI instead.
"""

from app.config import get_settings
from app.rag.ingest import ingest_directory
from app.rag.store import DocStore


def main() -> None:
    settings = get_settings()
    store = DocStore(settings)
    print(f"Indexing {settings.docs_dir} into collection '{settings.qdrant_collection}'…")
    stats = ingest_directory(store, settings)
    print(f"Done: {stats.files} files, {stats.chunks} chunks, {stats.total_points} points in collection.")
    store.client.close()


if __name__ == "__main__":
    main()
