from dataclasses import dataclass

from app.config import Settings
from app.rag.chunking import chunk_documents
from app.rag.loaders import load_directory
from app.rag.store import DocStore


@dataclass
class IngestStats:
    files: int
    chunks: int
    total_points: int


def ingest_directory(store: DocStore, settings: Settings) -> IngestStats:
    docs = load_directory(settings.docs_dir)
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    sources = {d.source for d in docs}
    # Drop old chunks first so a file that got shorter doesn't leave stale tail chunks behind.
    store.delete_sources(sources)
    store.upsert(chunks)
    return IngestStats(files=len(sources), chunks=len(chunks), total_points=store.count())
