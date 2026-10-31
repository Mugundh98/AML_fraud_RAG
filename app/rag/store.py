import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from app.config import Settings
from app.rag.chunking import Chunk

DENSE = "dense"
SPARSE = "sparse"


@dataclass
class SearchHit:
    text: str
    source: str
    title: str
    section: str
    page: int | None
    score: float


class DocStore:
    """Qdrant collection with hybrid search: dense embeddings + BM25 sparse vectors, fused with RRF."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.collection = settings.qdrant_collection
        if settings.qdrant_url:
            self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
            self.is_local = False
        else:
            # Embedded mode: only one process can open this folder at a time.
            settings.qdrant_local_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(settings.qdrant_local_path))
            self.is_local = True
        self._dense = None
        self._sparse = None

    @property
    def dense(self):
        if self._dense is None:
            from fastembed import TextEmbedding

            self._dense = TextEmbedding(self.settings.dense_model)
        return self._dense

    @property
    def sparse(self):
        if self._sparse is None:
            from fastembed import SparseTextEmbedding

            self._sparse = SparseTextEmbedding(self.settings.sparse_model)
        return self._sparse

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        dim = len(next(iter(self.dense.embed(["dimension probe"]))))
        self.client.create_collection(
            self.collection,
            vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        if not self.is_local:
            self.client.create_payload_index(self.collection, "source", models.PayloadSchemaType.KEYWORD)

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection, exact=True).count

    def delete_sources(self, sources: set[str]) -> None:
        if not sources or not self.client.collection_exists(self.collection):
            return
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="source", match=models.MatchAny(any=sorted(sources)))]
                )
            ),
        )

    def upsert(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        self.ensure_collection()
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [c.embedding_text for c in batch]
            dense_vectors = list(self.dense.embed(texts))
            sparse_vectors = list(self.sparse.embed(texts))
            points = [
                models.PointStruct(
                    # Deterministic IDs make re-ingesting the same file an overwrite, not a duplicate.
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{c.source}#{c.chunk_index}")),
                    vector={
                        DENSE: d.tolist(),
                        SPARSE: models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist()),
                    },
                    payload={
                        "text": c.text,
                        "source": c.source,
                        "title": c.title,
                        "section": c.section,
                        "page": c.page,
                        "chunk_index": c.chunk_index,
                    },
                )
                for c, d, s in zip(batch, dense_vectors, sparse_vectors)
            ]
            self.client.upsert(self.collection, points=points)

    def search(self, query: str, limit: int) -> list[SearchHit]:
        if not self.client.collection_exists(self.collection):
            return []
        dense_query = next(iter(self.dense.query_embed(query))).tolist()
        sparse_query = next(iter(self.sparse.query_embed(query)))
        result = self.client.query_points(
            self.collection,
            prefetch=[
                models.Prefetch(query=dense_query, using=DENSE, limit=limit * 4),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_query.indices.tolist(), values=sparse_query.values.tolist()
                    ),
                    using=SPARSE,
                    limit=limit * 4,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [
            SearchHit(
                text=p.payload["text"],
                source=p.payload["source"],
                title=p.payload["title"],
                section=p.payload.get("section") or "",
                page=p.payload.get("page"),
                score=p.score,
            )
            for p in result.points
        ]
