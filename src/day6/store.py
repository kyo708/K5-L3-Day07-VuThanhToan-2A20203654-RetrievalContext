"""Store and search chunk vectors in PostgreSQL + pgvector."""
import hashlib
from typing import Any

import psycopg

from .models import Chunk

# Metadata columns a caller may filter on; values are always bound as parameters.
FILTERABLE = {"chapter", "page", "document_id"}


def content_hash(chunk: Chunk) -> str:
    """Hash of everything that is embedded or stored, to detect changed chunks."""
    payload = f"{chunk.text}\x1f{chunk.page}\x1f{chunk.metadata.get('chapter')}\x1f{chunk.metadata.get('article')}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in vector) + "]"


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError(f"invalid embedding dimensions: {dimensions}")
        self.database_url = database_url
        self.dimensions = dimensions
        self.table = f"chunks_{dimensions}"  # a new model size never mixes with old vectors

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url)

    def create_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    chunk_id     TEXT PRIMARY KEY,
                    strategy     TEXT NOT NULL,
                    document_id  TEXT NOT NULL,
                    source       TEXT NOT NULL,
                    page         INTEGER NOT NULL,
                    chapter      INTEGER NOT NULL,
                    article      INTEGER,
                    text         TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    embedding    vector({self.dimensions}) NOT NULL
                )""")
            conn.execute(f"CREATE INDEX IF NOT EXISTS {self.table}_strategy_chapter "
                         f"ON {self.table} (strategy, chapter)")

    def existing_hashes(self, strategy: str, document_id: str) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute(f"SELECT chunk_id, content_hash FROM {self.table} "
                                "WHERE strategy = %s AND document_id = %s", (strategy, document_id)).fetchall()
        return dict(rows)

    def count(self, strategy: str) -> int:
        with self._connect() as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {self.table} WHERE strategy = %s", (strategy,)).fetchone()[0]

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert or update `chunks` (with matching `vectors`) by stable chunk_id."""
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise ValueError(f"vector has {len(vector)} dims, store expects {self.dimensions}")
        rows = [
            (c.chunk_id, strategy, c.document_id, c.source, c.page, c.metadata["chapter"],
             c.metadata.get("article"), c.text, content_hash(c), _vector_literal(v))
            for c, v in zip(chunks, vectors)
        ]
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(f"""
                INSERT INTO {self.table}
                    (chunk_id, strategy, document_id, source, page, chapter, article, text, content_hash, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    strategy = EXCLUDED.strategy, document_id = EXCLUDED.document_id,
                    source = EXCLUDED.source, page = EXCLUDED.page, chapter = EXCLUDED.chapter,
                    article = EXCLUDED.article, text = EXCLUDED.text,
                    content_hash = EXCLUDED.content_hash, embedding = EXCLUDED.embedding""", rows)

    def delete_obsolete(self, strategy: str, document_id: str, keep_ids: set[str]) -> int:
        """Remove chunks of this document/strategy that the latest chunking no longer produces."""
        stale = [cid for cid in self.existing_hashes(strategy, document_id) if cid not in keep_ids]
        if stale:
            with self._connect() as conn:
                conn.execute(f"DELETE FROM {self.table} WHERE chunk_id = ANY(%s)", (stale,))
        return len(stale)

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Exact cosine search (no ANN index), optionally restricted by metadata in SQL."""
        if len(query_vector) != self.dimensions:
            raise ValueError(f"query has {len(query_vector)} dims, store expects {self.dimensions}")
        where, params = ["strategy = %s"], [strategy]
        for key, value in (filters or {}).items():
            if key not in FILTERABLE:
                raise ValueError(f"filter on {key!r} is not allowed")
            where.append(f"{key} = %s")
            params.append(value)
        literal = _vector_literal(query_vector)
        sql = (f"SELECT chunk_id, text, 1 - (embedding <=> %s::vector) AS score, document_id, source, page, "
               f"chapter, article FROM {self.table} WHERE {' AND '.join(where)} "
               f"ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s")
        with self._connect() as conn:
            rows = conn.execute(sql, [literal, *params, literal, top_k]).fetchall()
        return [
            {"rank": rank, "chunk_id": r[0], "text": r[1], "score": round(float(r[2]), 6),
             "metadata": {"document_id": r[3], "source": r[4], "page": r[5], "chapter": r[6], "article": r[7]}}
            for rank, r in enumerate(rows, start=1)
        ]
