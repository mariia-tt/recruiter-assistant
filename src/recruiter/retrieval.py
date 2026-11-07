"""C3 Hybrid retriever: BM25 (exact terms) + dense embeddings (paraphrases), fused with RRF."""

import hashlib
import logging
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from recruiter import config
from recruiter.schemas import JobSpec, SearchHit

log = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.]*[a-z0-9+#]|[a-z0-9]")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is",
    "it", "of", "on", "or", "our", "that", "the", "this", "to", "was", "were", "will", "with",
    "you", "your", "we", "they", "their", "i", "my", "me", "city", "state", "company", "name",
}


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens that keep names like c++, c#, node.js intact."""
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    """Overlapping word windows so long CVs are fully covered by the encoder."""
    words = text.split()
    step = size - overlap
    return [" ".join(words[i : i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


@lru_cache(maxsize=1)
def _encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL)


def embed(texts: list[str], is_query: bool = False) -> np.ndarray:
    """L2-normalised embeddings; BGE models expect an instruction prefix on queries."""
    if is_query:
        texts = [config.EMBED_QUERY_PREFIX + t for t in texts]
    return _encoder().encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
    )


@dataclass
class HybridIndex:
    ids: list[str]
    bm25: BM25Okapi
    chunk_emb: np.ndarray  # (n_chunks, dim)
    chunk_owner: np.ndarray  # (n_chunks,) index into ids


def build_index(ids: list[str], texts: list[str]) -> HybridIndex:
    """Build BM25 over full texts and a chunk-level embedding matrix (cached on disk)."""
    chunks, owner = [], []
    for i, t in enumerate(texts):
        for c in chunk_words(t, config.CHUNK_WORDS, config.CHUNK_OVERLAP):
            chunks.append(c)
            owner.append(i)
    digest = hashlib.md5(("|".join(chunks) + config.EMBED_MODEL).encode()).hexdigest()[:12]
    cache = config.DATA_PROCESSED / f"emb_{digest}.npy"
    if cache.exists():
        emb = np.load(cache)
    else:
        log.info("embedding %d chunks with %s", len(chunks), config.EMBED_MODEL)
        emb = embed(chunks)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb)
    return HybridIndex(list(ids), BM25Okapi([tokenize(t) for t in texts]), emb, np.array(owner))


def bm25_scores(index: HybridIndex, query: str) -> np.ndarray:
    return index.bm25.get_scores(tokenize(query))


def dense_scores(index: HybridIndex, query: str) -> np.ndarray:
    """Per-document score = best-matching chunk (max pooling)."""
    sims = index.chunk_emb @ embed([query], is_query=True)[0]
    scores = np.full(len(index.ids), -1.0)
    np.maximum.at(scores, index.chunk_owner, sims)
    return scores


def _ranks(scores: np.ndarray) -> np.ndarray:
    """1-based rank of every document (1 = best)."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def search(
    index: HybridIndex,
    query: str,
    k: int = 50,
    mode: str = "hybrid",
    restrict: list[str] | None = None,
) -> list[SearchHit]:
    """Top-k documents for a query. mode: 'bm25' | 'dense' | 'hybrid' (RRF fusion).

    `restrict` limits ranking to a subset of ids (used to rerank a labelled pool in evals).
    """
    bm25_r = _ranks(bm25_scores(index, query)) if mode in ("bm25", "hybrid") else None
    dense_r = _ranks(dense_scores(index, query)) if mode in ("dense", "hybrid") else None
    if mode == "bm25":
        score = 1.0 / bm25_r
    elif mode == "dense":
        score = 1.0 / dense_r
    else:
        rrf = config.RRF_K
        score = 1.0 / (rrf + bm25_r) + 1.0 / (rrf + dense_r)

    allowed = None if restrict is None else {index.ids.index(r) for r in restrict}
    order = [i for i in np.argsort(-score, kind="stable") if allowed is None or i in allowed][:k]
    return [
        SearchHit(
            id=index.ids[i],
            score=float(score[i]),
            bm25_rank=int(bm25_r[i]) if bm25_r is not None else None,
            dense_rank=int(dense_r[i]) if dense_r is not None else None,
        )
        for i in order
    ]


def query_from_spec(spec: JobSpec) -> str:
    """Search query text built from the structured vacancy (must-haves repeated = boosted)."""
    parts = [spec.title or "", spec.seniority or ""]
    parts += spec.must_have_skills * 2 + spec.nice_to_have_skills
    if spec.domain:
        parts.append(spec.domain)
    return " ".join(p for p in parts if p)
