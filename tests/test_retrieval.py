import numpy as np

from recruiter import retrieval
from recruiter.schemas import JobSpec


def test_tokenize_keeps_tech_names():
    assert retrieval.tokenize("C++, C# and Node.js at Company Name.") == ["c++", "c#", "node.js"]


def test_chunk_words_covers_text():
    text = " ".join(str(i) for i in range(500))
    chunks = retrieval.chunk_words(text, size=200, overlap=40)
    assert chunks[0].split()[0] == "0" and chunks[-1].split()[-1] == "499"


def test_hybrid_search_with_fake_encoder(monkeypatch, tmp_path):
    vocab = ["python", "spark", "nurse", "patient"]

    def fake_embed(texts, is_query=False):
        v = np.array([[t.lower().count(w) for w in vocab] for t in texts], dtype=float) + 1e-6
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    monkeypatch.setattr(retrieval, "embed", fake_embed)
    monkeypatch.setattr(retrieval.config, "DATA_PROCESSED", tmp_path)
    ids = ["eng", "nurse", "other"]
    texts = ["data engineer python spark pipelines", "registered nurse patient care", "chef menu"]
    index = retrieval.build_index(ids, texts)
    hits = retrieval.search(index, "python spark engineer", k=2)
    assert hits[0].id == "eng"
    assert [h.id for h in retrieval.search(index, "python", restrict=["nurse", "other"])] != ["eng"]


def test_query_from_spec_boosts_must_haves():
    q = retrieval.query_from_spec(JobSpec(title="Data Engineer", must_have_skills=["spark"]))
    assert q.count("spark") == 2
