from app.rag.pipeline import normalize_citations


def test_wide_citations_become_square_brackets():
    text = "High priority【1】. EDD applies【3†source】, see also [2]."
    assert normalize_citations(text) == "High priority[1]. EDD applies[3], see also [2]."
