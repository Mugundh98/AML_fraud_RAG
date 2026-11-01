from app.rag.citations import Source, cited_sources, normalize_citations


def test_wide_citations_become_square_brackets():
    text = "High priority【1】. EDD applies【3†source】, see also [2]."
    assert normalize_citations(text) == "High priority[1]. EDD applies[3], see also [2]."


def test_only_cited_sources_are_kept():
    sources = [Source(n, "t", f"s{n}", "", None, 0.0, "x") for n in range(1, 6)]
    assert [s.number for s in cited_sources("A [2][4], B [1, 5].", sources)] == [1, 2, 4, 5]
    assert len(cited_sources("No citations here.", sources)) == 5
