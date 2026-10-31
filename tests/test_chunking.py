from app.rag.chunking import chunk_documents, split_text
from app.rag.loaders import Document

MARKDOWN = """# Typologies

Intro paragraph.

## Structuring

Cash deposits just under the threshold.

### Red flags

Many branches on the same day.

## Funnel accounts

Deposits in many states.
"""


def test_sections_carry_heading_path():
    doc = Document(text=MARKDOWN, source="t.md", title="Typologies", is_markdown=True)
    chunks = chunk_documents([doc], size=900, overlap=0)

    assert [c.section for c in chunks] == ["", "Structuring", "Structuring > Red flags", "Funnel accounts"]
    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3]
    assert chunks[2].embedding_text.startswith("Typologies > Structuring > Red flags\n")


def test_split_respects_size_and_overlaps():
    paragraphs = [f"Paragraph {i} " + "word " * 30 for i in range(10)]
    chunks = split_text("\n\n".join(paragraphs), size=400, overlap=60)

    assert len(chunks) > 1
    # Each chunk may exceed the size only by the carried-over overlap.
    assert all(len(c) <= 400 + 60 for c in chunks)
    # The start of each chunk repeats the end of the previous one.
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.split("\n\n")[0] in prev


def test_long_paragraph_is_hard_split():
    chunks = split_text("x" * 5 + " word" * 500, size=200, overlap=0)
    assert all(len(c) <= 200 for c in chunks)
