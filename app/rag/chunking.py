import re
from dataclasses import dataclass

from app.rag.loaders import Document

HEADING = re.compile(r"^(#{1,4})\s+(.*)$")


@dataclass
class Chunk:
    text: str
    source: str
    title: str
    section: str
    page: int | None
    chunk_index: int

    @property
    def embedding_text(self) -> str:
        # Prefixing title/section gives short chunks enough context to be found.
        header = " > ".join(part for part in (self.title, self.section) if part)
        return f"{header}\n{self.text}" if header else self.text


def chunk_documents(docs: list[Document], size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    counters: dict[str, int] = {}
    for doc in docs:
        sections = _markdown_sections(doc.text) if doc.is_markdown else [("", doc.text)]
        for section, body in sections:
            for piece in split_text(body, size, overlap):
                index = counters.get(doc.source, 0)
                counters[doc.source] = index + 1
                chunks.append(Chunk(piece, doc.source, doc.title, section, doc.page, index))
    return chunks


def _markdown_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown on headings, labelling each part with its heading path ("A > B")."""
    path: list[tuple[int, str]] = []
    sections: list[tuple[str, str]] = []
    current: list[str] = []

    def flush():
        body = "\n".join(current).strip()
        if body:
            # The document title (level 1) is already stored separately.
            sections.append((" > ".join(name for level, name in path if level > 1), body))
        current.clear()

    for line in text.splitlines():
        match = HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            path = [(lvl, name) for lvl, name in path if lvl < level] + [(level, match.group(2).strip())]
        else:
            current.append(line)
    flush()
    return sections


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Pack paragraphs into chunks of at most ~size chars, carrying `overlap` chars forward."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs:
        pieces.extend(_hard_split(paragraph, size) if len(paragraph) > size else [paragraph])

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > size:
            chunks.append(current)
            current = _tail(current, overlap)
        current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def _hard_split(text: str, size: int) -> list[str]:
    words = text.split()
    parts, current = [], ""
    for word in words:
        if current and len(current) + len(word) + 1 > size:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        parts.append(current)
    return parts


def _tail(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return ""
    tail = text[-overlap:]
    # Start on a word boundary so the overlap doesn't begin mid-word.
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else tail
