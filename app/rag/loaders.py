from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


@dataclass
class Document:
    text: str
    source: str  # path relative to the docs dir, used as a stable ID
    title: str
    page: int | None = None
    is_markdown: bool = False


def load_directory(docs_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() in SUPPORTED_SUFFIXES and path.is_file():
            docs.extend(load_file(path, docs_dir))
    return docs


def load_file(path: Path, docs_dir: Path) -> list[Document]:
    source = path.relative_to(docs_dir).as_posix()
    if path.suffix.lower() == ".pdf":
        return _load_pdf(path, source)
    text = path.read_text(encoding="utf-8")
    return [
        Document(
            text=text,
            source=source,
            title=_markdown_title(text) or path.stem.replace("_", " ").title(),
            is_markdown=path.suffix.lower() == ".md",
        )
    ]


def _load_pdf(path: Path, source: str) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    meta_title = reader.metadata.title if reader.metadata and reader.metadata.title else None
    title = meta_title or path.stem.replace("_", " ").title()
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Document(text=text, source=source, title=title, page=number))
    return pages


def _markdown_title(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None
