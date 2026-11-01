import re
from dataclasses import dataclass

# gpt-oss models cite as 【1】 or 【1†source】 instead of [1].
WIDE_CITATION = re.compile(r"【(\d+)(?:†[^】]*)?】")


@dataclass
class Source:
    number: int
    title: str
    source: str
    section: str
    page: int | None
    score: float
    text: str


CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def normalize_citations(text: str) -> str:
    return WIDE_CITATION.sub(r"[\1]", text)


def cited_sources(answer: str, sources: list[Source]) -> list[Source]:
    """The sources the answer cites ([2], [1][3] or [1, 3]); all of them if it cites none."""
    cited = {int(n) for group in CITATION.findall(answer) for n in group.split(",")}
    return [s for s in sources if s.number in cited] or sources
