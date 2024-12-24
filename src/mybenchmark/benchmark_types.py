from abc import ABC, abstractmethod
from collections.abc import Sequence

from pydantic import BaseModel, computed_field, model_validator
from typing_extensions import Self

class Snippet(BaseModel):
    file_path: str
    span: tuple[int, int]

    @computed_field  # type: ignore[misc]
    @property
    def answer(self) -> str:
        with open(f"./src/data/corpus/{self.file_path}") as f:
            return f.read()[self.span[0] : self.span[1]]


def validate_snippet_list(snippets: Sequence[Snippet]) -> None:
    snippets_by_file_path: dict[str, list[Snippet]] = {}
    for snippet in snippets:
        if snippet.file_path not in snippets_by_file_path:
            snippets_by_file_path[snippet.file_path] = [snippet]
        else:
            snippets_by_file_path[snippet.file_path].append(snippet)

    for _file_path, snippets in snippets_by_file_path.items():
        snippets = sorted(snippets, key=lambda x: x.span[0])
        for i in range(1, len(snippets)):
            if snippets[i - 1].span[1] >= snippets[i].span[0]:
                raise ValueError(
                    f"Spans are not disjoint! {snippets[i - 1].span} VS {snippets[i].span}"
                )


class QAGroundTruth(BaseModel):
    query: str
    snippets: list[Snippet]
    tags: list[str] = []

    @model_validator(mode="after")
    def validate_snippet_spans(self) -> Self:
        validate_snippet_list(self.snippets)
        return self


class Benchmark(BaseModel):
    tests: list[QAGroundTruth]


# Types for benchmarking a method


class Document(BaseModel):
    file_path: str
    content: str
