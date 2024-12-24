from benchmark_types import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Chunk:
    def __init__(self,document_id,span,content):
        self.document_id: str=document_id
        self.span: tuple[int, int]=span 
        self.content: str=content

class MyChunkingStrategy:
    def __init__(self,strategy_name,chunk_size):
        self.strategy_name=strategy_name
        self.chunk_size=chunk_size
    
    def get_chunks_by_documents(self,raw_documents:list[Document])->list[Chunk]:
        documents: dict[str, Document]={}
        for document in raw_documents:
            documents[document.file_path] = document
        chunks: list[Chunk] = []
        for document_id, document in documents.items():
            # Get chunks
            chunk_size = self.chunk_size
            match self.strategy_name:
                case "naive":
                    text_splits: list[str] = []
                    for i in range(0, len(document.content), chunk_size):
                        text_splits.append(document.content[i : i + chunk_size])
                case "rcts":
                    synthetic_data_splitter = RecursiveCharacterTextSplitter(
                        separators=[
                            "\n\n",
                            "\n",
                            "!",
                            "?",
                            ".",
                            ":",
                            ";",
                            ",",
                            " ",
                            "",
                        ],
                        chunk_size=chunk_size,
                        chunk_overlap=0,
                        length_function=len,
                        is_separator_regex=False,
                        strip_whitespace=False,
                    )
                    text_splits = synthetic_data_splitter.split_text(document.content)
            assert sum(len(text_split) for text_split in text_splits) == len(
                document.content
            )
            assert "".join(text_splits) == document.content

            # Get spans from chunks
            prev_span: tuple[int, int] | None = None
            for text_split in text_splits:
                prev_index = prev_span[1] if prev_span is not None else 0
                span = (prev_index, prev_index + len(text_split))
                chunks.append(
                    Chunk(
                        document_id=document_id,
                        span=span,
                        content=text_split
                    )
                )
                prev_span = span
        return chunks
