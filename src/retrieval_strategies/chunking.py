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
    
    #看这里！！！！！！
    #只需要你根据你的strategy_name(chunking策略名字)和chunk_size实现下面这个函数即可
    #下面这个函数作用就是把list[Document]文件列表，转化为list[Chunk]，chunk列表
    #chunk类已经定义在上方
    #需要加别的头文件的话，自己加就行了
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

                # 注意！！！！！！！
                #在这里实现你的方法，其余不用动
                # case "你的策略名1":
                #     pass
                # case "你的策略名2":
                #     pass
                # ...
                #type(document.conent)=str,你要做的就是把documnet.content分割，加入text_splits: list[str]即可
                #确保当前分割结果在text_splits中


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
