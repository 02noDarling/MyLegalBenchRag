from typing import Literal

from retrieval_strategies.chunking import MyChunkingStrategy
from retrieval_strategies.embedding import MyAIEmbeddingModel
from retrieval_strategies.rerank import MyAIRerankModel

chunk_strategy_names: list[Literal["naive", "rcts"]] = [
    "naive",
    "rcts"
]
embedding_models:list[MyAIEmbeddingModel]=[
    MyAIEmbeddingModel(embedding_model_name="hudaili-embedding")
]
rerank_models: list[MyAIEmbeddingModel | None] = [
    # None
    # AIRerankModel(company="cohere", model="rerank-english-v3.0"),
    MyAIRerankModel(rerank_model_name="hudaili-rerank")
]
chunk_sizes: list[int] = [500]
top_ks: list[int] = [1, 2, 4, 8, 16, 32, 64]

class MyRetrievalStrategy:
    def __init__(self,chunking_strategy,embedding_model,embedding_topk,rerank_model,rerank_topk,token_limit):
        self.chunking_strategy=chunking_strategy
        self.embedding_model=embedding_model 
        self.embedding_topk=embedding_topk
        self.rerank_model=rerank_model
        self.rerank_topk=rerank_topk
        self.token_limit=token_limit

RETRIEVAL_STRATEGIES: list[MyRetrievalStrategy] = []
for chunk_strategy_name in chunk_strategy_names:
    for chunk_size in chunk_sizes:
        chunking_strategy = MyChunkingStrategy(
            strategy_name=chunk_strategy_name,
            chunk_size=chunk_size
        )
        for embedding_model in embedding_models:
            for rerank_model in rerank_models:
                for top_k in top_ks:
                    RETRIEVAL_STRATEGIES.append(
                        MyRetrievalStrategy(
                            chunking_strategy=chunking_strategy,
                            embedding_model=embedding_model,
                            embedding_topk=300 if rerank_model is not None else top_k,
                            rerank_model=rerank_model,
                            rerank_topk=top_k,
                            token_limit=None,
                        )
                    )
