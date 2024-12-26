from typing import Literal

from retrieval_strategies.chunking import MyChunkingStrategy
from retrieval_strategies.embedding import MyAIEmbeddingModel
from retrieval_strategies.rerank import MyAIRerankModel

#这里给你的chunking策略取个名字
chunk_strategy_names: list[str] = [
    "naive",
    "rcts"
    #naive和rcts这两个策略是已经现实好的,具体代码在src/retrieval_strategies/chunking.py下，你也可以注释掉不用
    #下面自行添加你的策略名字
    #"你的策略1",
    #"你的策略2" ...
]

#这里自己设置chunk_size,可以是[100,200,300,...]
chunk_sizes: list[int] = [500]

#这里给你的embedding策略取个名字
embedding_models:list[MyAIEmbeddingModel]=[
    MyAIEmbeddingModel(embedding_model_name="hudaili-embedding")
    #以上是示例
    #下面自行添加你的策略名字
    #MyAIEmbeddingModel(embedding_model_name="your-embedding1")
    #MyAIEmbeddingModel(embedding_model_name="your-embedding2") ...
]

#这里给你的rerank策略取个名字，也可以选择不用rerank
rerank_models: list[MyAIEmbeddingModel | None] = [
    None
    # MyAIRerankModel(rerank_model_name="hudaili-rerank")
    #以上是示例 
    #下面自行添加你的策略名字
    # MyAIRerankModel(rerank_model_name="your-rerank1")
    # MyAIRerankModel(rerank_model_name="your-rerank2") ...
]

#这里自己设置top_ks,表示最终选取检索片段的数量
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
