from retrieval_strategies.chunking import Chunk
from typing import Literal

class MyAIEmbeddingModel:
    def __init__(self,embedding_model_name):
        self.embedding_model_name=embedding_model_name
    
    def get_embeddings_from_chunklist(self,chunks:list[Chunk]) ->list[list[float]]:
        result=[[ 0 for j in range(384) ] for j in range(len(chunks))]
        return result
    
    def get_embedding_from_query(self,query:str)->list[float]:
        result=[ 0 for i in range(384) ] 
        return result
