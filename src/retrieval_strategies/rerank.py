class MyAIRerankModel:
    def __init__(self,rerank_model_name):
        self.rerank_model_name=rerank_model_name

    def get_reranked_indices_from_texts_by_query(self,query:str,texts:list[str]):
        reranked_indices:list[int]=[]
        for i in range(len(texts)):
            reranked_indices.append(i)
        return reranked_indices
    
    # def get_reranklist_from_embeddings(self,embeddings)