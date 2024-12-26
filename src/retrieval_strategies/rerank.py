class MyAIRerankModel:
    def __init__(self,rerank_model_name):
        self.rerank_model_name=rerank_model_name


    #看这里！！！！！！
    #只需要你根据你的rerank_model_name(rerank策略名字)实现下面两个这个函数即可
    #需要加别的头文件的话，自己加就行了

    # 下面这个函数的作用是给定查询query和要rerank的texts字符串列表,基于你的策略返回rerank后的索引下标列表
    def get_reranked_indices_from_texts_by_query(self,query:str,texts:list[str]):
        reranked_indices:list[int]=[]  #我这里图方便直接返回[1,2,3,4,....,len(texts)-1],相当于没重排过，可以自行注释掉
        for i in range(len(texts)):
            reranked_indices.append(i)

        # if rerank_model_name=="your-rerank1":
            # 利用你的方法返回reranked_indices:list[int]这个重排后的索引下标列表
        # elif rerank_model_name=="your-rerank2":
        #     ...

        return reranked_indices
    