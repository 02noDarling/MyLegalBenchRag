from retrieval_strategies.chunking import Chunk
from typing import Literal

class MyAIEmbeddingModel:
    def __init__(self,embedding_model_name):
        self.embedding_model_name=embedding_model_name
    
    #看这里！！！！！！
    #只需要你根据你的embedding_model_name(embedding策略名字)实现下面两个这个函数即可
    #需要加别的头文件的话，自己加就行了

    #下面这个函数作用就是把list[Chunk]，转化为list[list[float]]，就是向量化，注意第一个函数是将数据集里的文档切割好向量化，第二个函数是将一个询问query:str向量化
    def get_embeddings_from_chunklist(self,chunks:list[Chunk]) ->list[list[float]]:
        result=[[ 0 for j in range(384) ] for j in range(len(chunks))] #我这里图方便直接返回len(chunks)长度的384维全0向量,可自行注释掉

        # if embedding_model_name=="your-embedding1":
        #     for chunk in chunks:
        #         chunk.content:str
        #         将chunk.content这个字符串向量化即可,然后加入results列表
        # elif embedding_model_name=="your-embedding2":
        #     ...


        return result
    
    #将query:str向量化
    def get_embedding_from_query(self,query:str)->list[float]:
        result=[ 0 for i in range(384) ] #我这里图方便直接返回384维全0向量,可自行注释掉

        # if embedding_model_name=="your-embedding1":
        #     query:str
        #     将query这个字符串向量化即可,然后加入result
        # elif embedding_model_name=="your-embedding2":
        #     ...

        return result
