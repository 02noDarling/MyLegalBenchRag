from benchmark_types import QAGroundTruth,Snippet
from retrieval_strategies.chunking import Chunk
from retrieval_strategies.retrieval_strategies import MyRetrievalStrategy
from database.mydatabase import MyDataBase

class QAResult:
    def __init__(self,qa_gt: QAGroundTruth, retrieved_snippets: list[Snippet]):
        self.qa_gt=qa_gt
        self.retrieved_snippets=retrieved_snippets
    
    def precision(self) -> float:
        total_retrieved_len = 0
        relevant_retrieved_len = 0
        for snippet in self.retrieved_snippets:
            total_retrieved_len += snippet.span[1] - snippet.span[0]
            # It's guaranteed that gt_snippets don't overlap
            for gt_snippet in self.qa_gt.snippets:
                if snippet.file_path == gt_snippet.file_path:
                    common_min = max(snippet.span[0], gt_snippet.span[0])
                    common_max = min(snippet.span[1], gt_snippet.span[1])
                    if common_max > common_min:
                        relevant_retrieved_len += common_max - common_min
        # print("total_retrieved_len=%f relevant_retrieved_len=%f"%(total_retrieved_len,relevant_retrieved_len))
        if total_retrieved_len == 0:
            return 0
        return relevant_retrieved_len / total_retrieved_len

    def recall(self) -> float:
        total_relevant_len = 0
        relevant_retrieved_len = 0
        for gt_snippet in self.qa_gt.snippets:
            total_relevant_len += gt_snippet.span[1] - gt_snippet.span[0]
            # It's guaranteed that gt_snippets don't overlap
            for snippet in self.retrieved_snippets:
                if snippet.file_path == gt_snippet.file_path:
                    common_min = max(snippet.span[0], gt_snippet.span[0])
                    common_max = min(snippet.span[1], gt_snippet.span[1])
                    if common_max > common_min:
                        relevant_retrieved_len += common_max - common_min
        if total_relevant_len == 0:
            return 0
        # print("here=%f"%(relevant_retrieved_len / total_relevant_len))
        return relevant_retrieved_len / total_relevant_len

class MyBenchmarkResult:
    def __init__(self,all_tests:list[QAGroundTruth],chunks:list[Chunk],retrieval_strategy:MyRetrievalStrategy,database:MyDataBase)->None:
        self.all_tests=all_tests
        self.chunks=chunks
        self.retrieval_strategy=retrieval_strategy
        self.database=database

    def get_result_from_benchmark_name(self,benchmark_name:str)->tuple[float, float]:
        selected_test:list[QAGroundTruth]=[]
        for test in self.all_tests:
            if test.tags[0]==benchmark_name:
                selected_test.append(test)
        
        total_precision:float=0
        total_recall:float=0
        for test in selected_test:
            query=test.query

            #retrieval
            indices=self.database.get_snippets_indices_from_query(
                query=query,
                embedding_model=self.retrieval_strategy.embedding_model,
                embedding_topk=self.retrieval_strategy.embedding_topk
            )
            snippets=[Snippet(file_path=self.chunks[index].document_id,span=self.chunks[index].span) for index in indices]

            #rerank
            if self.retrieval_strategy.rerank_model !=None:
                reranked_indices=self.retrieval_strategy.rerank_model.get_reranked_indices_from_texts_by_query(query=query,texts=[self.chunks[index].content for index in indices])
                new_snippets=[snippets[index] for index in reranked_indices[:self.retrieval_strategy.rerank_topk]]
                snippets=new_snippets
            #get_result
            # snippets=test.snippets


            qaresult=QAResult(qa_gt=test,retrieved_snippets=snippets)
            total_precision+=qaresult.precision()/len(selected_test)
            total_recall+=qaresult.recall()/len(selected_test)

        return (total_precision,total_recall)



