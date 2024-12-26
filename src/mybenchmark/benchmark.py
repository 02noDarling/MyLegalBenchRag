from benchmark_types import Benchmark, Document, QAGroundTruth
from retrieval_strategies.retrieval_strategies import RETRIEVAL_STRATEGIES
from database.mydatabase import MyDataBase
import random,os
import datetime as dt
from benchmark_result import MyBenchmarkResult
import pandas as pd

benchmark_name_to_weight: dict[str, float] = {
    "privacy_qa": 0.25,
    "contractnli": 0.25,
    "maud": 0.25,
    "cuad": 0.25,
}

# This takes a random sample of the benchmark, to speed up query processing.
# p-values can be calculated, to statistically predict the theoretical performance on the "full test"
MAX_TESTS_PER_BENCHMARK = 194
# This sorts the tests by document,
# so that the MAX_TESTS_PER_BENCHMARK tests are over the fewest number of documents,
# This speeds up ingestion processing, but
# p-values cannot be calculated, because this settings drastically reduces the search space size.
SORT_BY_DOCUMENT = True

def main():
    all_tests: list[QAGroundTruth] = []
    weights: list[float] = []
    document_file_paths_set: set[str] = set()
    used_document_file_paths_set: set[str] = set()
    
    #读文件
    for benchmark_name, weight in benchmark_name_to_weight.items():
        with open(f"./src/data/benchmarks/{benchmark_name}.json") as f:
            benchmark = Benchmark.model_validate_json(f.read())
            tests = benchmark.tests
            document_file_paths_set |= {
                snippet.file_path for test in tests for snippet in test.snippets
            }
            # Cap queries for a given benchmark
            if len(tests) > MAX_TESTS_PER_BENCHMARK:
                if SORT_BY_DOCUMENT:
                    # Use random based on file path seed, rather than the file path itself, to prevent bias.
                    tests = sorted(
                        tests,
                        key=lambda test: (
                            random.seed(test.snippets[0].file_path),
                            random.random(),
                        )[1],
                    )
                else:
                    # Keep seed consistent, for better caching / testing.
                    random.seed(benchmark_name)
                    random.shuffle(tests)
                tests = tests[:MAX_TESTS_PER_BENCHMARK]
            used_document_file_paths_set |= {
                snippet.file_path for test in tests for snippet in test.snippets
            }
            for test in tests:
                test.tags = [benchmark_name]
            all_tests.extend(tests)
            weights.extend([weight / len(tests)] * len(tests))
    benchmark = Benchmark(
        tests=all_tests,
    )

    corpus: list[Document] = []
    for document_file_path in sorted(
        document_file_paths_set
        if not SORT_BY_DOCUMENT
        else used_document_file_paths_set
    ):
        with open(f"./src/data/corpus/{document_file_path}") as f:
            corpus.append(
                Document(
                    file_path=document_file_path,
                    content=f.read(),
                )
            )
    
    #创建benchmark_results文件夹
    run_name = dt.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    benchmark_path = f"./src/benchmark_results/{run_name}"
    os.makedirs(benchmark_path, exist_ok=True)

    #循环每一种检索策略
    rows: list[dict[str, str | None | int | float]] = []
    for i, retrieval_strategy in enumerate(RETRIEVAL_STRATEGIES):
        #chunking
        chunks=retrieval_strategy.chunking_strategy.get_chunks_by_documents(corpus)
        
        #embedding
        dataset_embeddings=retrieval_strategy.embedding_model.get_embeddings_from_chunklist(chunks)
        
        #创建数据库，并将向量插入数据库
        sqlite_db_file_path="./src/data/cache/baseline.db"
        database=MyDataBase(sqlite_db_file_path=sqlite_db_file_path,embedding_size=len(dataset_embeddings[0]))
        database.insert_embeddings_into_database(chunks=chunks,dataset_embeddings=dataset_embeddings)
        
        #retrieval rerank get_result
        benchmarkresults=MyBenchmarkResult(all_tests=all_tests,chunks=chunks,retrieval_strategy=retrieval_strategy,database=database)

        row: dict[str, str | None | int | float] = {
            "i": i,
            "chunk_strategy_name": retrieval_strategy.chunking_strategy.strategy_name,
            "chunk_size": retrieval_strategy.chunking_strategy.chunk_size,
            "embedding_model": retrieval_strategy.embedding_model.embedding_model_name,
            "top_k": retrieval_strategy.embedding_topk,
            "rerank_model": retrieval_strategy.rerank_model.rerank_model_name
            if retrieval_strategy.rerank_model is not None
            else None,
            "top_k_rerank": retrieval_strategy.rerank_topk,
            "token_limit": retrieval_strategy.token_limit,
        }

        total_precision:float=0
        total_recall:float=0

        for benchmark_name, weight in benchmark_name_to_weight.items():
            precision,recall=benchmarkresults.get_result_from_benchmark_name(benchmark_name=benchmark_name)
            row[f"{benchmark_name}|precision"] = precision
            row[f"{benchmark_name}|recall"] = recall
            total_precision+=weight*precision
            total_recall+=weight*recall
        
        row["total_precision"] = total_precision
        row["total_recall"] = total_recall
        rows.append(row)
    
    #save_result
    df = pd.DataFrame(rows)
    df.to_csv(f"{benchmark_path}/results.csv", index=False)

    print(f'All Benchmark runs saved to: "{benchmark_path}"')
if __name__=="__main__":
    main()