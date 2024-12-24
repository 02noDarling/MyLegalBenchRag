from retrieval_strategies.chunking import Chunk
import os
import sqlite3
import sqlite_vec  # type: ignore
from tqdm import tqdm
import struct
from retrieval_strategies.embedding import MyAIEmbeddingModel
from benchmark_types import Snippet
from typing import cast

def serialize_f32(vector: list[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack(f"{len(vector)}f", *vector)

class MyDataBase:
    def __init__(self,sqlite_db_file_path,embedding_size):
        self.sqlite_db_file_path=sqlite_db_file_path
        if os.path.exists(self.sqlite_db_file_path):
            os.remove(self.sqlite_db_file_path)
        
        directory = os.path.dirname(sqlite_db_file_path)
        if not os.path.exists(directory):
            print(f"Directory does not exist. Creating directory: {directory}")
            os.makedirs(directory)  # 创建目录

        self.sqlite_db = sqlite3.connect(self.sqlite_db_file_path)
        self.sqlite_db.enable_load_extension(True)
        sqlite_vec.load(self.sqlite_db)
        self.sqlite_db.enable_load_extension(False)
        # Set RAM Usage and create vector table
        self.sqlite_db.execute(f"PRAGMA mmap_size = {3*1024*1024*1024}")
        self.sqlite_db.execute(
            f"CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[{embedding_size}])"
        )

    def insert_embeddings_into_database(self,chunks: list[Chunk], dataset_embeddings: list[list[float]]) -> None:
        # 初始化进度条
        progress_bar: tqdm | None = None
        progress_bar = tqdm(
            total=len(chunks), desc="Inserting Embeddings into database", ncols=100
        )

        batch_size = 1000  # 批量插入的大小，可以根据实际情况调整

        # 准备插入数据
        insert_data = [
            (i, serialize_f32(embedding))  # 假设 serialize_f32 是处理嵌入的序列化函数
            for i, embedding in enumerate(dataset_embeddings)
        ]

        with self.sqlite_db as db:
            # 分批次插入数据
            for i in range(0, len(insert_data), batch_size):
                batch = insert_data[i:i + batch_size]
                db.executemany(
                    "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
                    batch,
                )
                progress_bar.update(len(batch))  # 更新进度条，每插入一批数据更新一次

        # 完成后关闭进度条
        if progress_bar:
            progress_bar.close()

    def get_snippets_indices_from_query(self,query:str,embedding_model:MyAIEmbeddingModel,embedding_topk:int)->list[Snippet]:
        query_embedding=embedding_model.get_embedding_from_query(query)
        rows = self.sqlite_db.execute(
            """
            SELECT
                rowid,
                distance
            FROM vec_items
            WHERE embedding MATCH ?
            and k = ?
            ORDER BY distance ASC
            """,
            [serialize_f32(query_embedding), embedding_topk],
        ).fetchall()
        indices = [cast(int, row[0]) for row in rows]
        return indices