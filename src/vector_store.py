import shutil
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class VectorEngine:
    # 负责向量数据库的生命周期管理与高性能检索

    def __init__(self, db_path, embedding_model):
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    def rebuild_db(self, documents):
        # 清理并重建向量库
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)

        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        return vector_db

    def get_retriever(self, k):
        if not os.path.exists(self.db_path):
            raise FileNotFoundError("请先上传文件并索引")

        vector_db = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings
        )
        return vector_db.as_retriever(search_kwargs={"k": k})