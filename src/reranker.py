from sentence_transformers import CrossEncoder
import numpy as np

class Reranker:
    # 针对初筛结果进行精排
    def __init__(self, model_name='BAAI/bge-reranker-base'):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, top_n=10):
        if not documents:
            return []

        # 构造问题片段，进行打分
        sentence_pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(sentence_pairs)

        # 按分数从高到低排序
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # 选取前n个最相关的片段
        reranked_docs = [documents[i] for i, score in indexed_scores[:top_n]]
        return reranked_docs