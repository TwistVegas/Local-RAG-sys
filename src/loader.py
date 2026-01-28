from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import re


class DocumentProcessor:
    # 负责解析 PDF 并进行语义切分

    def __init__(self, chunk_size, chunk_overlap):
        # 这里的参数从ConfigManager读取后传入
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
            separators=["\n\n", "\n"]
        )

    def _clean_text(self, text):
        # 过滤常见页眉、页脚及参考文献干扰
        # 1. 过滤常见的 IEEE/arXiv 水印标识
        text = re.sub(r'Except for this watermark.*IEEE Xplore.*', '', text)
        text = re.sub(r'arXiv:\d+\.\d+v\d+ \[cs\..*\] \d+ \w+ \d+', '', text)

        # 2. 识别参考文献起始点
        # 如果当前块包含大量类似 [1], [2] 或 References 字符，可以标记或截断
        ref_keywords = ["References", "参考文献", "BIBLIOGRAPHY"]
        for kw in ref_keywords:
            if kw in text:
                text = text.split(kw)[0]

        return text.strip()

    def process_files(self, file_paths):
        all_docs = []
        for path in file_paths:
            loader = PyPDFLoader(path)
            docs = loader.load()
            for d in docs:
                d.page_content = self._clean_text(d.page_content)
                d.metadata["source"] = os.path.basename(path)

            docs = [d for d in docs if len(d.page_content) > 50]
            all_docs.extend(self.splitter.split_documents(docs))
        return all_docs