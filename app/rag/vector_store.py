"""
向量存储封装层

封装 Chroma，提供统一的 add / search / delete 接口。
"""

from typing import Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from app.core.config import settings
from app.core.logger import logger


class VectorStoreService:
    """Chroma 向量存储服务"""

    def __init__(self):
        self.collection_name = settings.get("chroma.collection_name", "knowledge_base")
        self.persist_dir = settings.get("chroma.persist_directory", "data/chroma_db")

        model_name = settings.get("embedding.model", "BAAI/bge-small-zh-v1.5")
        device = settings.get("embedding.device", "cpu")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )

        logger.info(
            f"向量库已加载: collection={self.collection_name}, "
            f"文档数={self.count}"
        )

    @property
    def count(self) -> int:
        try:
            return self.store._collection.count()
        except Exception:
            return 0

    def add_texts(self, texts: list[str], metadatas: Optional[list[dict]] = None) -> list[str]:
        """添加文本到向量库"""
        ids = self.store.add_texts(texts=texts, metadatas=metadatas)
        logger.info(f"向量库新增 {len(ids)} 条记录")
        return ids

    def similarity_search(self, query: str, k: int = 3) -> list[Document]:
        """相似度搜索"""
        return self.store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 3) -> list[tuple[Document, float]]:
        """相似度搜索（带分数）"""
        return self.store.similarity_search_with_score(query, k=k)

    def get_retriever(self, k: Optional[int] = None) -> VectorStoreRetriever:
        """获取检索器"""
        k = k or settings.get("retrieval.top_k", 3)
        return self.store.as_retriever(search_kwargs={"k": k})

    def delete_collection(self):
        """清空集合"""
        try:
            self.store.delete_collection()
            logger.info("向量库已清空")
        except Exception as e:
            logger.error(f"清空向量库失败: {e}")


vector_store = VectorStoreService()
