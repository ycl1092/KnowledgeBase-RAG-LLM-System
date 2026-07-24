"""
RAG 检索增强生成服务

整合检索 + Rerank + Prompt 模板 + LLM 生成。
"""

from app.core.config import settings
from app.core.logger import logger
from app.models.llm_client import llm
from app.rag.vector_store import vector_store
from app.rag.reranker import reranker


class RagService:
    """RAG 服务"""

    SYSTEM_PROMPT = """你是一位专业的客服助手，根据提供的参考资料回答问题。

规则：
1. 仅根据参考资料回答，不要编造信息
2. 如果参考资料中没有相关信息，请说"根据现有资料，我没有找到相关信息"
3. 回答简洁、专业，使用中文
4. 可以引用信息来源（产品名称或文档名）"""

    def __init__(self):
        self.top_k = settings.get("retrieval.top_k", 5)
        self.candidate_k = settings.get("reranker.candidate_k", self.top_k * 3)
        self.llm = llm
        self.reranker = reranker

    def _retrieve(self, question: str) -> list:
        """检索 + 可选重排序"""
        candidate_k = self.candidate_k if self.reranker.enabled else self.top_k
        docs = vector_store.similarity_search_with_score(question, k=candidate_k)
        docs = self.reranker.rerank(question, docs)
        return docs

    def query(self, question: str, history: list[dict] = None) -> str:
        """单轮 RAG 查询"""
        logger.info(f"RAG 查询: {question[:80]}")

        docs = self._retrieve(question)

        if not docs:
            logger.warning("未检索到相关文档")
            context = "（无相关参考资料）"
        else:
            context_parts = []
            for doc, score in docs:
                source = doc.metadata.get("source", "未知")
                context_parts.append(
                    f"[来源: {source} | 相关度: {score:.3f}]\n{doc.page_content}"
                )
            context = "\n\n---\n\n".join(context_parts)
            logger.info(f"检索到 {len(docs)} 个相关片段")

        messages = self._build_messages(question, context, history or [])
        answer = self.llm.chat(messages)
        return answer

    def query_stream(self, question: str, history: list[dict] = None):
        """流式 RAG 查询"""
        docs = self._retrieve(question)

        if not docs:
            context = "（无相关参考资料）"
        else:
            context_parts = []
            for doc, score in docs:
                source = doc.metadata.get("source", "未知")
                context_parts.append(f"[来源: {source}]\n{doc.page_content}")
            context = "\n\n---\n\n".join(context_parts)

        messages = self._build_messages(question, context, history or [])
        yield from self.llm.chat_stream(messages)

    def _build_messages(
        self, question: str, context: str, history: list[dict]
    ) -> list[dict]:
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        for msg in history[-6:]:
            messages.append(msg)
        messages.append({
            "role": "user",
            "content": f"【参考资料】\n{context}\n\n【问题】\n{question}\n\n回答：",
        })
        return messages


rag = RagService()
