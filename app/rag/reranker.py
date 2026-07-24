"""
重排序服务

用 CrossEncoder 对检索结果重新打分，提升精度。
"""

from app.core.config import settings
from app.core.logger import logger


class RerankerService:
    """Cross-Encoder 重排序"""

    def __init__(self):
        self.model = None
        self.model_name = settings.get("reranker.model", "BAAI/bge-reranker-v2-m3")
        self.enabled = settings.get("reranker.enabled", False)
        self.top_k = settings.get("retrieval.top_k", 5)

        if self.enabled:
            self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Reranker 模型已加载: {self.model_name}")
        except Exception as e:
            logger.warning(f"Reranker 加载失败 ({e})，跳过重排序")
            self.enabled = False

    def rerank(self, query: str, docs_with_scores: list) -> list:
        """对候选文档重排序，返回前 top_k 个"""
        if not self.enabled or not docs_with_scores or self.model is None:
            return docs_with_scores[:self.top_k]

        pairs = [(query, doc.page_content) for doc, _ in docs_with_scores]
        scores = self.model.predict(pairs)

        scored = list(zip(docs_with_scores, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        logger.info(
            f"Rerank: {len(docs_with_scores)} 候选 → "
            f"最高分 {scored[0][1]:.3f}，最低分 {scored[-1][1]:.3f}"
        )
        return [(doc, float(score)) for (doc, _), score in scored[:self.top_k]]


reranker = RerankerService()
