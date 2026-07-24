"""
RAG 知识库系统 — FastAPI 服务入口

启动: uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logger import logger
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    logger.info("=" * 50)
    logger.info("RAG 知识库系统启动")
    logger.info(f"Root: {settings.ROOT_DIR}")
    logger.info(f"LLM: {settings.LLM_MODEL} @ {settings.LLM_BASE_URL}")
    logger.info(f"LLM 就绪: {True if settings.LLM_API_KEY else False}")
    logger.info("=" * 50)
    yield
    logger.info("RAG 知识库系统关闭")


app = FastAPI(
    title="KnowledgeBase RAG System",
    description="生产级 RAG 知识库问答系统",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.get("server.host", "0.0.0.0"),
        port=settings.get("server.port", 8000),
        reload=True,
    )
