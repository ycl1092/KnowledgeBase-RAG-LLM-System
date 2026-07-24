"""
API 路由

健康检查、知识库上传、RAG 问答。
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from app.core.logger import logger
from app.rag.rag_service import rag
from app.rag.knowledge_base import knowledge_base
from app.rag.vector_store import vector_store

router = APIRouter()


# ── 数据模型 ──

class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    answer: str
    sources: int = 0


class UploadResponse(BaseModel):
    status: str
    detail: str


# ── 健康检查 ──

@router.get("/health")
def health():
    return {
        "status": "ok",
        "doc_count": vector_store.count,
        "llm_ready": rag.llm.is_ready,
    }


# ── 知识库上传 ──

@router.post("/upload/text", response_model=UploadResponse)
def upload_text(text: str, source: str = "manual"):
    try:
        result = knowledge_base.upload_text(text, source=source)
        return UploadResponse(status="success" if "success" in result else "skipped", detail=result)
    except Exception as e:
        logger.error(f"上传文本失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload/file", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        # 由 knowledge_base 自动识别格式
        result = knowledge_base.upload_bytes(content, file.filename or 'unknown')
        return UploadResponse(status="success" if "success" in result else "skipped", detail=result)
    except Exception as e:
        logger.error(f"上传文件失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload/ingest", response_model=UploadResponse)
def ingest_directory(dir_path: str = ""):
    try:
        results = knowledge_base.ingest_directory(dir_path if dir_path else None)
        return UploadResponse(status="success", detail=str(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── RAG 问答 ──

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        answer = rag.query(req.question, req.history)
        return ChatResponse(answer=answer, sources=vector_store.count)
    except Exception as e:
        logger.error(f"RAG 查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/status")
def status():
    return {
        "doc_count": vector_store.count,
        "llm_ready": rag.llm.is_ready,
        "model": rag.llm.model,
        "top_k": rag.top_k,
    }
