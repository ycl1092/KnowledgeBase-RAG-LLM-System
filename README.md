# RAG 知识库问答系统

基于 FastAPI + Chroma + DeepSeek 的工程化 RAG 知识库问答系统。支持多格式文档上传、向量检索、Rerank 重排序、RAGAS 指标评测。

## 技术栈

后端框架: FastAPI | 前端界面: Streamlit | 向量数据库: Chroma
Embedding: BAAI/bge-small-zh-v1.5 | Rerank: BAAI/bge-reranker-v2-m3
LLM: DeepSeek v4 Flash | 评测: RAGAS | 容器化: Docker

## 功能

- 多格式文档上传（TXT/MD/PDF/DOCX/JSON/图片）
- 智能文本分块 + Chroma 向量化 + MD5 去重
- 语义检索（Top-K 可配置）
- Rerank 重排序（CrossEncoder，按需开启）
- DeepSeek LLM 生成回答（流式输出）
- 对话历史管理
- RAGAS 量化评估（Faithfulness/Context Precision/Context Recall）
- Docker 一键部署

## 快速开始

```
pip install -r requirements.txt
cp .env.example .env  # 编辑填入 API Key
uvicorn app.main:app --host 0.0.0.0 --port 8000  # 启动后端
streamlit run frontend/app_chat.py  # 启动前端
```

Docker 部署:
```
docker-compose up -d
```

## API

- GET /api/v1/health - 健康检查
- POST /api/v1/upload/text - 上传文本
- POST /api/v1/upload/file - 上传文件
- POST /api/v1/upload/ingest - 批量导入
- POST /api/v1/chat - RAG 问答
- GET /api/v1/status - 系统状态

## 评测结果

Faithfulness: 0.88 | Context Precision: 0.90 | Context Recall: 0.81

## 项目结构

app/ 核心代码 (main.py FastAPI入口, api/routes.py API路由, core/ 配置与日志, models/ LLM客户端, rag/ 检索服务与重排序)
frontend/ Streamlit界面 (app_chat.py 聊天, app_upload.py 上传)
config/rag.yaml 配置文件
tests/ 评测脚本与测试数据
Dockerfile + docker-compose.yml 容器化部署

## License: MIT
