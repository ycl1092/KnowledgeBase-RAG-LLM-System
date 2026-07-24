<div align="center">

#  RAG 知识库问答系统

> 基于 FastAPI + Chroma + DeepSeek 的工程级 RAG 知识库问答系统

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-339933)](https://www.langchain.com/)
[![Chroma](https://img.shields.io/badge/Chroma-0.5%2B-FC6D26)](https://www.trychroma.com/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek%20v4%20Flash-4A90D9)](https://platform.deepseek.com/)
[![Docker](https://img.shields.io/badge/Docker-✅-2496ED)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

</div>

---

##  项目简介

一套完整的 RAG 检索增强生成系统，支持多格式文档上传、语义检索和 AI 回答生成。系统采用模块化架构，配置驱动，Docker 一键部署。

> **适用场景：** 企业知识库、产品说明书查询、智能客服问答

---

##  核心功能

| 功能 | 说明 |
|------|------|
|   **多格式上传** | 支持 TXT / MD / PDF / DOCX / JSON / 图片 |
| ⚡ **智能分块** | RecursiveCharacterTextSplitter，可调参数 |
|   **向量检索** | Chroma + BGE Embedding，Top-K 可配置 |
| ⚙️ **Rerank 重排序** | CrossEncoder 按需开启，提升检索精度 |
|   **LLM 生成** | DeepSeek v4 Flash，流式输出，自动重试 |
|   **对话记忆** | 多轮对话上下文理解 |
|   **RAGAS 评测** | 3 项指标量化评估系统质量 |
|   **Docker 部署** | 一键启动，开箱即用 |

---

##  架构

```
+-----------+     +-----------+     +-----------+
|  Streamlit |────▶|  FastAPI  |────▶|  Chroma   |
|  前端界面   |     |  REST API  |     |  向量库    |
+-----------+     +-----------+     +-----------+
      │                 │                 │
      │                 ▼                 ▼
      │          +-----------+     +-----------+
      │          |  DeepSeek |     |  BGE      |
      │          |  LLM 生成  |     | Embedding  |
      │          +-----------+     +-----------+
      │                 │
      ▼                 ▼
+-----------+     +-----------+
|  上传管理   |     |  Rerank   |
|  多格式解析  |     |  重排序    |
+-----------+     +-----------+
```

---

##  技术栈

| 层级 | 技术 | 选型理由 |
|------|------|---------|
| 后端 | FastAPI + Uvicorn | 高性能异步框架，自动生成 API 文档 |
| 前端 | Streamlit | 快速构建数据应用，支持流式输出 |
| 向量库 | Chroma | 轻量级本地持久化，无需额外服务 |
| Embedding | BGE-small-zh-v1.5 | 中文语义理解能力强，轻量级 |
| Rerank | BGE-reranker-v2-m3 | 交叉编码器，提升检索精度 |
| LLM | DeepSeek v4 Flash | 性价比高，OpenAI 兼容接口 |
| 评测 | RAGAS | 业界标准 RAG 评估框架 |
| 部署 | Docker + Compose | 环境隔离，一键部署 |

---

##  快速开始

### 环境要求

- Python 3.11+
- DeepSeek API Key（[申请地址](https://platform.deepseek.com/)）

### 安装与运行

```bash
# 1. 克隆仓库
git clone https://github.com/ycl1092/KnowledgeBase-RAG-LLM-System.git
cd KnowledgeBase-RAG-LLM-System

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 API Key

# 4. 启动后端
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 5. 新开终端，启动前端
streamlit run frontend/app_chat.py
```

浏览器打开 http://localhost:8501 即可使用。

### Docker 部署

```bash
docker-compose up -d
```

---

##  API 文档

| 方法 | 端点 | 说明 |
|------|------|------|
| `GET` | `/api/v1/health` | 健康检查 |
| `POST` | `/api/v1/upload/text` | 上传文本 |
| `POST` | `/api/v1/upload/file` | 上传文件（PDF/DOCX 等）|
| `POST` | `/api/v1/upload/ingest` | 批量导入目录 |
| `POST` | `/api/v1/chat` | RAG 问答 |
| `GET` | `/api/v1/status` | 系统状态 |

> 启动后端后，访问 http://localhost:8000/docs 可查看交互式 API 文档。

---

##  配置说明

系统配置集中在 `config/rag.yaml`，关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `llm.model` | `deepseek-v4-flash` | LLM 模型 |
| `llm.temperature` | `0.3` | 生成温度 |
| `llm.max_tokens` | `2048` | 最大回答长度 |
| `chunk.size` | `300` | 分块大小 |
| `chunk.overlap` | `50` | 块重叠 |
| `retrieval.top_k` | `5` | 检索数量 |
| `reranker.enabled` | `false` | 是否启用 Rerank |

---

##  评测结果

使用 RAGAS 框架对 16 个测试问题进行评估（基于真实扫地机器人知识库）：

```
  Faithfulness（忠实度）:       0.88  ✅
  Context Precision（检索精度）:  0.90  ✅
  Context Recall（检索召回）:    0.81  ✅
```

| 指标 | 分数 | 含义 |
|------|:----:|------|
| Faithfulness | **0.88** | 回答忠于文档内容，不凭空编造 |
| Context Precision | **0.90** | 检索到的文档高度相关 |
| Context Recall | **0.81** | 所需文档大部分被检索到 |

> 评测说明：使用 RAGAS 开源框架，通过 DeepSeek 作为评判 LLM，逐题计算 3 项指标后取均值。

---

##  项目结构

```
├── app/                          # 核心代码
│   ├── main.py                   # FastAPI 入口
│   ├── api/routes.py             # RESTful 路由
│   ├── core/
│   │   ├── config.py             # YAML 配置加载
│   │   └── logger.py             # 结构化日志
│   ├── models/
│   │   └── llm_client.py         # LLM 客户端（重试+流式）
│   └── rag/
│       ├── vector_store.py       # Chroma 向量库
│       ├── knowledge_base.py     # 知识库管理（多格式解析+去重）
│       ├── rag_service.py        # RAG 检索增强服务
│       └── reranker.py           # CrossEncoder 重排序
├── frontend/
│   ├── app_chat.py               # Streamlit 聊天界面
│   └── app_upload.py             # Streamlit 上传界面
├── config/
│   └── rag.yaml                  # 系统配置
├── tests/
│   ├── evaluate.py               # RAGAS 评测脚本
│   └── test_questions.json       # 测试问题集
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

##  开发计划

- [ ] 启用 Rerank 重排序（需 HuggingFace 网络通畅）
- [ ] 接入多模态 LLM 解析图片（GPT-4o Vision / Qwen-VL）
- [ ] 增加更多文件格式支持
- [ ] CI/CD 流水线

---

<div align="center">

**Made with   by [ycl1092](https://github.com/ycl1092)**

</div>
