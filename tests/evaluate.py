"""RAGAS 评测"""
import json, os, time, warnings
from pathlib import Path
import sys
warnings.filterwarnings("ignore", message="LangchainLLMWrapper is deprecated")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.core.config import settings
from app.rag.vector_store import vector_store
from app.rag.rag_service import rag

# 从 .env 直接读 API Key（settings 读的是 LLM_API_KEY，但 .env 里写的是 OPENAI_API_KEY）
_api_key = os.getenv("OPENAI_API_KEY") or settings.LLM_API_KEY
_base_url = os.getenv("OPENAI_BASE_URL") or settings.LLM_BASE_URL or "https://api.deepseek.com/v1"
os.environ["OPENAI_API_KEY"] = _api_key
os.environ["OPENAI_BASE_URL"] = _base_url

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

eval_llm = LangchainLLMWrapper(ChatOpenAI(model="deepseek-v4-flash", temperature=0))

questions_data = json.load(open(Path(__file__).parent / "test_questions.json", encoding="utf-8"))
data = {"user_input": [], "response": [], "retrieved_contexts": [], "ground_truth": []}

for i, q in enumerate(questions_data):
    print(f"\n[{i+1}/{len(questions_data)}] {q['question'][:40]}...")
    docs = vector_store.similarity_search_with_score(q["question"], k=rag.top_k)
    data["retrieved_contexts"].append([d.page_content for d, _ in docs])
    ctx = "\n\n---\n\n".join(f"[来源: {d.metadata.get('source','?')}]\n{d.page_content}" for d, _ in docs)
    answer = rag.llm.chat(rag._build_messages(q["question"], ctx or "（无）", []))
    data["user_input"].append(q["question"])
    data["response"].append(answer)
    data["ground_truth"].append(q.get("ground_truth", ""))
    print(f"  回答: {answer[:200]}...  |  上下文: {len(data['retrieved_contexts'][-1])} 个")

print("\n计算 RAGAS 评分...")
dataset = Dataset.from_dict(data)
metrics = [faithfulness, context_precision, context_recall]
result = evaluate(dataset, metrics=metrics, llm=eval_llm)

print(f"\n{'='*50}")
scores_by_col = {}
for col in ["faithfulness", "context_precision", "context_recall"]:
    raw = result[col]
    if raw and not isinstance(raw[0], BaseException):
        avg = sum(raw) / len(raw)
        scores_by_col[col] = (avg, raw)
        print(f"  {col}: {avg:.4f}")

print(f"\n{'─'*50}\n逐题:")
for i, q in enumerate(questions_data):
    line = f"  [{i+1}] {q['question'][:30]:30s}"
    for col in scores_by_col:
        line += f" | {col}={scores_by_col[col][1][i]:.3f}"
    print(line)

out = Path(__file__).parent / "results"
out.mkdir(parents=True, exist_ok=True)
ts = time.strftime("%Y%m%d_%H%M%S")
rpt = {"timestamp": ts, "reranker_enabled": rag.reranker.enabled,
       "results": {col: {"avg": v[0], "per_question": v[1]} for col, v in scores_by_col.items()}}
json.dump(rpt, open(out / f"ragas_report_{ts}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"评测完成")
