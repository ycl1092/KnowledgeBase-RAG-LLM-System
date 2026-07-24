"""
Streamlit 聊天界面

连接后端 API 实现 RAG 问答。
"""

import streamlit as st
import requests

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(page_title="RAG 知识库问答", page_icon="")
st.title("知识库智能问答")
st.caption(f"后端: {API_BASE}")

# ── 会话状态 ──
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())[:8]

# ── 侧边栏 ──
with st.sidebar:
    st.header(f"会话: {st.session_state.session_id}")
    if st.button("清除历史"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("知识库状态")
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=3)
        info = resp.json()
        st.metric("文档数", info["doc_count"])
        st.metric("LLM 状态", "" if info["llm_ready"] else "未配置")
    except Exception:
        st.error("后端未连接")

# ── 历史消息 ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 输入 ──
if prompt := st.chat_input("请输入问题..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/chat",
                    json={
                        "question": prompt,
                        "history": st.session_state.messages[:-1],
                    },
                    timeout=60,
                )
                answer = resp.json()["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"请求失败: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"（错误）{e}"})
        st.rerun()
