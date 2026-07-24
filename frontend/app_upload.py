"""
Streamlit 知识库上传界面
"""

import streamlit as st
import requests

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(page_title="知识库管理", page_icon="")
st.title("知识库管理")

tab1, tab2 = st.tabs(["上传文件", "导入目录"])

# ── Tab 1: 上传文件 ──
with tab1:
    uploaded = st.file_uploader("选择文件", type=["txt", "md", "pdf", "docx", "json", "png", "jpg", "jpeg"])

    if uploaded is not None:
        st.write(f"文件名: {uploaded.name}")
        st.write(f"大小: {len(uploaded.getvalue()) / 1024:.1f} KB")

        if st.button("上传到知识库"):
            with st.spinner("正在处理..."):
                files = {"file": (uploaded.name, uploaded.getvalue(), "text/plain")}
                try:
                    resp = requests.post(f"{API_BASE}/upload/file", files=files, timeout=60)
                    result = resp.json()
                    if result["status"] == "success":
                        st.success(result["detail"])
                    else:
                        st.info(result["detail"])
                except Exception as e:
                    st.error(f"上传失败: {e}")

# ── Tab 2: 批量导入 ──
with tab2:
    st.info("批量导入 data/raw/ 目录下的所有 TXT 文件")
    if st.button("开始导入"):
        with st.spinner("正在批量导入..."):
            try:
                resp = requests.post(f"{API_BASE}/upload/ingest", timeout=300)
                st.success(f"导入完成: {resp.json()['detail']}")
            except Exception as e:
                st.error(f"导入失败: {e}")

# ── 状态栏 ──
st.divider()
col1, col2 = st.columns(2)
with col1:
    try:
        resp = requests.get(f"{API_BASE}/status", timeout=3)
        info = resp.json()
        st.metric("知识库文档数", info["doc_count"])
    except Exception:
        st.warning("后端未连接")
with col2:
    st.metric("服务状态", "运行中", delta=None,
              delta_color="off" if False else "normal")
