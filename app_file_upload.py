"""
基于Streamlit完成网页上传服务
"""
import time
import streamlit as st
from knowledge_base import KnowledgeBaseService

st.title("知识库更新服务")

upload_file=st.file_uploader(
    "请上传所要更新知识的txt文件",
    type=["txt"],
    accept_multiple_files=False
)

if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

if upload_file is not None:
    name=upload_file.name
    type=upload_file.type
    size=upload_file.size/1024

    st.subheader(f"文件名：{name}")
    st.write(f"文件类型：{type} | 文件大小：{size:.2f}kb")

    text =upload_file.getvalue().decode("utf-8")

    with st.spinner("载入知识库ing....."):
        time.sleep(2)
        result = st.session_state["service"].upload_by_str(text,name)
        st.write(result)