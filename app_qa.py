
from rag import RagService
import streamlit as st
import config_data as config

st.title("攻略助手")
st.divider()

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "欢迎来到明日方舟攻略助手，请输入问题"}]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input("请输入问题")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    ai_res_list = []
    with st.spinner("正在思考ing..."):
        response_stream = st.session_state["rag"].chain.stream({"input": prompt},config.session_config)

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        st.chat_message("assistant").write_stream(capture(response_stream,ai_res_list))
        st.session_state["messages"].append({"role": "assistant", "content": "".join(ai_res_list)})
