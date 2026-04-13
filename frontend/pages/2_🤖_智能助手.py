import streamlit as st
import requests
import json

st.set_page_config(page_title="智能洞察助手", page_icon="🤖")

st.title("🤖 智能洞察助手")
st.markdown("我是您的百万客群经营助手，您可以问我关于客户数据、AUM预测、流失预警等任何问题。")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 接收用户输入
if prompt := st.chat_input("例如：帮我预测一下未来三个月的全行AUM增长趋势"):
    # 1. 将用户输入显示在界面上
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. 将用户输入加入历史记录
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. 准备调用 FastAPI 后端
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # 向 FastAPI 发送流式请求
            # 注意：这里的 messages 需要符合后端定义的 List[Dict[str, str]] 格式
            response = requests.post(
                "http://127.0.0.1:8000/api/v1/chat",
                json={"messages": st.session_state.messages},
                stream=True
            )
            
            if response.status_code == 200:
                # 解析 Server-Sent Events 流
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith("data: "):
                            data_str = line_text[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data_json = json.loads(data_str)
                                full_response = data_json.get("text", "")
                                # 实时更新 UI 占位符
                                message_placeholder.markdown(full_response + "▌")
                            except json.JSONDecodeError:
                                pass
                # 最终显示（去掉光标）
                message_placeholder.markdown(full_response)
            else:
                st.error(f"后端接口报错: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.error("无法连接到后端服务，请确认 FastAPI (127.0.0.1:8000) 已启动。")
            
    # 4. 将助手的回复加入历史记录
    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})
