import streamlit as st

st.set_page_config(
    page_title="百万客群经营助手",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏦 百万客群智能经营系统")
st.markdown("---")
st.markdown("""
### 欢迎使用！
请在左侧边栏选择您需要的功能模块：
* **📊 客群大屏**：查看全行宏观资产趋势与分布（支持 FastAPI 实时数据）。
* **🤖 智能助手**：对话式 BI，自然语言查询与多模型（LightGBM/ARIMA等）推理。
""")
