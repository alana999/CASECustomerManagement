import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="客群大屏", page_icon="📊", layout="wide")

st.title("📊 百万客群宏观监控大屏")
st.markdown("数据实时同步自后端 FastAPI 数据网关。")

API_BASE = "http://127.0.0.1:8000/api/v1/dashboard"

def fetch_data(endpoint):
    try:
        res = requests.get(f"{API_BASE}/{endpoint}", timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        st.error(f"无法连接后端服务获取数据 ({endpoint}): {e}")
    return None

# 1. 顶部核心指标卡片 (KPI)
kpi_data = fetch_data("stats/kpi")
if kpi_data:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="全行总客户数 (人)", value=f"{kpi_data['total_customers']:,}")
    with col2:
        # AUM 以亿元为单位展示
        aum_yi = kpi_data['total_aum'] / 100000000
        st.metric(label="全行总 AUM (亿元)", value=f"{aum_yi:,.2f} 亿")
    with col3:
        st.metric(label="高净值客户数 (AUM>=100万)", value=f"{kpi_data['hwni_count']:,}", delta="核心发力点")

st.markdown("---")

# 2. 中间图表区：资产分布与城市分布
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("💰 资产结构分布")
    asset_data = fetch_data("stats/asset_distribution")
    if asset_data:
        df_asset = pd.DataFrame(asset_data)
        fig_asset = px.pie(
            df_asset, 
            values='value', 
            names='name', 
            hole=0.4, # 环形图
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig_asset, use_container_width=True)

with col_right:
    st.subheader("🏙️ 客户城市分布")
    city_data = fetch_data("stats/city_distribution")
    if city_data:
        df_city = pd.DataFrame(city_data)
        fig_city = px.bar(
            df_city, 
            x='city_level', 
            y='count',
            text='count',
            color='city_level',
            labels={'city_level': '城市等级', 'count': '客户数量'}
        )
        st.plotly_chart(fig_city, use_container_width=True)

st.markdown("---")
st.info("💡 **提示**：如需深入分析特定客群或预测未来趋势，请前往左侧边栏的【🤖 智能助手】页面，使用多 Agent 分析工具。")