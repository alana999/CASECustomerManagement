import sys
import os
# 将项目根目录添加到 sys.path 中，解决 ModuleNotFoundError: No module named 'backend'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# 导入路由
from backend.routers import chat, dashboard

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="百万客群经营助手 API",
    description="基于 FastAPI 和 Qwen-Agent 的后端服务",
    version="1.0.0"
)

# 配置 CORS（允许前端 Streamlit 跨域请求）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应配置为具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["Dashboard"])

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "message": "Backend is running!"}

if __name__ == "__main__":
    # 启动命令: python -m backend.main
    logger.info("Starting FastAPI server...")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
