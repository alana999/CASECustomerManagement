import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.responses import StreamingResponse

# 引入我们封装好的 bot
from backend.agent.bot_engine import bot

router = APIRouter()

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    处理大模型对话请求，返回流式响应 (Server-Sent Events 风格)
    """
    try:
        # 这里为了演示流式效果，使用一个生成器
        def generate():
            # bot.run 会 yield 完整的 messages 列表，最后一条是最新回复
            for responses in bot.run(request.messages):
                if responses:
                    latest_response = responses[-1]['content']
                    # 按照 Server-Sent Events 格式发送
                    yield f"data: {json.dumps({'text': latest_response}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
