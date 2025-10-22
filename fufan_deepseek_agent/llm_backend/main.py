# 导入 FastAPI 框架和常用组件
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict

# 导入自定义服务层
from app.services.llm_factory import LLMFactory        # 用于统一创建不同类型的大语言模型（LLM）服务
from app.services.search_service import SearchService  # 搜索服务
from app.services.rag_service import RAGService        # RAG 文档处理服务（RAG = Retrieval-Augmented Generation）
from app.services.rag_chat_service import RAGChatService  # 文档问答服务（RAG Chat）

# 其他依赖
from fastapi.staticfiles import StaticFiles
from datetime import datetime
from pathlib import Path

# 日志与配置
from app.core.logger import get_logger, log_structured  # 日志模块
from app.core.middleware import LoggingMiddleware        # 自定义日志中间件
from app.core.config import settings                    # 全局配置
from app.api import api_router                          # 用户注册、登录、管理等子路由

# ======================
# 基础配置
# ======================

# 上传文件目录（用于 RAG 功能）
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)  # 若目录不存在则创建

# 初始化日志记录器（指定服务名为 main）
logger = get_logger(service="main")

# 创建 FastAPI 应用实例
app = FastAPI(title="AssistGen REST API")

# 添加日志中间件（替代 FastAPI 默认控制台日志）
app.add_middleware(LoggingMiddleware)

# 配置跨域中间件（允许前端访问 API）
# allow_origins=["*"] 表示允许任意域名访问，生产环境中应改为固定前端域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 将 api_router（用户管理、鉴权等接口）挂载到 /api 路径下
app.include_router(api_router, prefix="/api")

# ======================
# 数据模型定义（请求体格式）
# ======================

class ReasonRequest(BaseModel):
    """推理接口请求体"""
    messages: List[Dict[str, str]]

class ChatMessage(BaseModel):
    """普通聊天请求体"""
    messages: List[Dict[str, str]]

class RAGChatRequest(BaseModel):
    """基于文档的聊天请求体"""
    messages: List[Dict[str, str]]
    index_id: str  # 文档索引ID，用于检索RAG生成答案


# ======================
# 普通聊天接口
# ======================
@app.post("/chat")
async def chat_endpoint(request: ChatMessage):
    """聊天接口：用于调用普通 LLM 聊天"""
    try:
        logger.info("Processing chat request")  # 写入日志
        chat_service = LLMFactory.create_chat_service()  # 动态生成聊天服务实例
        
        # 结构化日志输出（记录消息数量和最后一句内容）
        log_structured("chat_request", {
            "message_count": len(request.messages),
            "last_message": request.messages[-1]["content"][:100] + "..."
        })
        
        # 返回流式响应（StreamingResponse）
        # 这样可以让客户端实时接收到模型的生成输出，而不是一次性返回完整回答
        return StreamingResponse(
            chat_service.generate_stream(request.messages),
            media_type="text/event-stream"  # SSE（Server-Sent Events）
        )
    
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ======================
# 推理接口
# ======================
@app.post("/reason")
async def reason_endpoint(request: ReasonRequest):
    """推理接口：用于逻辑推理、代码解释等任务"""
    try:
        logger.info("Processing reasoning request")
        reasoner = LLMFactory.create_reasoner_service()  # 创建推理模型服务实例
        
        log_structured("reason_request", {
            "message_count": len(request.messages),
            "last_message": request.messages[-1]["content"][:100] + "..."
        })
        
        return StreamingResponse(
            reasoner.generate_stream(request.messages),
            media_type="text/event-stream"
        )
    
    except Exception as e:
        logger.error(f"Reasoning error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ======================
# 搜索增强聊天接口
# ======================
@app.post("/search")
async def search_endpoint(request: ChatMessage):
    """带搜索功能的聊天接口：模型在回答前先进行知识检索"""
    try:
        search_service = SearchService()
        return StreamingResponse(
            search_service.generate_stream(request.messages[0]["content"]),
            media_type="text/event-stream"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ======================
# 文件上传接口（RAG准备）
# ======================
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文件接口：将文件保存到服务器并进行 RAG 预处理"""
    try:
        logger.info(f"Uploading file: {file.filename}")
        
        # 为文件生成带时间戳的唯一文件名，防止覆盖
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = UPLOAD_DIR / filename
        
        # 确保上传目录存在
        UPLOAD_DIR.mkdir(exist_ok=True)
        
        # 异步读取上传文件内容
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
            
        # 获取文件元信息
        file_type = file.content_type
        file_ext = Path(file.filename).suffix.lower()
        
        file_info = {
            "filename": filename,
            "original_name": file.filename,
            "size": len(content),
            "type": file_type,
            "path": str(file_path).replace('\\', '/'),
        }
        
        print(f"文件已保存到: {file_path}")
        
        # 初始化 RAG 服务并处理上传文件
        rag_service = RAGService()
        rag_result = await rag_service.process_file(file_info)
        
        # 合并文件信息与处理结果
        result = {**file_info, **rag_result}
        
        # 结构化日志
        log_structured("file_upload", {
            "filename": file.filename,
            "size": len(content),
            "type": file_type
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        return {"error": str(e)}
    
    return f"data: {result}\n\n"


# ======================
# 基于文档的聊天（RAG Chat）
# ======================
@app.post("/chat-rag")
async def rag_chat_endpoint(request: RAGChatRequest):
    """RAG聊天接口：在上传文档的基础上进行上下文问答"""
    try:
        rag_chat_service = RAGChatService()
        
        return StreamingResponse(
            rag_chat_service.generate_stream(
                request.messages,
                request.index_id  # 指定文档索引
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ======================
# 健康检查接口
# ======================
@app.get("/health")
async def health_check():
    """健康检查接口，用于检测服务是否正常运行"""
    return {"status": "ok"}


# ======================
# 静态文件挂载（前端页面）
# ======================
# 用于挂载前端静态页面（例如 Vue/React 构建后的 dist 目录）
STATIC_DIR = Path(__file__).parent / "static" / "dist"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
