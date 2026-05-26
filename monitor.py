# -*- coding: utf-8 -*-
import sys, os
if getattr(sys, 'frozen', False):
    # 打包环境：数据文件从 exe 旁边的 _internal/ 读（与 tts_server 共用）
    _exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    _shared_internal = os.path.join(_exe_dir, '_internal')
    if os.path.isdir(_shared_internal):
        _PROJECT_ROOT = _shared_internal
    elif hasattr(sys, '_MEIPASS'):
        _PROJECT_ROOT = sys._MEIPASS
    else:
        _PROJECT_ROOT = _exe_dir
else:
    _PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)  # 确保 CWD 指向项目根，find_models() 等使用相对路径 'static'
import mimetypes
mimetypes.add_type("application/javascript", ".js")
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from config import MONITOR_SERVER_PORT, MAIN_SERVER_PORT
from utils.config_manager import get_config_manager, get_reserved
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import uvicorn
from fastapi.templating import Jinja2Templates
from utils.frontend_utils import find_models, find_model_config_file, find_model_directory
from utils.live2d_resources import (
    compose_expression_refs,
    derive_emotion_mapping,
    get_overlay_entry,
    load_vtube_expression_refs,
    read_model_config,
    resolve_live2d_context,
    scan_live2d_assets,
)
from utils.workshop_utils import get_default_workshop_folder
from utils.preferences import load_user_preferences

# Setup logger
from utils.logger_config import setup_logging
logger, log_config = setup_logging(service_name="Monitor", log_level=logging.INFO)

# 获取资源路径（支持打包后的环境）
def get_resource_path(relative_path):
    """获取资源的绝对路径 — 始终基于 _PROJECT_ROOT（共用 _internal 或项目根）"""
    return os.path.join(_PROJECT_ROOT, relative_path)

templates = Jinja2Templates(directory=get_resource_path(""))

# 存储所有连接的客户端
connected_clients = set()
subtitle_clients = set()
current_subtitle = ""
should_clear_next = False


async def cleanup_disconnected_clients():
    """定期清理断开的连接"""
    while True:
        try:
            for client in list(connected_clients):
                try:
                    await client.send_json({"type": "heartbeat"})
                except Exception as e:
                    logger.warning(f"心跳检测失败，移除客户端: {e}")
                    connected_clients.discard(client)
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"清理客户端错误: {e}")
            await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: 启动/关闭时的资源管理"""
    task = asyncio.create_task(cleanup_disconnected_clients())
    logger.info(f"Monitor 服务已启动，端口: {MONITOR_SERVER_PORT}")
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)

# 挂载静态文件
app.mount("/static", StaticFiles(directory=get_resource_path("static")), name="static")
_config_manager = get_config_manager()

# 挂载用户Live2D目录（与main_server.py保持一致，CFA感知）
_readable_live2d = _config_manager.readable_live2d_dir
_serve_live2d_path = str(_readable_live2d) if _readable_live2d else str(_config_manager.live2d_dir)
if os.path.exists(_serve_live2d_path):
    app.mount("/user_live2d", StaticFiles(directory=_serve_live2d_path), name="user_live2d")
    logger.info(f"已挂载用户Live2D目录: {_serve_live2d_path}")
# CFA 场景：可写回退目录额外挂载
if _readable_live2d and str(_config_manager.live2d_dir) != _serve_live2d_path:
    _writable_live2d_path = str(_config_manager.live2d_dir)
    if os.path.exists(_writable_live2d_path):
        app.mount("/user_live2d_local", StaticFiles(directory=_writable_live2d_path), name="user_live2d_local")
        logger.info(f"已挂载本地Live2D目录(CFA回退): {_writable_live2d_path}")

# 挂载创意工坊目录（与main_server.py保持一致）
workshop_path = get_default_workshop_folder()
if workshop_path and os.path.exists(workshop_path):
    app.mount("/workshop", StaticFiles(directory=workshop_path), name="workshop")
    logger.info(f"已挂载创意工坊目录: {workshop_path}")

@app.get("/")
async def root_redirect():
    """根路径：重定向到当前角色的 viewer 页面"""
    try:
        _, her_name, _, _, _, _, _, _, _ = _config_manager.get_character_data()
    except Exception:
        her_name = ""
    if her_name:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=f"/{her_name}")
    return HTMLResponse("<h3>No character configured</h3>", status_code=404)

@app.get("/subtitle")
async def get_subtitle():
    return FileResponse(get_resource_path('templates/subtitle.html'))

@app.get("/api/config/page_config")
async def get_page_config(lanlan_name: str = ""):
    """获取页面配置（lanlan_name 和 model_path）"""
    try:
        _, her_name, _, lanlan_basic_config, _, _, _, _, _ = _config_manager.get_character_data()
        target_name = lanlan_name if lanlan_name else her_name

        # 大小写不敏感匹配角色名（URL 可能使用小写）
        char_data = lanlan_basic_config.get(target_name, {})
        if not char_data and target_name:
            target_lower = target_name.lower()
            for key in lanlan_basic_config:
                if key.lower() == target_lower:
                    target_name = key  # 使用 config 里的真实名称
                    char_data = lanlan_basic_config[key]
                    break

        live2d_model_path = get_reserved(
            char_data,
            'avatar',
            'live2d',
            'model_path',
            default='mao_pro',
            legacy_keys=('live2d',),
        )
        if not isinstance(live2d_model_path, str):
            live2d_model_path = str(live2d_model_path) if live2d_model_path is not None else 'mao_pro'
        if live2d_model_path.endswith('.model3.json'):
            parts = live2d_model_path.replace('\\', '/').split('/')
            live2d = parts[-2] if len(parts) >= 2 else parts[-1].removesuffix('.model3.json')
        else:
            live2d = live2d_model_path

        models = find_models()
        model_path = next((m["path"] for m in models if m["name"] == live2d), find_model_config_file(live2d))

        return {
            "success": True,
            "lanlan_name": target_name,
            "model_path": model_path
        }
    except Exception as e:
        logger.error(f"获取页面配置失败: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/config/preferences")
async def get_preferences():
    """获取用户偏好设置（与main_server.py保持一致）"""
    preferences = load_user_preferences()
    return preferences

@app.post("/api/config/preferences")
async def save_preferences_readonly():
    """Monitor 为只读模式，不允许保存偏好设置"""
    return JSONResponse(status_code=200, content={"success": True, "readonly": True})

@app.post("/api/emotion/analysis")
async def emotion_analysis_proxy(request: Request):
    """代理情绪分析请求到主服务器"""
    import aiohttp
    try:
        data = await request.json()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://127.0.0.1:{MAIN_SERVER_PORT}/api/emotion/analysis",
                json=data,
                timeout=aiohttp.ClientTimeout(total=10.0)
            ) as resp:
                result = await resp.json()
                return JSONResponse(status_code=resp.status, content=result)
    except aiohttp.ClientError as e:
        logger.warning(f"情绪分析代理失败（主服务器可能未运行）: {e}")
        return JSONResponse(status_code=502, content={"error": "主服务器不可达"})
    except Exception as e:
        logger.error(f"情绪分析代理错误: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get('/api/live2d/emotion_mapping/{model_name}')
def get_emotion_mapping(model_name: str, item_id: str = ""):
    """获取情绪映射配置"""
    try:
        context = resolve_live2d_context(model_name=model_name, item_id=item_id or None)
        entry, source = get_overlay_entry(_config_manager, context)
        if entry and isinstance(entry.get("mapping"), dict):
            emotion_mapping = entry["mapping"]
        else:
            config_data = read_model_config(context.model_config_path)
            emotion_mapping = config_data.get("EmotionMapping")
            source = "model" if emotion_mapping else "derived"
            if not emotion_mapping:
                emotion_mapping = derive_emotion_mapping(config_data)
        return {
            "success": True,
            "config": emotion_mapping,
            "source": source,
            "model_identity": context.model_identity,
        }
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"success": False, "error": str(e)})
    except Exception as e:
        logger.error(f"获取情绪映射配置失败: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.get('/api/live2d/model_files/{model_name}')
def get_model_files(model_name: str, item_id: str = ""):
    """获取指定 Live2D 模型的动作和表情文件列表"""
    try:
        context = resolve_live2d_context(model_name=model_name, item_id=item_id or None)
        config_data = read_model_config(context.model_config_path)
        assets = scan_live2d_assets(context.actual_model_dir)
        vtube_refs = load_vtube_expression_refs(context.actual_model_dir)
        return {
            "success": True,
            "motion_files": assets["motion_files"],
            "expression_files": assets["expression_files"],
            "expression_refs": compose_expression_refs(
                config_data,
                assets["expression_files"],
                vtube_refs,
            ),
            "model_identity": context.model_identity,
            "model_config_url": context.model_config_url,
        }
    except FileNotFoundError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"获取模型文件列表失败: {e}")
        return {"success": False, "error": str(e)}

@app.get('/api/live2d/load_model_parameters/{model_name}')
def load_model_parameters(model_name: str):
    """从模型目录的 parameters.json 文件加载参数"""
    try:
        model_dir, _ = find_model_directory(model_name)
        if not model_dir or not os.path.exists(model_dir):
            return {"success": False, "error": f"模型 {model_name} 不存在"}

        parameters_file = os.path.join(model_dir, 'parameters.json')
        if not os.path.exists(parameters_file):
            return {"success": True, "parameters": {}}

        with open(parameters_file, 'r', encoding='utf-8') as f:
            parameters = json.load(f)

        if not isinstance(parameters, dict):
            return {"success": True, "parameters": {}}

        return {"success": True, "parameters": parameters}
    except Exception as e:
        logger.error(f"加载模型参数失败: {e}")
        return {"success": False, "error": str(e), "parameters": {}}


@app.get("/{lanlan_name}", response_class=HTMLResponse)
async def get_index(request: Request, lanlan_name: str):
    return templates.TemplateResponse("templates/viewer.html", {
        "request": request
    })


@app.websocket("/subtitle_ws")
async def subtitle_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"字幕客户端已连接: {websocket.client}")

    subtitle_clients.add(websocket)

    try:
        if current_subtitle:
            await websocket.send_json({
                "type": "subtitle",
                "text": current_subtitle
            })
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info(f"字幕客户端已断开: {websocket.client}")
    finally:
        subtitle_clients.discard(websocket)


async def broadcast_subtitle():
    """广播字幕到所有字幕客户端"""
    global current_subtitle, should_clear_next
    if should_clear_next:
        await clear_subtitle()
        should_clear_next = False
        await asyncio.sleep(0.3)

    clients = subtitle_clients.copy()
    for client in clients:
        try:
            await client.send_json({
                "type": "subtitle",
                "text": current_subtitle
            })
        except Exception as e:
            logger.warning(f"字幕广播错误: {e}")
            subtitle_clients.discard(client)


async def clear_subtitle():
    """清空字幕"""
    global current_subtitle
    current_subtitle = ""

    clients = subtitle_clients.copy()
    for client in clients:
        try:
            await client.send_json({
                "type": "clear"
            })
        except Exception as e:
            logger.warning(f"清空字幕错误: {e}")
            subtitle_clients.discard(client)


# 主服务器连接端点
@app.websocket("/sync/{lanlan_name}")
async def sync_endpoint(websocket: WebSocket, lanlan_name: str):
    await websocket.accept()
    logger.info(f"[SYNC] 主服务器已连接: {websocket.client}")

    try:
        while True:
            try:
                global current_subtitle
                data = await asyncio.wait_for(websocket.receive_text(), timeout=25)

                data = json.loads(data)
                msg_type = data.get("type", "unknown")

                if msg_type == "gemini_response":
                    subtitle_text = data.get("text", "")
                    current_subtitle += subtitle_text
                    if subtitle_text:
                        await broadcast_subtitle()

                elif msg_type == "turn end":
                    # 回合结束，准备清空字幕（下一条消息到来时清空）
                    global should_clear_next
                    should_clear_next = True

                if msg_type != "heartbeat":
                    await broadcast_message(data)
            except asyncio.exceptions.TimeoutError:
                pass
    except WebSocketDisconnect:
        logger.info(f"[SYNC] 主服务器已断开: {websocket.client}")
    except Exception as e:
        logger.error(f"[SYNC] 同步端点错误: {e}")


# 二进制数据同步端点
@app.websocket("/sync_binary/{lanlan_name}")
async def sync_binary_endpoint(websocket: WebSocket, lanlan_name: str):
    await websocket.accept()
    logger.info(f"[BINARY] 主服务器二进制连接已建立: {websocket.client}")

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=25)
                if len(data) > 4:
                    await broadcast_binary(data)
            except asyncio.exceptions.TimeoutError:
                pass
    except WebSocketDisconnect:
        logger.info(f"[BINARY] 主服务器二进制连接已断开: {websocket.client}")
    except Exception as e:
        logger.error(f"[BINARY] 二进制同步端点错误: {e}")


# 客户端连接端点
@app.websocket("/ws/{lanlan_name}")
async def websocket_endpoint(websocket: WebSocket, lanlan_name: str):
    await websocket.accept()
    logger.info(f"[CLIENT] 查看客户端已连接: {websocket.client}, 当前总数: {len(connected_clients) + 1}")

    connected_clients.add(websocket)

    try:
        while True:
            try:
                await websocket.receive_text()
            except Exception:
                try:
                    await websocket.receive_bytes()
                except Exception:
                    await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        logger.info(f"[CLIENT] 查看客户端已断开: {websocket.client}")
    except Exception as e:
        logger.warning(f"[CLIENT] 客户端连接异常: {e}")
    finally:
        connected_clients.discard(websocket)
        logger.info(f"[CLIENT] 已移除客户端，当前剩余: {len(connected_clients)}")


async def broadcast_message(message):
    """广播消息到所有客户端"""
    clients = connected_clients.copy()
    disconnected_clients = []

    for client in clients:
        try:
            await client.send_json(message)
        except Exception as e:
            logger.warning(f"[BROADCAST] 广播错误到 {client.client}: {e}")
            disconnected_clients.append(client)

    for client in disconnected_clients:
        connected_clients.discard(client)


async def broadcast_binary(data):
    """广播二进制数据到所有客户端"""
    clients = connected_clients.copy()
    disconnected_clients = []

    for client in clients:
        try:
            await client.send_bytes(data)
        except Exception as e:
            logger.warning(f"[BINARY BROADCAST] 二进制广播错误到 {client.client}: {e}")
            disconnected_clients.append(client)

    for client in disconnected_clients:
        connected_clients.discard(client)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=MONITOR_SERVER_PORT, reload=False)
