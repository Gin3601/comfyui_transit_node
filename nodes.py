# -*- coding: utf-8 -*-
"""
CYAI 中转站 —— doubao-seedance 视频生成节点（ComfyUI 自定义节点）

接口契约（火山方舟 Ark 原生格式，经 CYAI 中转站 https://www.cyai.club 透传）：
  - 创建任务：POST {base_url}{submit_url}
  - 查询任务：GET  {base_url}{poll_url_template}，其中 {task_id} 会被替换
  - 鉴权：Authorization: Bearer sk-xxx

请求体（content 数组 + 顶层参数）：
  {
    "model": "doubao-seedance-2-0-260128",
    "content": [
      {"type": "text", "text": "提示词"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}, "role": "first_frame"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}, "role": "reference_image"}
    ],
    "resolution": "720p", "ratio": "16:9", "duration": 5,
    "generate_audio": true, "watermark": false, "seed": -1, "return_last_frame": false
  }

响应（顶层字段，无 data 嵌套）：
  创建 -> {"id": "cgt-xxx", ...}
  查询 -> {"id": "...", "status": "succeeded"|"failed", "content": {"video_url": "..."},
            "usage": {"completion_tokens": N}, "error": {"code": "...", "message": "..."}}

要点：
  - 状态枚举：queued / running -> succeeded / failed（火山是过去式 succeeded，不是 SUCCESS）
  - 视频地址：content.video_url（TOS 临时签名 URL，24 小时有效，需及时下载）
  - 首帧/尾帧模式 ratio 必须为 adaptive；首尾帧与多图参考(reference_image)互斥
  - 多图参考：content 里多个 role=reference_image 项，建议 1-4 张
  - 输出是 ComfyUI VIDEO 对象（非字符串路径），可直接接预览/保存节点
"""

import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid

import numpy as np
import torch
from PIL import Image

# ComfyUI 进度与中断适配（节点只在 ComfyUI 内运行，这些模块必定存在）
from comfy.utils import ProgressBar
import comfy.model_management

RESOLUTION_OPTIONS = ["480p", "720p", "1080p"]
RATIO_OPTIONS = ["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"]

# CYAI / 火山方舟固定接口路径（可被节点上的 submit_url / poll_url_template 覆盖）
DEFAULT_SUBMIT_URL = "/api/v3/contents/generations/tasks"
DEFAULT_POLL_URL_TEMPLATE = "/api/v3/contents/generations/tasks/{task_id}"

DONE_STATUSES = {"SUCCEEDED"}
FAIL_STATUSES = {"FAILED", "CANCELLED", "CANCELED"}


def _image_to_data_uri(image: torch.Tensor, max_side: int = 1024, quality: int = 80,
                       min_side: int = 300) -> str:
    """把 ComfyUI 的 IMAGE 张量 [B,H,W,C] (0-1 float) 转成 base64 data URI。

    火山方舟要求单张图 <10MB、整个请求体 <64MB。参考图/首尾帧只需让模型看清
    主体与风格，无需原始分辨率，故默认限制最长边 1024、JPEG 质量 80；
    单张约 100-300KB，9 张约 1-3MB，远低于 64MB 上限。

    另：接口要求**宽和高都 ≥300px**（实测宽度不足报
    `expected the width to be at least 300px`），过小的图会被等比放大到 300px。
    """
    if image.dim() == 4:
        image = image[0]
    arr = (image.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    pil = Image.fromarray(arr)
    if pil.mode == "RGBA":
        pil = pil.convert("RGB")
    w, h = pil.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        pil = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    w, h = pil.size
    if min(w, h) < min_side:                      # 接口下限：宽高均需 ≥300px
        scale = min_side / float(min(w, h))
        pil = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _uniform_square(image: torch.Tensor, max_side: int) -> torch.Tensor:
    """把单张 [H,W,C] 图像缩放到 max_side×max_side 正方形（长边缩放、白底居中填充）。

    RGBA 先做白底合成再转 RGB，避免透明像素在 JPEG 里落成黑色。
    返回 [1, max_side, max_side, 3] 的 float32 tensor。
    """
    arr = (image.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    pil = Image.fromarray(arr)
    if pil.mode == "RGBA":
        bg = Image.new("RGB", pil.size, (255, 255, 255))
        bg.paste(pil, mask=pil.getchannel("A"))
        pil = bg
    elif pil.mode != "RGB":
        pil = pil.convert("RGB")
    w, h = pil.size
    scale = max_side / float(max(w, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    pil = pil.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (max_side, max_side), (255, 255, 255))
    canvas.paste(pil, ((max_side - nw) // 2, (max_side - nh) // 2))
    out = np.array(canvas).astype(np.float32) / 255.0
    return torch.from_numpy(out)[None, ...]


def _resolve_url(base: str, url_or_path: str) -> str:
    """绝对 URL 原样返回；相对路径拼到 base_url 后面。"""
    url_or_path = (url_or_path or "").strip()
    if url_or_path.startswith(("http://", "https://")):
        return url_or_path
    base = (base or "").rstrip("/")
    if not url_or_path:
        return base
    return f"{base}/{url_or_path.lstrip('/')}"


def _split_ids(value) -> list[str]:
    """把 ID 归一成列表。接受：逗号/空格分隔的字符串、或 ComfyUI 传来的 list。

    ComfyUI 把上游 STRING 输出接到 STRING 输入时可能只传第一个元素，
    也可能传整个 list，这里两种都兼容。
    """
    import re
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(_split_ids(v))
        return out
    return [t for t in re.split(r"[,，\s]+", str(value).strip()) if t]


def _friendly_privacy_error(text: str) -> str | None:
    """把真人脸隐私拦截错误翻译成人话 + 修复指引；非该错误返回 None。"""
    if "InputImageSensitiveContentDetected" in text or "may contain real person" in text:
        return (
            "输入图片被判定包含「真人」（真人脸隐私拦截，错误码 "
            "InputImageSensitiveContentDetected.PrivacyInformation）。\n"
            "base64 和公网 URL 两种传法都会被扫，**只有「素材引用」能绕开**。\n"
            "【最快检查】如果你在视频节点上开的 upload_images 没生效：\n"
            "  确认 upload_images 已打开、api_key 已填，且图是接在\n"
            "  reference_images / first_frame 上（接上后节点会自动转素材，无需再接 asset_ids）。\n"
            "【手动流程】若要自己建素材：\n"
            "  1. 用「CYAI 上传素材」把图（或公网 URL）变成素材，拿到 asset_id；\n"
            "  2. 在本节点 asset_ids 里填该 ID（多个用逗号分隔）；\n"
            "  3. 清空 first_frame / last_frame / reference_images。\n"
            "【真人素材】真人面容需走活体认证：创建认证会话 → 本人认证 → 查询认证结果拿 GroupId →\n"
            "  上传素材到该组 → 再用 asset_ids 引用。AI 生成的虚拟人像用 AIGC 组即可，无需认证。"
        )
    return None


def _http_json(method: str, url: str, headers: dict, payload: dict | None = None, timeout: float = 60.0):
    """发送 HTTP 请求并解析 JSON，出错时抛出带响应内容的异常。"""
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        body = e.read()
        text = body.decode("utf-8", "replace")
        hint = _friendly_privacy_error(text)
        raise RuntimeError(
            hint or ("API 请求失败 (HTTP %s)：%s" % (e.code, text[:1500]))
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError("无法连接到中转站 %s：%s" % (url, e.reason)) from e

    if not body:
        return {}
    text = body.decode("utf-8", "replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError("响应不是有效 JSON（HTTP 200 但内容异常）：%s" % text[:800]) from e


def _download(url: str, headers: dict, timeout: float = 300.0) -> bytes:
    """下载视频字节。视频 URL 是 TOS 临时签名链接，一般无需鉴权头。"""
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            "下载视频失败 (HTTP %s)：%s" % (e.code, e.read()[:500].decode("utf-8", "replace"))
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError("下载视频时无法连接 %s：%s" % (url, e.reason)) from e


def _video_from_bytes(video_bytes: bytes):
    """把 mp4 字节包装成 ComfyUI VIDEO 输出对象（延迟导入 comfy_api）。"""
    from comfy_api.latest import InputImpl
    return InputImpl.VideoFromFile(io.BytesIO(video_bytes))


def _image_from_bytes(img_bytes: bytes) -> torch.Tensor:
    """把图片字节（JPEG/PNG 等）解码成 ComfyUI IMAGE 张量 [1,H,W,C] (0-1 float)。"""
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def _jpeg_bytes_from_image(image: torch.Tensor, max_side: int = 2048,
                           quality: int = 90, min_side: int = 300) -> bytes:
    """把 IMAGE 张量转成 JPEG 字节。用于上传公网图床。

    素材接口要求宽度 300-6000px，这里保证：最长边 ≤ max_side，最短边 ≥ min_side。
    """
    if image.dim() == 4:
        image = image[0]
    arr = (image.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    pil = Image.fromarray(arr)
    if pil.mode == "RGBA":
        bg = Image.new("RGB", pil.size, (255, 255, 255))
        bg.paste(pil, mask=pil.getchannel("A"))
        pil = bg
    elif pil.mode != "RGB":
        pil = pil.convert("RGB")

    w, h = pil.size
    if max(w, h) > max_side:                      # 太大：等比缩到最长边
        s = max_side / float(max(w, h))
        pil = pil.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    w, h = pil.size
    if min(w, h) < min_side:                      # 太小：等比放大到最短边（素材接口 ≥300px）
        s = min_side / float(min(w, h))
        pil = pil.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
        if min(pil.size) > 6000:                  # 放大后越过上限则回退原图
            pil = Image.fromarray(arr).convert("RGB")

    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _upload_to_host(img_bytes: bytes, filename: str = "image.jpg",
                    provider: str = "uguu", timeout: float = 60.0) -> str:
    """把图片字节 POST 到公网图床，返回直链 URL。

    只用免费匿名图床（无需账号/key）。返回的链接多为临时链接（数小时），
    但素材接口会立即下载并转存到平台托管存储，因此足以完成「建素材」这一步。
    """
    boundary = "----CYAIUpload" + uuid.uuid4().hex
    body = b""
    if provider == "uguu":          # https://uguu.se  —— 匿名，返回 https://n.uguu.se/xxx.jpg
        body += ("--%s\r\n" % boundary).encode()
        body += ('Content-Disposition: form-data; name="files[]"; filename="%s"\r\n' % filename).encode()
        body += b"Content-Type: image/jpeg\r\n\r\n"
        body += img_bytes
        body += ("\r\n--%s--\r\n" % boundary).encode()
        url = "https://uguu.se/upload.php"
    elif provider == "catbox":      # https://catbox.moe —— 匿名，永久链接（但机房 IP 常被拒）
        body += ("--%s\r\n" % boundary).encode()
        body += b'Content-Disposition: form-data; name="reqtype"\r\n\r\nfileupload\r\n'
        body += ("--%s\r\n" % boundary).encode()
        body += ('Content-Disposition: form-data; name="fileToUpload"; filename="%s"\r\n' % filename).encode()
        body += b"Content-Type: image/jpeg\r\n\r\n"
        body += img_bytes
        body += ("\r\n--%s--\r\n" % boundary).encode()
        url = "https://catbox.moe/user/api.php"
    else:
        raise ValueError("不支持的图床：%s" % provider)

    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "multipart/form-data; boundary=%s" % boundary,
                 "User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", "replace").strip()
    except urllib.error.HTTPError as e:
        raise RuntimeError("图床上传失败 (HTTP %s)：%s" % (e.code, e.read()[:300].decode("utf-8", "replace"))) from e
    except urllib.error.URLError as e:
        raise RuntimeError("图床不可达（%s）：%s；可换 image_host 再试" % (provider, e.reason)) from e

    if provider == "uguu":
        try:
            data = json.loads(text)
            link = (data.get("files") or [{}])[0].get("url") or ""
        except json.JSONDecodeError:
            link = ""
    else:
        link = text if text.startswith("http") else ""
    if not link:
        raise RuntimeError("图床上传未返回 URL：%s" % text[:300])
    return link


# 火山方舟素材资产接口（Action 兼容入口）常量
ARK_VERSION = "2024-01-01"
ARK_SERVICE = "ark"


def _ark_action(base: str, api_key: str, action: str, payload: dict, timeout: float = 60.0):
    """调用火山方舟素材资产接口（Action 兼容入口），返回 Result 对象。

    统一 POST {base}/api/?Action=<action>&Version=2024-01-01，
    鉴权 Authorization: Bearer <token>，无需 HMAC 签名。
    响应为火山原生格式：
      成功：{"ResponseMetadata": {...}, "Result": {...}}
      失败：{"ResponseMetadata": {...}, "Error": {"Code": ..., "Message": ...}}
    业务失败抛出 RuntimeError；成功返回 Result（可能为空 dict）。
    """
    url = "%s/api/?Action=%s&Version=%s" % (base.rstrip("/"), action, ARK_VERSION)
    headers = {"Authorization": "Bearer %s" % api_key, "Content-Type": "application/json"}
    resp = _http_json("POST", url, headers, payload, timeout=timeout)
    if not isinstance(resp, dict):
        raise RuntimeError("%s 响应格式异常：%s" % (action, str(resp)[:800]))
    err = resp.get("Error") or resp.get("error")
    if isinstance(err, dict) and err:
        raise RuntimeError(
            "%s 失败：%s %s" % (action, err.get("Code") or err.get("code") or "",
                               err.get("Message") or err.get("message") or "")
        )
    result = resp.get("Result")
    return result if isinstance(result, dict) else {}


def _resolve_field(data, dot_path: str):
    """按点路径从 JSON 取值，支持整数索引访问列表（如 videos.0.url）。

    路径为空返回 None；遇到无法继续的类型返回 None。
    """
    if not dot_path:
        return None
    current = data
    for part in str(dot_path).strip().split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, (list, tuple)):
            try:
                current = current[int(part)]
            except (ValueError, IndexError, TypeError):
                return None
        else:
            return None
        if current is None:
            return None
    return current


def _resolve_first(data, paths: str):
    """从左到右尝试多个「|」分隔的候选路径，返回第一个非空值。"""
    for path in [p.strip() for p in str(paths).split("|") if p.strip()]:
        v = _resolve_field(data, path)
        if v is not None and v != "":
            return v
    return None


def _extract_task_id(created, task_id_field: str):
    """按可配路径从提交响应取任务 ID（支持多路径兜底）。"""
    return _resolve_first(created, task_id_field)


def _extract_status(state: dict, status_field: str) -> str:
    """按可配路径从查询结果取状态，大写归一。"""
    v = _resolve_first(state, status_field)
    return str(v or "").upper()


def _extract_error(state: dict, error_path: str = "error.message|error.code") -> str | None:
    """失败时按可配路径返回错误信息（message/code），否则 None。"""
    return _resolve_first(state, error_path)


def _extract_video_url(state: dict, video_url_field: str):
    """按可配路径从查询结果取 mp4 地址（支持多路径兜底）。"""
    return _resolve_first(state, video_url_field)

# ===== 协议配方（recipe）：把「提交/轮询/解析」的供应商差异抽成数据 =====

ARK_SEEDANCE_RECIPE = {
    "name": "ark_seedance",
    "display_name": "火山方舟 doubao-seedance（Ark v3）",
    "async": True,
    "auth": {
        "headers": {
            "Authorization": "Bearer {api_key}",
            "Content-Type": "application/json",
        },
    },
    "submit": {
        "method": "POST",
        "url": "{base_url}/api/v3/contents/generations/tasks",
        "body": (
            '{"model": {model}, "content": {content}, "resolution": {resolution}, '
            '"ratio": {ratio}, "duration": {duration}, "generate_audio": {generate_audio}, '
            '"watermark": {watermark}, "seed": {seed}, "return_last_frame": {return_last_frame}}'
        ),
    },
    "poll_url_template": "{base_url}/api/v3/contents/generations/tasks/{task_id}",
    "content_items": {
        "text": '{"type": "text", "text": {prompt}}',
        "first_frame": '{"type": "image_url", "image_url": {"url": {image}}, "role": "first_frame"}',
        "last_frame": '{"type": "image_url", "image_url": {"url": {image}}, "role": "last_frame"}',
        "reference_image": '{"type": "image_url", "image_url": {"url": {image}}, "role": "reference_image"}',
        "asset": '{"type": "image_url", "image_url": {"url": {image}}, "role": "reference_image"}',
    },
    "task_id_path": "id|data.id|data.task_id|task_id",
    "status_path": "status|data.status",
    "video_url_path": "content.video_url|video_url|data.output|data.video_url",
    "error_path": "error.message|error.code",
    "done_statuses": ["succeeded", "completed", "success", "done"],
    "fail_statuses": ["failed", "error", "cancelled", "canceled"],
}


# 火山方舟 doubao-seedream 图像生成（同步返回 data[].url / b64_json）
ARK_SEEDREAM_RECIPE = {
    "name": "ark_seedream",
    "display_name": "火山方舟 doubao-seedream（图像生成，同步）",
    "async": False,
    "auth": {
        "headers": {
            "Authorization": "Bearer {api_key}",
            "Content-Type": "application/json",
        },
    },
    "submit": {
        "method": "POST",
        "url": "{base_url}/api/v3/images/generations",
        "body": (
            '{"model": {model}, "prompt": {prompt}, "size": {size}, '
            '"seed": {seed}, "watermark": {watermark}, "response_format": {response_format}, "n": {n}}'
        ),
    },
    "image_field": "image",
    "result_path": "data",
    "error_path": "error.message|error.code",
}


def _json_str(value) -> str:
    """把值序列化成 JSON 字符串字面量（带引号 + 转义），用于注入模板。"""
    return json.dumps(str(value), ensure_ascii=False)


def _render_template(template: str, values: dict) -> str:
    """替换模板里的 {placeholder} 占位符；values 的键是占位符名，值须为已序列化字符串。

    用正则一次性替换：未知占位符（如 {task_id}）保留原样留给运行时处理；
    替换进去的值不会被二次扫描，避免内容里的花括号被误替换。
    """
    return re.sub(r"\{(\w+)\}", lambda m: values.get(m.group(1), m.group(0)), template)


def _status_set(statuses) -> set:
    """把状态词（list 或逗号分隔字符串）归一成小写集合。"""
    if isinstance(statuses, (list, tuple)):
        return {str(s).strip().lower() for s in statuses if str(s).strip()}
    return {s.strip().lower() for s in str(statuses).split(",") if s.strip()}


def _load_recipe(recipe):
    """解析 recipe 字段：内置名 / 文件路径 / JSON 字符串；空返回 None（走 legacy）。"""
    recipe = (recipe or "").strip()
    if not recipe:
        return None
    if recipe == "ark_seedance":
        return dict(ARK_SEEDANCE_RECIPE)
    if recipe == "ark_seedream":
        return dict(ARK_SEEDREAM_RECIPE)
    if os.path.isfile(recipe):
        with open(recipe, encoding="utf-8") as f:
            return json.load(f)
    try:
        return json.loads(recipe)
    except json.JSONDecodeError:
        raise ValueError(
            "recipe 无法解析：请填内置名 ark_seedance、或贴 recipe JSON、或填文件路径"
        ) from None


def _build_protocol_config(recipe, base, api_key, submit_url, poll_url_template,
                           task_id_field, status_field, video_url_field,
                           done_statuses, fail_statuses, prompt, model, resolution, ratio,
                           duration, generate_audio, watermark, seed, return_last_frame):
    """把 recipe（或旧显式字段）归一成统一的协议配置 dict。

    recipe 提供时优先，缺失项回退到节点显式字段；无 recipe 时用旧字段构造
    legacy 配置，行为与此前完全一致（向后兼容）。
    """
    r = _load_recipe(recipe)

    if r:
        auth_cfg = r.get("auth") or {}
        headers_tpl = auth_cfg.get("headers") or {}
        auth_headers = {k: _render_template(v, {"api_key": api_key})
                        for k, v in headers_tpl.items()}
        submit = r.get("submit") or {}
        submit_method = submit.get("method", "POST")
        submit_url_raw = submit.get("url") or DEFAULT_SUBMIT_URL
        submit_url = _resolve_url(base, _render_template(submit_url_raw, {"base_url": base}))
        body_template = submit.get("body") or "{}"
        poll_tpl = r.get("poll_url_template") or DEFAULT_POLL_URL_TEMPLATE
        poll_url_template = _resolve_url(
            base, _render_template(poll_tpl, {"base_url": base}))
        content_items = r.get("content_items") or {}
        task_id_path = r.get("task_id_path") or task_id_field
        status_path = r.get("status_path") or status_field
        video_url_path = r.get("video_url_path") or video_url_field
        error_path = r.get("error_path") or "error.message|error.code"
        done_set = _status_set(r.get("done_statuses") or done_statuses)
        fail_set = _status_set(r.get("fail_statuses") or fail_statuses)
        is_async = bool(r.get("async", True))
    else:
        auth_headers = {
            "Authorization": "Bearer %s" % api_key,
            "Content-Type": "application/json",
        }
        submit_method = "POST"
        submit_url = _resolve_url(base, submit_url or DEFAULT_SUBMIT_URL)
        body_template = (
            '{"model": {model}, "content": {content}, "resolution": {resolution}, '
            '"ratio": {ratio}, "duration": {duration}, "generate_audio": {generate_audio}, '
            '"watermark": {watermark}, "seed": {seed}, "return_last_frame": {return_last_frame}}'
        )
        poll_url_template = _resolve_url(base, (poll_url_template or DEFAULT_POLL_URL_TEMPLATE).strip())
        content_items = {
            "text": '{"type": "text", "text": {prompt}}',
            "first_frame": '{"type": "image_url", "image_url": {"url": {image}}, "role": "first_frame"}',
            "last_frame": '{"type": "image_url", "image_url": {"url": {image}}, "role": "last_frame"}',
            "reference_image": '{"type": "image_url", "image_url": {"url": {image}}, "role": "reference_image"}',
            "asset": '{"type": "image_url", "image_url": {"url": {image}}, "role": "reference_image"}',
        }
        task_id_path = task_id_field
        status_path = status_field
        video_url_path = video_url_field
        error_path = "error.message|error.code"
        done_set = _status_set(done_statuses)
        fail_set = _status_set(fail_statuses)
        is_async = True

    body_values = {
        "prompt": _json_str(prompt),
        "model": _json_str(model),
        "resolution": _json_str(resolution),
        "ratio": _json_str(ratio),
        "duration": str(int(duration)),
        "seed": str(int(seed)),
        "generate_audio": "true" if generate_audio else "false",
        "watermark": "true" if watermark else "false",
        "return_last_frame": "true" if return_last_frame else "false",
    }

    return {
        "auth_headers": auth_headers,
        "submit_method": submit_method,
        "submit_url": submit_url,
        "body_template": body_template,
        "body_values": body_values,
        "poll_url_template": poll_url_template,
        "content_items": content_items,
        "task_id_path": task_id_path,
        "status_path": status_path,
        "video_url_path": video_url_path,
        "error_path": error_path,
        "done_set": done_set,
        "fail_set": fail_set,
        "async": is_async,
    }


def _render_content(content_items, prompt, first_frame, last_frame, reference_images,
                    asset_ids, image_max_side, image_quality):
    """按 content_items 模板把文本/首尾帧/参考图/素材组装成 content JSON 数组字符串。"""
    parts = []

    def _item(tpl, **vals):
        return _render_template(tpl, {k: _json_str(v) for k, v in vals.items()})

    if prompt and "text" in content_items:
        parts.append(_item(content_items["text"], prompt=prompt))
    if first_frame is not None and "first_frame" in content_items:
        uri = _image_to_data_uri(first_frame, max_side=image_max_side, quality=image_quality)
        parts.append(_item(content_items["first_frame"], image=uri))
    if last_frame is not None and "last_frame" in content_items:
        uri = _image_to_data_uri(last_frame, max_side=image_max_side, quality=image_quality)
        parts.append(_item(content_items["last_frame"], image=uri))
    if reference_images is not None and "reference_image" in content_items:
        ref = reference_images if reference_images.dim() == 4 else reference_images.unsqueeze(0)
        for i in range(ref.shape[0]):
            uri = _image_to_data_uri(ref[i], max_side=image_max_side, quality=image_quality)
            parts.append(_item(content_items["reference_image"], image=uri))
    if asset_ids and "asset" in content_items:
        for aid in _split_ids(asset_ids):
            parts.append(_item(content_items["asset"], image="asset://" + aid))

    return "[" + ", ".join(parts) + "]"


class CYAiSeedanceVideo:
    """CYAI 中转站 doubao-seedance 视频生成（文生视频 / 图生视频 / 首尾帧 / 多图参考）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {"default": "", "tooltip": "中转站 API Key（sk-xxx）"},
                ),
                "base_url": (
                    "STRING",
                    {"default": "https://www.cyai.club", "tooltip": "中转站地址"},
                ),
                "submit_url": (
                    "STRING",
                    {"default": DEFAULT_SUBMIT_URL,
                     "tooltip": "提交任务的 POST 地址（相对 base_url 的路径或完整 URL）"},
                ),
                "poll_url_template": (
                    "STRING",
                    {"default": DEFAULT_POLL_URL_TEMPLATE,
                     "tooltip": "查询任务的 GET 地址模板，{task_id} 会被替换"},
                ),
                "model": (
                    "STRING",
                    {"default": "doubao-seedance-2-0-260128",
                     "tooltip": "视频生成模型名（可自由输入任意模型；常见：doubao-seedance-2-0-260128 / 2-0-fast-260128 / 2-0-mini-260615 / 2-5-260628）"},
                ),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "", "tooltip": "视频描述提示词"},
                ),
                "duration": (
                    "INT",
                    {"default": 5, "min": 4, "max": 30, "step": 1,
                     "tooltip": "视频时长（秒）。2.0 系列最大 10s，2.5 最大 30s"},
                ),
                "resolution": (
                    RESOLUTION_OPTIONS,
                    {"default": "720p", "tooltip": "分辨率（fast 系列不支持 1080p）"},
                ),
                "ratio": (
                    RATIO_OPTIONS,
                    {"default": "16:9",
                     "tooltip": "画幅比例。首帧/尾帧模式必须选 adaptive"},
                ),
                "generate_audio": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "是否生成声音"},
                ),
                "watermark": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "是否添加 AI 水印"},
                ),
                "seed": (
                    "INT",
                    {"default": -1, "min": -1, "max": 2147483647, "step": 1,
                     "tooltip": "随机种子（-1 表示随机）"},
                ),
                "return_last_frame": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "是否返回最后一帧图像"},
                ),
            },
            "optional": {
                "first_frame": (
                    "IMAGE",
                    {"tooltip": "可选：首帧图（图生视频，取 batch 第一张）"},
                ),
                "last_frame": (
                    "IMAGE",
                    {"tooltip": "可选：尾帧图（首尾帧模式）"},
                ),
                "reference_images": (
                    "IMAGE",
                    {"tooltip": "可选：多张参考图（role=reference_image，建议 1-4 张），与首尾帧互斥"},
                ),
                "poll_interval": (
                    "INT",
                    {"default": 10, "min": 1, "max": 60, "step": 1,
                     "tooltip": "轮询间隔（秒）"},
                ),
                "max_wait": (
                    "INT",
                    {"default": 900, "min": 30, "max": 3600, "step": 1,
                     "tooltip": "最长等待时间（秒）"},
                ),
                "image_max_side": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 4096, "step": 64,
                     "tooltip": "上传图像的最长边（像素），越大越清晰但请求体越大。默认 1024"},
                ),
                "image_quality": (
                    "INT",
                    {"default": 80, "min": 30, "max": 100, "step": 5,
                     "tooltip": "上传图像的 JPEG 质量（30-100），越低文件越小。默认 80"},
                ),
                "task_id_field": (
                    "STRING",
                    {"default": "id|data.id|data.task_id|task_id",
                     "tooltip": "提交响应里任务 ID 的字段路径，多个用 | 分隔依次尝试（支持 data.id 点路径）"},
                ),
                "status_field": (
                    "STRING",
                    {"default": "status|data.status",
                     "tooltip": "查询响应里状态的字段路径，多个用 | 分隔依次尝试"},
                ),
                "video_url_field": (
                    "STRING",
                    {"default": "content.video_url|video_url|data.output|data.video_url",
                     "tooltip": "查询响应里视频地址的字段路径，多个用 | 分隔依次尝试（支持 videos.0.url 数组索引）"},
                ),
                "done_statuses": (
                    "STRING",
                    {"default": "succeeded,completed,success,done",
                     "tooltip": "逗号分隔的成功状态词（不区分大小写）"},
                ),
                "fail_statuses": (
                    "STRING",
                    {"default": "failed,error,cancelled,canceled",
                     "tooltip": "逗号分隔的失败状态词（不区分大小写）"},
                ),
                # 新增控件一律追加在最后：ComfyUI 按「位置」给控件赋值，
                # 插在中间会让旧工作流整体错位一格（曾把 image_max_side 冲成 80）。
                "asset_ids": (
                    "STRING",
                    {"default": "", "tooltip": "可选：素材 ID（逗号分隔），按 asset://<id> 引用。一般是接「图像合并」的 asset_ids 输出；若开了 upload_images 则由节点自己生成"},
                ),
                "upload_images": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "一键模式：自动把接进来的图上传成素材并引用，绕开真人检测（会额外消耗一次素材上传）"},
                ),
                "image_host": (
                    ["uguu", "catbox"],
                    {"default": "uguu", "tooltip": "upload_images 开启时用的免费图床"},
                ),
                "asset_group_id": (
                    "STRING",
                    {"default": "", "tooltip": "upload_images 开启时：素材组 ID，留空自动建 AIGC 组"},
                ),
                "asset_name": (
                    "STRING",
                    {"default": "", "tooltip": "upload_images 开启时：素材名称，留空自动取名"},
                ),
                "upload_max_side": (
                    "INT",
                    {"default": 2048, "min": 300, "max": 6000, "step": 64,
                     "tooltip": "upload_images 开启时：上传图最长边"},
                ),
                "upload_quality": (
                    "INT",
                    {"default": 90, "min": 30, "max": 100, "step": 5,
                     "tooltip": "upload_images 开启时：上传图 JPEG 质量"},
                ),
                "recipe": (
                    "STRING",
                    {"default": "",
                     "tooltip": "协议配方：内置名 ark_seedance（火山方舟 doubao-seedance），或贴 recipe JSON / 填文件路径接入其它视频 API。留空 = 用上方显式的提交/轮询/字段路径参数（旧行为）"},
                ),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    OUTPUT_TOOLTIPS = ("生成视频（ComfyUI VIDEO 对象，可接预览/保存节点）",)
    FUNCTION = "generate"
    CATEGORY = "CYAI/Seedance"
    DESCRIPTION = "CYAI 中转站 doubao-seedance 视频生成（文生视频 / 图生视频 / 首尾帧 / 多图参考），输出 VIDEO"

    def generate(
        self,
        api_key,
        base_url,
        submit_url,
        poll_url_template,
        model,
        prompt,
        duration,
        resolution,
        ratio,
        generate_audio,
        watermark,
        seed,
        return_last_frame,
        first_frame=None,
        last_frame=None,
        reference_images=None,
        asset_ids="",
        poll_interval=10,
        max_wait=900,
        image_max_side=1024,
        image_quality=80,
        task_id_field="id|data.id|data.task_id|task_id",
        status_field="status|data.status",
        video_url_field="content.video_url|video_url|data.output|data.video_url",
        done_statuses="succeeded,completed,success,done",
        fail_statuses="failed,error,cancelled,canceled",
        upload_images=False,
        image_host="uguu",
        asset_group_id="",
        asset_name="",
        upload_max_side=2048,
        upload_quality=90,
        recipe="",
    ):
        api_key = (api_key or "").strip()
        prompt = (prompt or "").strip()
        if not api_key:
            raise ValueError("api_key 不能为空，请填写中转站的 sk-xxx")
        if not prompt:
            raise ValueError("prompt 不能为空")

        base = (base_url or "").strip().rstrip("/")
        if not base:
            base = "https://www.cyai.club"

        # 模式互斥校验
        has_frames = first_frame is not None or last_frame is not None

        # 一键模式：把接进来的图自动上传成素材，用 asset:// 引用（绕开真人检测）
        if upload_images:
            frames = []
            for im in (first_frame, last_frame, reference_images):
                if im is None:
                    continue
                t = im if im.dim() == 4 else im.unsqueeze(0)
                frames.extend(t[i] for i in range(t.shape[0]))
            if not frames:
                raise ValueError(
                    "upload_images 已开启，但没有接任何图。请把图接到 "
                    "reference_images（或 first_frame）后再开启；"
                    "若已有素材 ID，请直接填 asset_ids 并把 upload_images 关掉。"
                )
            auto_ids, _gid = _upload_frames_to_assets(
                frames, base, api_key, asset_name, image_host, "Image",
                "default", asset_group_id, "", upload_max_side, upload_quality,
                poll_interval, max_wait)
            # 自动上传后改为「素材引用」模式：清空首尾帧，避免与 base64 混传
            first_frame = last_frame = reference_images = None
            asset_ids = ",".join(filter(None, [asset_ids, auto_ids]))
            has_frames = False

        has_refs = reference_images is not None or bool(_split_ids(asset_ids))
        if has_frames and has_refs:
            raise ValueError("首帧/尾帧模式与多图参考(reference_images / asset_ids)互斥，只能选一种")

        # 首尾帧模式要求 ratio=adaptive（火山方舟硬约束）
        if has_frames and ratio != "adaptive":
            raise ValueError("首帧/尾帧模式要求 ratio 必须设为 adaptive，请把画幅比例改为 adaptive")

        # 解析协议配置（recipe 优先，旧显式字段兜底，向后兼容）
        cfg = _build_protocol_config(
            recipe, base, api_key, submit_url, poll_url_template,
            task_id_field, status_field, video_url_field, done_statuses, fail_statuses,
            prompt, model, resolution, ratio, duration, generate_audio, watermark,
            seed, return_last_frame,
        )

        headers = cfg["auth_headers"]

        # 按 recipe 的 content_items 模板组装 content 数组
        content_str = _render_content(
            cfg["content_items"], prompt, first_frame, last_frame,
            reference_images, asset_ids, image_max_side, image_quality)

        # 渲染 body 模板 -> 最终请求体
        body_values = dict(cfg["body_values"])
        body_values["content"] = content_str
        body_str = _render_template(cfg["body_template"], body_values)
        try:
            body = json.loads(body_str)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                "请求体模板渲染后不是有效 JSON：%s；渲染结果：%s" % (e, body_str[:1500])) from e

        created = _http_json(cfg["submit_method"], cfg["submit_url"], headers, body, timeout=180.0)

        task_id = _extract_task_id(created, cfg["task_id_path"])
        if not task_id:
            raise RuntimeError("提交任务失败，响应里找不到任务 ID（路径 %s）：%s"
                               % (cfg["task_id_path"], created))

        # 同步接口：提交响应直接带视频地址，跳过轮询
        if not cfg["async"]:
            video_url = _extract_video_url(created, cfg["video_url_path"])
            if not video_url:
                raise RuntimeError("同步接口未返回视频地址（路径 %s）：%s"
                                   % (cfg["video_url_path"], created))
            return (_video_from_bytes(_download(video_url, headers)),)

        poll_url = cfg["poll_url_template"].replace("{task_id}", str(task_id))

        done_set = cfg["done_set"]
        fail_set = cfg["fail_set"]

        deadline = time.time() + float(max_wait)

        # 进度条：按最大轮询次数显示，每轮 poll 前进一格
        total_rounds = max(1, int(float(max_wait) / max(1, float(poll_interval))) + 1)
        pbar = ProgressBar(total_rounds)

        # 先查一次，再进入 sleep+查询 的循环，避免任务秒完成还白等一个轮询间隔
        last_state = _http_json("GET", poll_url, headers, timeout=60.0)

        while time.time() < deadline:
            # 用户点击取消 / 中断时，让 ComfyUI 能真正停掉这个节点
            comfy.model_management.throw_exception_if_processing_interrupted()

            status = _extract_status(last_state, cfg["status_path"]).lower()

            if status in fail_set:
                err = _extract_error(last_state, cfg["error_path"])
                raise RuntimeError(
                    "视频生成任务失败：%s%s" % (err or "", "" if err else " %s" % (last_state,))
                )

            if status in done_set:
                video_url = _extract_video_url(last_state, cfg["video_url_path"])
                if not video_url:
                    raise RuntimeError(
                        "任务已完成，但按路径 %s 未找到视频地址：%s"
                        % (cfg["video_url_path"], last_state)
                    )
                pbar.update(total_rounds)  # 完成，进度走满
                video_bytes = _download(video_url, headers)
                return (_video_from_bytes(video_bytes),)

            pbar.update(1)

            # sleep 拆成小段，每段检查一次中断，让取消更及时
            remain = float(poll_interval)
            while remain > 0:
                step = min(remain, 2.0)
                time.sleep(step)
                remain -= step
                comfy.model_management.throw_exception_if_processing_interrupted()
            last_state = _http_json("GET", poll_url, headers, timeout=60.0)

        raise RuntimeError("轮询超时（%s 秒），最后状态：%s" % (int(max_wait), last_state))


class CYAiImageGen:
    """通用图像生成节点（recipe 驱动），默认火山方舟 doubao-seedream（同步接口）。

    复用视频节点同一套 recipe 基础设施（_load_recipe / _render_template / _http_json /
    _resolve_url / _resolve_field / _download），差异只在：输出 IMAGE、请求体字段
    （size/seed/watermark/response_format）、响应结构（data[].url 或 b64_json）。
    输出 IMAGE 可直接接视频节点的 first_frame / reference_images，做「文生图 → 图生视频」。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {"default": "", "tooltip": "中转站 API Key（sk-xxx）"},
                ),
                "base_url": (
                    "STRING",
                    {"default": "https://www.cyai.club", "tooltip": "中转站地址"},
                ),
                "model": (
                    "STRING",
                    {"default": "doubao-seedream-4-0-250828",
                     "tooltip": "图像生成模型名（可自由输入；常见 doubao-seedream-4-0 / 4-5 / 5-0 系列）"},
                ),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "", "tooltip": "图像描述提示词"},
                ),
                "size": (
                    "STRING",
                    {"default": "2K", "tooltip": "输出尺寸：1K/2K/3K/4K 档位，或像素尺寸如 2048x2048"},
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {"tooltip": "可选：图生图参考图（多张 = 多图融合）"},
                ),
                "seed": (
                    "INT",
                    {"default": -1, "min": -1, "max": 2147483647, "step": 1,
                     "tooltip": "随机种子（-1 表示随机）"},
                ),
                "watermark": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "是否添加 AI 水印"},
                ),
                "recipe": (
                    "STRING",
                    {"default": "",
                     "tooltip": "协议配方：内置名 ark_seedream（火山方舟 doubao-seedream），或贴 recipe JSON / 填文件路径接入其它图像 API。留空 = 默认 ark_seedream"},
                ),
                "submit_url": (
                    "STRING",
                    {"default": "",
                     "tooltip": "可选：提交接口地址（相对 base_url 或完整 URL）。留空用配方/内置默认；中转站端口不同时直接改这里，不必写整份 recipe"},
                ),
                "response_format": (
                    ["url", "b64_json"],
                    {"default": "url",
                     "tooltip": "返回格式：url = 临时下载链接，b64_json = base64 内嵌。按中转站支持情况选"},
                ),
                "extra_body": (
                    "STRING",
                    {"multiline": True, "default": "",
                     "tooltip": "可选：额外请求体字段（JSON 对象），合并进请求体并覆盖同名字段。中转站字段命名/结构与默认不同时，在这里自由声明任意字段，如 {\"width\": 1024, \"height\": 1024, \"cfg_scale\": 7.5}"},
                ),
                "url_field": (
                    "STRING",
                    {"default": "url",
                     "tooltip": "响应里图片 URL 的字段名（图片项内）。中转站命名不同时改，如 url / image_url / link"},
                ),
                "b64_field": (
                    "STRING",
                    {"default": "b64_json",
                     "tooltip": "响应里图片 base64 的字段名（图片项内），如 b64_json / base64 / image_b64"},
                ),
                "result_path": (
                    "STRING",
                    {"default": "data",
                     "tooltip": "响应里图片数组的字段路径（点路径），如 data / result.images / output。也支持直接指向字符串 URL 或字符串数组"},
                ),
                "image_field": (
                    "STRING",
                    {"default": "image",
                     "tooltip": "图生图参考字段名（请求体里承载参考图的字段）。中转站命名不同时改，如 image / init_image / input_image / image_url"},
                ),
                "n": (
                    "INT",
                    {"default": 1, "min": 1, "max": 10, "step": 1,
                     "tooltip": "一次生成张数（n）。部分中转站支持一次多张；不支持的中转站会忽略或报错，保持 1 即可"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_TOOLTIPS = ("生成的图像（ComfyUI IMAGE 张量，可接预览/保存，或视频节点的参考图）",)
    FUNCTION = "generate"
    CATEGORY = "CYAI/Image"
    DESCRIPTION = "通用图像生成（recipe 驱动），默认火山方舟 doubao-seedream，输出 IMAGE"

    def generate(self, api_key, base_url, model, prompt, size,
                 image=None, seed=-1, watermark=False, recipe="",
                 submit_url="", response_format="url", extra_body="",
                 url_field="url", b64_field="b64_json", result_path="data",
                 image_field="image", n=1):
        api_key = (api_key or "").strip()
        prompt = (prompt or "").strip()
        if not api_key:
            raise ValueError("api_key 不能为空，请填写中转站的 sk-xxx")
        if not prompt:
            raise ValueError("prompt 不能为空")

        base = (base_url or "").strip().rstrip("/")
        if not base:
            base = "https://www.cyai.club"

        r = _load_recipe(recipe) or dict(ARK_SEEDREAM_RECIPE)

        # 鉴权头
        auth_cfg = r.get("auth") or {}
        headers = {k: _render_template(v, {"api_key": api_key})
                   for k, v in (auth_cfg.get("headers") or {}).items()}

        # 提交地址：显式 submit_url > 配方 submit.url > 内置默认
        submit = r.get("submit") or {}
        method = submit.get("method", "POST")
        if (submit_url or "").strip():
            url = _resolve_url(base, (submit_url or "").strip())
        elif submit.get("url"):
            url = _resolve_url(base, _render_template(submit.get("url"), {"base_url": base}))
        else:
            url = _resolve_url(base, "/api/v3/images/generations")
        body_template = submit.get("body") or "{}"

        # 通用标量占位符
        body_values = {
            "prompt": _json_str(prompt),
            "model": _json_str(model),
            "size": _json_str(size),
            "seed": str(int(seed)),
            "watermark": "true" if watermark else "false",
            "response_format": _json_str(response_format),
            "n": str(int(n)),
        }

        # 图生图：image 字段名（显式参数 > recipe > 默认），单图=字符串、多图=数组
        image_field = (image_field or "").strip() or (r.get("image_field") or "image")
        image_val = None
        if image is not None:
            imgs = image if image.dim() == 4 else image.unsqueeze(0)
            uris = [_image_to_data_uri(imgs[i]) for i in range(imgs.shape[0])]
            image_val = uris[0] if len(uris) == 1 else uris

        body_str = _render_template(body_template, body_values)
        try:
            body = json.loads(body_str)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                "请求体模板渲染后不是有效 JSON：%s；渲染结果：%s" % (e, body_str[:1500])) from e
        if image_val is not None:
            body[image_field] = image_val

        # 自由字段：extra_body 是 JSON 对象，合并进请求体并覆盖同名字段
        if (extra_body or "").strip():
            try:
                extra = json.loads(extra_body)
            except json.JSONDecodeError as e:
                raise RuntimeError("extra_body 不是有效 JSON：%s" % e) from e
            if not isinstance(extra, dict):
                raise RuntimeError("extra_body 必须是 JSON 对象（{...}）")
            body.update(extra)

        resp = _http_json(method, url, headers, body, timeout=180.0)

        # 异步接口（少数图像 API 是任务制）：轮询到完成再取结果
        if r.get("async"):
            task_id = _resolve_first(resp, r.get("task_id_path") or "id|data.id")
            if not task_id:
                raise RuntimeError("提交响应里找不到任务 ID：%s" % resp)
            poll_tpl = r.get("poll_url_template") or ""
            poll_url = _resolve_url(base, _render_template(poll_tpl, {"base_url": base}))
            poll_url = poll_url.replace("{task_id}", str(task_id))
            status_path = r.get("status_path") or "status"
            done_set = _status_set(r.get("done_statuses") or ["succeeded", "success", "done"])
            fail_set = _status_set(r.get("fail_statuses") or ["failed", "error"])
            error_path = r.get("error_path") or "error.message|error.code"
            deadline = time.time() + 900.0
            while time.time() < deadline:
                comfy.model_management.throw_exception_if_processing_interrupted()
                state = _http_json("GET", poll_url, headers, timeout=60.0)
                st = str(_resolve_first(state, status_path) or "").lower()
                if st in fail_set:
                    raise RuntimeError("图像生成任务失败：%s"
                                       % (_resolve_first(state, error_path) or state))
                if st in done_set:
                    resp = state
                    break
                time.sleep(5.0)
            else:
                raise RuntimeError("图像生成轮询超时")

        # 解析图片：result_path 指向图片数组（每项 url 或 b64）；字段名可配
        result_path = (result_path or "").strip() or (r.get("result_path") or "data")
        url_field = (url_field or "").strip() or (r.get("url_field") or "url")
        b64_field = (b64_field or "").strip() or (r.get("b64_field") or "b64_json")
        raw = _resolve_field(resp, result_path)
        if raw is None:
            raise RuntimeError("响应里找不到图像（路径 %s）：%s" % (result_path, resp))

        # 归一成列表：dict -> [dict]，str -> [url]，list 保持
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict):
            items = [raw]
        elif isinstance(raw, str):
            items = [raw]
        else:
            raise RuntimeError("图像字段结构无法识别（路径 %s）：%s" % (result_path, resp))

        frames = []
        for it in items:
            if isinstance(it, str):                      # 直接是 URL 字符串
                frames.append(_image_from_bytes(_download(it, headers)))
            elif isinstance(it, dict):
                b64 = it.get(b64_field) or it.get("b64_json")
                u = it.get(url_field) or it.get("url")
                if b64:
                    frames.append(_image_from_bytes(base64.b64decode(b64)))
                elif u:
                    frames.append(_image_from_bytes(_download(u, headers)))
                else:
                    raise RuntimeError("图像项既无 %s 也无 %s：%s" % (url_field, b64_field, it))
            else:
                raise RuntimeError("图像项结构异常：%s" % it)
        if not frames:
            raise RuntimeError("未解析出任何图像")

        return (torch.cat(frames, dim=0),)


class CYAiSeedanceUsage:
    """查询 CYAI 中转站任务消耗（从任务列表累计 usage.completion_tokens）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {"default": "", "tooltip": "中转站 API Key（sk-xxx）"},
                ),
                "base_url": (
                    "STRING",
                    {"default": "https://www.cyai.club", "tooltip": "中转站地址"},
                ),
                "list_url": (
                    "STRING",
                    {"default": DEFAULT_SUBMIT_URL,
                     "tooltip": "任务列表 GET 地址（相对 base_url 或完整 URL）"},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("usage_info",)
    OUTPUT_TOOLTIPS = ("任务消耗统计（任务数 / 累计 token）",)
    FUNCTION = "query"
    CATEGORY = "CYAI/Seedance"
    DESCRIPTION = "查询中转站历史任务与 token 消耗（余额需到网页后台查看）"

    def query(self, api_key, base_url, list_url):
        api_key = (api_key or "").strip()
        if not api_key:
            raise ValueError("api_key 不能为空，请填写中转站的 sk-xxx")

        base = (base_url or "").strip().rstrip("/")
        if not base:
            base = "https://www.cyai.club"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        url = _resolve_url(base, list_url or DEFAULT_SUBMIT_URL)
        resp = _http_json("GET", url, headers, timeout=30.0)

        items = resp.get("items") if isinstance(resp, dict) else None
        if not isinstance(items, list):
            raise RuntimeError("查询任务列表失败，响应格式异常：%s" % (str(resp)[:800],))

        total = resp.get("total", len(items))
        succeeded = 0
        failed = 0
        total_tokens = 0
        for it in items:
            st = str(it.get("status") or "").lower()
            if st == "succeeded":
                succeeded += 1
            elif st == "failed":
                failed += 1
            usage = it.get("usage")
            if isinstance(usage, dict):
                total_tokens += int(usage.get("completion_tokens") or 0)

        lines = [
            "中转站任务消耗统计",
            f"  历史任务总数：{total}",
            f"  成功：{succeeded}  失败：{failed}",
            f"  累计消耗：{total_tokens:,} tokens",
            "",
            "说明：",
            "  - 该 key 无法通过 API 查询余额（/api/user/self 鉴权不通过），",
            "    请到 cyai.club 网页后台查看剩余额度。",
            "  - token 到金额的换算以中转站定价为准。",
        ]
        info = "\n".join(lines)
        return (info,)


# 素材上传相关的内部工具（图像合并不再上传；上传由视频节点 upload_images 触发）

def _ensure_group(base_url, api_key, group_id, group_name, fallback_name, project_name):
    """group_id 为空则自动创建 AIGC 组；返回 group_id。"""
    group_id = (group_id or "").strip()
    if group_id:
        return group_id
    payload = {
        "Name": (group_name or "").strip() or ("auto-" + (fallback_name or "asset"))[:64],
        "ProjectName": (project_name or "default").strip(),
        "GroupType": "AIGC",
    }
    result = _ark_action(base_url, api_key, "CreateAssetGroup", payload, timeout=60.0)
    gid = result.get("Id") or ""
    if not gid:
        raise RuntimeError("自动创建素材组失败，响应里没有 Id：%s" % result)
    return gid


def _wait_assets_active(base_url, api_key, asset_ids, project_name, poll_interval, max_wait):
    """轮询 GetAsset 直到全部 Active；任一 Failed 或超时报错。"""
    pending = set(asset_ids)
    deadline = time.time() + float(max_wait)
    detail = None
    while pending and time.time() < deadline:
        comfy.model_management.throw_exception_if_processing_interrupted()
        for aid in list(pending):
            detail = _ark_action(base_url, api_key, "GetAsset",
                                 {"Id": aid, "ProjectName": project_name}, timeout=60.0)
            status = str(detail.get("Status") or "").lower()
            if status == "active":
                pending.discard(aid)
            elif status == "failed":
                err = detail.get("Error") or {}
                raise RuntimeError("素材处理失败（%s）：%s %s" % (
                    aid, err.get("Code") or "", err.get("Message") or ""))
        if pending:
            time.sleep(float(poll_interval))
    if pending:
        raise RuntimeError("素材处理超时（%s 秒），未就绪：%s" % (int(max_wait), sorted(pending)))


def _upload_frames_to_assets(frames, base_url, api_key, name, image_host, asset_type,
                             project_name, group_id, group_name, upload_max_side,
                             upload_quality, poll_interval, max_wait):
    """把一批 IMAGE 帧逐张上传图床 → 建素材 → 轮询到 Active。

    返回 (asset_ids_逗号串, group_id)。
    """
    api_key = (api_key or "").strip()
    if not api_key:
        raise ValueError("开启素材上传（upload_images=true）时必须填写 api_key")
    name = (name or "").strip() or "asset"
    project_name = (project_name or "default").strip()
    group_id = _ensure_group(base_url, api_key, group_id, group_name, name, project_name)

    asset_ids = []
    for i, frame in enumerate(frames):
        img_bytes = _jpeg_bytes_from_image(
            frame, max_side=int(upload_max_side), quality=int(upload_quality))
        url = _upload_to_host(img_bytes, "cyai_%d_%d.jpg" % (int(time.time()), i),
                              provider=image_host)
        asset_name = name if len(frames) == 1 else "%s-%d" % (name, i + 1)
        result = _ark_action(base_url, api_key, "CreateAsset", {
            "GroupId": group_id, "Name": asset_name, "URL": url,
            "AssetType": asset_type, "ProjectName": project_name}, timeout=120.0)
        aid = result.get("Id") or ""
        if not aid:
            raise RuntimeError("创建素材失败，响应里没有 Id：%s" % result)
        asset_ids.append(aid)

    _wait_assets_active(base_url, api_key, asset_ids, project_name, poll_interval, max_wait)
    return (",".join(asset_ids), group_id)


class CYAiImageBatch:
    """图像合并节点：把多张图按顺序拼成一个 IMAGE batch。

    提供 9 个可选输入口 image_1~image_9（Seedance 2.0 多图参考上限 9 张）；
    每个口也能接本身带 batch 的图像（如 Load Image Sequence）。

    注意：本节点**不再负责上传素材**。要实现「写实人物绕开真人检测」，
    请把输出接到「CYAI Seedance 视频生成」并打开它的 `upload_images` 开关
    （视频节点会在内部把图上传成素材并以 asset:// 引用）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, 10):
            optional[f"image_{i}"] = (
                "IMAGE",
                {"tooltip": f"第 {i} 张参考图（按顺序拼接）"},
            )
        optional["max_side"] = (
            "INT",
            {"default": 1024, "min": 256, "max": 4096, "step": 64,
             "tooltip": "统一缩放边长（正方形、白底居中填充）。多图尺寸/比例不一致时，靠它把每张图归一成同尺寸再拼接"},
        )
        return {"optional": optional}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_TOOLTIPS = ("合并后的图像 batch（按 image_1 → image_9 顺序）",)
    FUNCTION = "merge"
    CATEGORY = "CYAI/Seedance"
    DESCRIPTION = "把最多 9 张图合并成一个 IMAGE batch，用于接 Seedance 的 reference_images 多图参考输入"

    def merge(self, max_side=1024, **kwargs):
        frames = []
        for i in range(1, 10):
            img = kwargs.get(f"image_{i}")
            if img is None:
                continue
            if img.dim() == 3:
                img = img.unsqueeze(0)
            frames.extend(img[j] for j in range(img.shape[0]))
        if not frames:
            raise ValueError("至少连接一张图像（image_1 ~ image_9）")

        # 不同来源的图尺寸/比例往往不同，torch.cat(dim=0) 要求 H/W 完全一致，
        # 直接拼接会报 "Sizes of tensors must match"。这里把每张图统一归一成
        # max_side 正方形（长边缩放 + 白底居中填充），既保证拼接成功，
        # 又避免异形比例/透明图在 JPEG 里产生黑边污染参考图。
        normalized = [_uniform_square(f, int(max_side)) for f in frames]
        return (torch.cat(normalized, dim=0),)


class CYAiCreateVerifySession:
    """创建火山方舟真人认证会话（CreateVisualValidateSession）。

    输出 H5Link（本人活体认证链接）与 BytedToken（查询结果用）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "中转站 API Key（sk-xxx）"}),
                "base_url": ("STRING", {"default": "https://www.cyai.club", "tooltip": "中转站地址"}),
                "callback_url": ("STRING", {"default": "", "tooltip": "真人认证完成后的回调地址（必填）"}),
                "project_name": ("STRING", {"default": "default", "tooltip": "项目名，默认 default"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("h5_link", "byted_token")
    OUTPUT_TOOLTIPS = ("真人认证 H5 链接（发给本人完成活体认证）", "认证会话令牌（供查询认证结果使用）")
    FUNCTION = "create_session"
    CATEGORY = "CYAI/Asset"
    DESCRIPTION = "创建火山方舟真人认证会话，输出 H5Link（活体认证）与 BytedToken"

    def create_session(self, api_key, base_url, callback_url, project_name="default"):
        api_key = (api_key or "").strip()
        if not api_key:
            raise ValueError("api_key 不能为空")
        callback_url = (callback_url or "").strip()
        if not callback_url:
            raise ValueError("callback_url 不能为空，请填写认证完成后的回调地址")

        result = _ark_action(base_url, api_key, "CreateVisualValidateSession", {
            "CallbackURL": callback_url,
            "ProjectName": (project_name or "default").strip(),
        }, timeout=60.0)
        token = result.get("BytedToken") or ""
        link = result.get("H5Link") or ""
        if not token:
            raise RuntimeError("创建真人认证会话失败，响应里没有 BytedToken：%s" % result)
        return (link, token)


class CYAiGetVerifyResult:
    """查询真人认证结果（GetVisualValidateResult），轮询直到拿到真人素材组 GroupId。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "中转站 API Key（sk-xxx）"}),
                "base_url": ("STRING", {"default": "https://www.cyai.club", "tooltip": "中转站地址"}),
                "byted_token": ("STRING", {"default": "", "tooltip": "CreateVisualValidateSession 返回的 BytedToken"}),
                "project_name": ("STRING", {"default": "default", "tooltip": "项目名，默认 default"}),
                "poll_interval": ("INT", {"default": 10, "min": 1, "max": 60, "step": 1, "tooltip": "轮询间隔（秒）"}),
                "max_wait": ("INT", {"default": 600, "min": 30, "max": 3600, "step": 1, "tooltip": "最长等待（秒），认证需本人操作，建议给足时间"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("group_id",)
    OUTPUT_TOOLTIPS = ("真人认证对应的素材组 ID",)
    FUNCTION = "get_result"
    CATEGORY = "CYAI/Asset"
    DESCRIPTION = "轮询查询真人认证结果，返回真人素材组 GroupId"

    def get_result(self, api_key, base_url, byted_token, project_name="default",
                   poll_interval=10, max_wait=600):
        api_key = (api_key or "").strip()
        byted_token = (byted_token or "").strip()
        if not api_key:
            raise ValueError("api_key 不能为空")
        if not byted_token:
            raise ValueError("byted_token 不能为空，请先创建真人认证会话")

        payload = {"BytedToken": byted_token}
        if (project_name or "").strip():
            payload["ProjectName"] = (project_name or "").strip()

        deadline = time.time() + float(max_wait)
        last_err = ""
        while time.time() < deadline:
            comfy.model_management.throw_exception_if_processing_interrupted()
            try:
                result = _ark_action(base_url, api_key, "GetVisualValidateResult", payload, timeout=60.0)
            except RuntimeError as e:
                last_err = str(e)  # 认证未完成时上游可能报错，视为未就绪继续等
            else:
                group_id = result.get("GroupId") or ""
                if group_id:
                    return (group_id,)
            time.sleep(float(poll_interval))
        raise RuntimeError(
            "真人认证结果查询超时（%s 秒），尚未拿到 GroupId；请确认本人已在 H5 链接完成认证。最后错误：%s"
            % (int(max_wait), last_err or "无")
        )


class CYAiAssetGroup:
    """素材组节点（火山方舟素材资产接口）。

    action 单选：
      - 创建素材组（AIGC 虚拟人像）：CreateAssetGroup → 返回 group_id
      - 查询素材组列表：ListAssetGroups → 返回列表文本
      - 查询素材组详情：GetAssetGroup → 返回详情文本
    AIGC 组用于托管虚拟人像素材（动漫/漫剧/插画/AI 生成角色），
    上传后的素材可在视频任务里用 asset://<id> 引用，绕开 base64 真人检测。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "中转站 API Key（sk-xxx）"}),
                "base_url": ("STRING", {"default": "https://www.cyai.club", "tooltip": "中转站地址"}),
                "action": (
                    ["创建素材组（AIGC）", "查询素材组列表", "查询素材组详情"],
                    {"default": "创建素材组（AIGC）", "tooltip": "选择要执行的操作"},
                ),
                "project_name": ("STRING", {"default": "default", "tooltip": "项目名，默认 default"}),
            },
            "optional": {
                "name": ("STRING", {"default": "", "tooltip": "创建：素材组名称（必填，建议 ≤64 字符）"}),
                "description": ("STRING", {"default": "", "tooltip": "创建：素材组描述（可选）"}),
                "group_type": ("STRING", {"default": "AIGC", "tooltip": "创建：素材组类型。AIGC=虚拟人像；真人素材组须走真人认证流程"}),
                "group_id": ("STRING", {"default": "", "tooltip": "查询详情：素材组 ID"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("group_id", "info")
    OUTPUT_TOOLTIPS = ("创建成功/详情查询的素材组 ID（可接「上传素材」的 group_id）", "列表或详情文本")
    FUNCTION = "run"
    CATEGORY = "CYAI/Asset"
    DESCRIPTION = "素材组操作：创建 AIGC 虚拟人像组 / 查询列表 / 查询详情"

    def run(self, api_key, base_url, action, project_name="default",
            name="", description="", group_type="AIGC", group_id=""):
        api_key = (api_key or "").strip()
        project_name = (project_name or "default").strip()
        if not api_key:
            raise ValueError("api_key 不能为空")

        if action == "创建素材组（AIGC）":
            name = (name or "").strip()
            if not name:
                raise ValueError("name 不能为空，请填写素材组名称")
            payload = {"Name": name, "ProjectName": project_name, "GroupType": group_type or "AIGC"}
            if description:
                payload["Description"] = description
            result = _ark_action(base_url, api_key, "CreateAssetGroup", payload, timeout=60.0)
            gid = result.get("Id") or ""
            if not gid:
                raise RuntimeError("创建素材组失败，响应里没有 Id：%s" % result)
            return (gid, "创建成功：%s（%s）" % (gid, name))

        if action == "查询素材组列表":
            # Filter 必填（可为空对象），否则上游报 Filter is required
            result = _ark_action(base_url, api_key, "ListAssetGroups", {
                "Filter": {"GroupType": group_type} if group_type else {},
                "PageNumber": 1, "PageSize": 20, "ProjectName": project_name,
            }, timeout=60.0)
            items = result.get("Items") or []
            lines = ["素材组共 %s 个：" % result.get("TotalCount", len(items))]
            for it in items:
                lines.append("  %s  %s  %s" % (it.get("Id"), it.get("GroupType"), it.get("Name")))
            return ("", "\n".join(lines))

        # 查询素材组详情
        group_id = (group_id or "").strip()
        if not group_id:
            raise ValueError("group_id 不能为空，请填写要查询的素材组 ID")
        result = _ark_action(base_url, api_key, "GetAssetGroup", {
            "Id": group_id, "ProjectName": project_name}, timeout=60.0)
        return (result.get("Id") or group_id, json.dumps(result, ensure_ascii=False, indent=2))


class CYAiCreateAsset:
    """创建素材（CreateAsset + 轮询 GetAsset 到 Active）。

    图片来源二选一：
      - 接 image 输入（本地图）→ 自动上传公网图床拿 URL → 创建素材（全自动，推荐）
      - 或直接填 url（已经有公网链接时）
    group_id 留空时自动创建 AIGC 素材组。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "中转站 API Key（sk-xxx）"}),
                "base_url": ("STRING", {"default": "https://www.cyai.club", "tooltip": "中转站地址"}),
                "name": ("STRING", {"default": "", "tooltip": "素材名称（必填）"}),
                "image_host": (
                    ["uguu", "catbox"],
                    {"default": "uguu", "tooltip": "自动上传用的免费匿名图床：uguu 稳定但链接数小时失效；catbox 永久但机房 IP 常被拒"},
                ),
                "asset_type": (["Image", "Video", "Audio"], {"default": "Image", "tooltip": "素材类型，需与真实文件一致"}),
                "project_name": ("STRING", {"default": "default", "tooltip": "项目名，默认 default"}),
                "poll_interval": ("INT", {"default": 5, "min": 1, "max": 60, "step": 1, "tooltip": "轮询间隔（秒）"}),
                "max_wait": ("INT", {"default": 300, "min": 30, "max": 1800, "step": 1, "tooltip": "最长等待（秒）"}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "要上传的本地图（接「图像合并」的输出）→ 自动传图床拿 URL"}),
                "url": ("STRING", {"default": "", "tooltip": "已有公网 URL 时直接填这里（与 image 二选一，填了优先用 URL）"}),
                "group_id": ("STRING", {"default": "", "tooltip": "可选：所属素材组 ID。留空则自动创建 AIGC 组；真人素材填认证结果给的 GroupId"}),
                "group_name": ("STRING", {"default": "", "tooltip": "可选：group_id 留空时，自动创建素材组的名称"}),
                "upload_max_side": ("INT", {"default": 2048, "min": 300, "max": 6000, "step": 64, "tooltip": "上传图最长边（素材接口上限 6000px）"}),
                "upload_quality": ("INT", {"default": 90, "min": 30, "max": 100, "step": 5, "tooltip": "上传图 JPEG 质量"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("asset_ids", "group_id")
    OUTPUT_TOOLTIPS = ("素材 ID（多张则逗号分隔，直接接视频节点的 asset_ids）", "所属素材组 ID（可复用）")
    FUNCTION = "create_asset"
    CATEGORY = "CYAI/Asset"
    DESCRIPTION = "本地图(batch)→图床→素材（或直接给 URL），轮询到 Active；输出逗号分隔的 asset_ids"

    def create_asset(self, api_key, base_url, name, image_host="uguu", asset_type="Image",
                     project_name="default", poll_interval=5, max_wait=300,
                     image=None, url="", group_id="", group_name="",
                     upload_max_side=2048, upload_quality=90):
        api_key = (api_key or "").strip()
        name = (name or "").strip()
        url = (url or "").strip()
        group_id = (group_id or "").strip()
        project_name = (project_name or "default").strip()
        if not api_key:
            raise ValueError("api_key 不能为空")
        if not name:
            raise ValueError("name 不能为空，请填写素材名称")
        if not url and image is None:
            raise ValueError("请连接 image 输入（本地图）或在 url 填公网地址，二者必填其一")

        # 给公网 URL 时无需建组前置（_ensure_group 会处理），直接建素材
        if url:
            group_id = _ensure_group(base_url, api_key, group_id, group_name, name, project_name)
            result = _ark_action(base_url, api_key, "CreateAsset", {
                "GroupId": group_id, "Name": name, "URL": url,
                "AssetType": asset_type, "ProjectName": project_name}, timeout=120.0)
            asset_id = result.get("Id") or ""
            if not asset_id:
                raise RuntimeError("创建素材失败，响应里没有 Id：%s" % result)
            _wait_assets_active(base_url, api_key, [asset_id], project_name,
                                poll_interval, max_wait)
            return (asset_id, group_id)

        # 本地图：按 batch 展开，逐张上传建素材
        frames = list(image if image.dim() == 4 else image.unsqueeze(0))
        return _upload_frames_to_assets(
            frames, base_url, api_key, name, image_host, asset_type,
            project_name, group_id, group_name, upload_max_side, upload_quality,
            poll_interval, max_wait)
