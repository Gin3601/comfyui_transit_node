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
import time
import urllib.error
import urllib.request

import numpy as np
import torch
from PIL import Image

# ComfyUI 进度与中断适配（节点只在 ComfyUI 内运行，这些模块必定存在）
from comfy.utils import ProgressBar
import comfy.model_management

# 用户提供的模型列表（CYAI 中转站）
MODEL_OPTIONS = [
    "doubao-seedance-2-0-260128",
    "doubao-seedance-2-0-fast-260128",
    "doubao-seedance-2-0-mini-260615",
    "doubao-seedance-2-5-260628",
]

RESOLUTION_OPTIONS = ["480p", "720p", "1080p"]
RATIO_OPTIONS = ["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"]

# CYAI / 火山方舟固定接口路径（可被节点上的 submit_url / poll_url_template 覆盖）
DEFAULT_SUBMIT_URL = "/api/v3/contents/generations/tasks"
DEFAULT_POLL_URL_TEMPLATE = "/api/v3/contents/generations/tasks/{task_id}"

DONE_STATUSES = {"SUCCEEDED"}
FAIL_STATUSES = {"FAILED", "CANCELLED", "CANCELED"}


def _image_to_data_uri(image: torch.Tensor, max_side: int = 1024, quality: int = 80) -> str:
    """把 ComfyUI 的 IMAGE 张量 [B,H,W,C] (0-1 float) 转成 base64 data URI。

    火山方舟要求单张图 <10MB、整个请求体 <64MB。参考图/首尾帧只需让模型看清
    主体与风格，无需原始分辨率，故默认限制最长边 1024、JPEG 质量 80；
    单张约 100-300KB，9 张约 1-3MB，远低于 64MB 上限。
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
        raise RuntimeError(
            "API 请求失败 (HTTP %s)：%s" % (e.code, body[:1500].decode("utf-8", "replace"))
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


def _extract_error(state: dict) -> str | None:
    """失败时返回 error.message（或 error.code），否则 None。"""
    err = state.get("error")
    if isinstance(err, dict):
        return err.get("message") or err.get("code")
    return None


def _extract_video_url(state: dict, video_url_field: str):
    """按可配路径从查询结果取 mp4 地址（支持多路径兜底）。"""
    return _resolve_first(state, video_url_field)


class CYAiSeedanceVideo:
    """CYAI 中转站 doubao-seedance 视频生成（文生视频 / 图生视频 / 首尾帧 / 多图参考）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {"default": "", "tooltip": "CYAI 中转站 API Key（sk-xxx）"},
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
                    MODEL_OPTIONS,
                    {"default": "doubao-seedance-2-0-260128", "tooltip": "视频生成模型"},
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
        poll_interval=10,
        max_wait=900,
        image_max_side=1024,
        image_quality=80,
        task_id_field="id|data.id|data.task_id|task_id",
        status_field="status|data.status",
        video_url_field="content.video_url|video_url|data.output|data.video_url",
        done_statuses="succeeded,completed,success,done",
        fail_statuses="failed,error,cancelled,canceled",
    ):
        api_key = (api_key or "").strip()
        prompt = (prompt or "").strip()
        if not api_key:
            raise ValueError("api_key 不能为空，请填写 CYAI 中转站的 sk-xxx")
        if not prompt:
            raise ValueError("prompt 不能为空")

        base = (base_url or "").strip().rstrip("/")
        if not base:
            base = "https://www.cyai.club"

        # 模式互斥校验
        has_frames = first_frame is not None or last_frame is not None
        has_refs = reference_images is not None
        if has_frames and has_refs:
            raise ValueError("首帧/尾帧模式与多图参考(reference_images)互斥，只能选一种")

        # 首尾帧模式要求 ratio=adaptive（火山方舟硬约束）
        if has_frames and ratio != "adaptive":
            raise ValueError("首帧/尾帧模式要求 ratio 必须设为 adaptive，请把画幅比例改为 adaptive")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # 构造 content 数组
        content = [{"type": "text", "text": prompt}]
        if first_frame is not None:
            content.append({
                "type": "image_url",
                "image_url": {"url": _image_to_data_uri(first_frame, max_side=image_max_side, quality=image_quality)},
                "role": "first_frame",
            })
        if last_frame is not None:
            content.append({
                "type": "image_url",
                "image_url": {"url": _image_to_data_uri(last_frame, max_side=image_max_side, quality=image_quality)},
                "role": "last_frame",
            })
        if reference_images is not None:
            ref = reference_images
            if ref.dim() == 3:
                ref = ref.unsqueeze(0)  # 单张 [H,W,C] -> batch [1,H,W,C]
            n = ref.shape[0]
            for i in range(n):
                single = ref[i]
                content.append({
                    "type": "image_url",
                    "image_url": {"url": _image_to_data_uri(single, max_side=image_max_side, quality=image_quality)},
                    "role": "reference_image",
                })

        body = {
            "model": model,
            "content": content,
            "resolution": resolution,
            "ratio": ratio,
            "duration": int(duration),
            "generate_audio": bool(generate_audio),
            "watermark": bool(watermark),
            "seed": int(seed),
            "return_last_frame": bool(return_last_frame),
        }

        submit_endpoint = _resolve_url(base, submit_url or DEFAULT_SUBMIT_URL)
        created = _http_json("POST", submit_endpoint, headers, body, timeout=180.0)

        task_id = _extract_task_id(created, task_id_field)
        if not task_id:
            raise RuntimeError("提交任务失败，响应里找不到任务 ID（路径 %s）：%s" % (task_id_field, created))

        poll_template = (poll_url_template or DEFAULT_POLL_URL_TEMPLATE).strip()
        poll_url = _resolve_url(base, poll_template.replace("{task_id}", str(task_id)))

        # 状态词集合（小写归一）
        done_set = {s.strip().lower() for s in str(done_statuses).split(",") if s.strip()}
        fail_set = {s.strip().lower() for s in str(fail_statuses).split(",") if s.strip()}

        deadline = time.time() + float(max_wait)

        # 进度条：按最大轮询次数显示，每轮 poll 前进一格
        total_rounds = max(1, int(float(max_wait) / max(1, float(poll_interval))) + 1)
        pbar = ProgressBar(total_rounds)

        # 先查一次，再进入 sleep+查询 的循环，避免任务秒完成还白等一个轮询间隔
        last_state = _http_json("GET", poll_url, headers, timeout=60.0)

        while time.time() < deadline:
            # 用户点击取消 / 中断时，让 ComfyUI 能真正停掉这个节点
            comfy.model_management.throw_exception_if_processing_interrupted()

            status = _extract_status(last_state, status_field).lower()

            if status in fail_set:
                err = _extract_error(last_state)
                raise RuntimeError(
                    "视频生成任务失败：%s%s" % (err or "", "" if err else " %s" % (last_state,))
                )

            if status in done_set:
                video_url = _extract_video_url(last_state, video_url_field)
                if not video_url:
                    raise RuntimeError(
                        "任务已完成，但按路径 %s 未找到视频地址：%s" % (video_url_field, last_state)
                    )
                pbar.update(total_rounds)  # 完成，进度走满
                video_bytes = _download(video_url, headers)
                video = _video_from_bytes(video_bytes)
                return (video,)

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


class CYAiSeedanceUsage:
    """查询 CYAI 中转站任务消耗（从任务列表累计 usage.completion_tokens）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {"default": "", "tooltip": "CYAI 中转站 API Key（sk-xxx）"},
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
            raise ValueError("api_key 不能为空，请填写 CYAI 中转站的 sk-xxx")

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
            "CYAI 中转站任务消耗统计",
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


class CYAiImageBatch:
    """图像合并节点：把多张图按顺序拼成一个 IMAGE batch，用于接到多图参考输入。

    提供 9 个可选输入口 image_1~image_9（Seedance 2.0 多图参考上限 9 张）；
    每个口也能接本身带 batch 的图像（如 Load Image Sequence），总数上限由
    模型（2.0=9 张 / 2.5=30 张）和 64MB 请求体共同决定。
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
