# -*- coding: utf-8 -*-
"""火山方舟素材资产接口（Action 兼容入口）独立测试脚本 —— 用于验证「真人脸误判」走素材库是否可行。

不依赖 ComfyUI，纯标准库，可直接用任何 Python 3.9+ 运行。

用法：
  set CYAI_API_KEY=sk-xxxx           # Windows cmd
  export CYAI_API_KEY=sk-xxxx        # bash
  python test_asset_flow.py list-groups
  python test_asset_flow.py create-group --name demo-group
  python test_asset_flow.py create-asset --group <GID> --name p1 --url https://.../a.jpg
  python test_asset_flow.py list-assets --group <GID>
  python test_asset_flow.py probe-video --asset <AID>     # 提交一个最小视频任务，看是否仍被真人检测拦截
  python test_asset_flow.py run-all --url https://.../a.jpg   # 建组→传素材→轮询→视频探针

说明：
  - 服务端只接受「公网可访问的 http/https URL」，不接受 base64 / 内网 / 本地路径。
  - 统一 POST {base}/api/?Action=<Action>&Version=2024-01-01，Authorization: Bearer <token>。
  - 已禁用系统代理（Clash MITM 会拦 HTTPS），必要时设 CYAI_NO_PROXY=1（默认开）。
"""

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request

ARK_VERSION = "2024-01-01"


def _base_url() -> str:
    return (os.environ.get("CYAI_BASE") or "https://www.cyai.club").rstrip("/")


def _token() -> str:
    tok = (os.environ.get("CYAI_API_KEY") or "").strip()
    if not tok:
        sys.exit("请先设置环境变量 CYAI_API_KEY（sk-xxx）")
    return tok


def _opener(no_proxy: bool = True):
    handlers = []
    if no_proxy:
        handlers.append(urllib.request.ProxyHandler({}))
    ctx = ssl.create_default_context()
    handlers.append(urllib.request.HTTPSHandler(context=ctx))
    return urllib.request.build_opener(*handlers)


def _post(action: str, payload: dict, timeout: float = 60.0) -> dict:
    url = "%s/api/?Action=%s&Version=%s" % (_base_url(), action, ARK_VERSION)
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={
        "Authorization": "Bearer %s" % _token(),
        "Content-Type": "application/json",
    })
    try:
        with _opener().open(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        body = e.read()
    except urllib.error.URLError as e:
        sys.exit("网络错误：%s（若被代理拦截，请设 CYAI_NO_PROXY=1）" % e.reason)
    try:
        return json.loads(body.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        sys.exit("响应不是 JSON：%s" % body[:500].decode("utf-8", "replace"))


def _ark(action: str, payload: dict, timeout: float = 60.0) -> dict:
    """调用并返回 Result；有 Error 则打印原始响应并退出。"""
    resp = _post(action, payload, timeout=timeout)
    print("  [raw] %s" % json.dumps(resp, ensure_ascii=False)[:600])
    err = resp.get("Error") or resp.get("error")
    if isinstance(err, dict) and err:
        sys.exit("  !! %s 失败：%s %s" % (
            action, err.get("Code") or err.get("code") or "",
            err.get("Message") or err.get("message") or ""))
    result = resp.get("Result")
    return result if isinstance(result, dict) else {}


def cmd_list_groups(args):
    print("== ListAssetGroups ==")
    r = _ark("ListAssetGroups", {
        "Filter": {}, "PageNumber": 1, "PageSize": 20, "ProjectName": "default"})
    print("  素材组数量：%s" % r.get("TotalCount"))
    for it in (r.get("Items") or []):
        print("   - %s  %s  %s" % (it.get("Id"), it.get("Name"), it.get("GroupType")))


def cmd_create_group(args):
    print("== CreateAssetGroup ==")
    r = _ark("CreateAssetGroup", {
        "Name": args.name, "GroupType": args.type, "ProjectName": "default"})
    print("  group_id = %s" % r.get("Id"))
    return r.get("Id")


def _poll_asset(asset_id: str, interval: float = 3.0, max_wait: float = 180.0):
    deadline = time.time() + max_wait
    last = None
    while time.time() < deadline:
        r = _ark("GetAsset", {"Id": asset_id, "ProjectName": "default"})
        last = r
        st = str(r.get("Status") or "").lower()
        print("    status=%s" % (r.get("Status") or "?"))
        if st == "active":
            return r
        if st == "failed":
            print("  !! 素材 Failed：%s" % json.dumps(r.get("Error") or {}, ensure_ascii=False))
            return r
        time.sleep(interval)
    print("  !! 轮询超时，最后：%s" % json.dumps(last or {}, ensure_ascii=False))
    return last


def cmd_create_asset(args):
    print("== CreateAsset ==")
    r = _ark("CreateAsset", {
        "GroupId": args.group, "Name": args.name, "URL": args.url,
        "AssetType": args.type, "ProjectName": "default"}, timeout=120.0)
    asset_id = r.get("Id")
    print("  asset_id = %s" % asset_id)
    if asset_id:
        print("  轮询 GetAsset…")
        _poll_asset(asset_id, max_wait=args.max_wait)
    return asset_id


def cmd_list_assets(args):
    print("== ListAssets ==")
    payload = {"PageNumber": 1, "PageSize": 20, "ProjectName": "default",
               "Filter": {"GroupIds": [args.group]}}
    r = _ark("ListAssets", payload)
    print("  素材数量：%s" % r.get("TotalCount"))
    for it in (r.get("Items") or []):
        print("   - %s  %s  %s" % (it.get("Id"), it.get("Status"), it.get("Name")))


def cmd_get_asset(args):
    print("== GetAsset ==")
    r = _ark("GetAsset", {"Id": args.asset, "ProjectName": "default"})
    print("  %s" % json.dumps(r, ensure_ascii=False, indent=2)[:1200])
    return r


def cmd_probe_video(args):
    """最小视频任务：只引用素材 ID，不含任何 base64 图。

    目的：看素材引用是否绕开视频任务侧的真人检测。
    """
    print("== 视频探针：提交一个引用 asset:// 的最小任务 ==")
    content = [
        {"type": "text", "text": "a person walking slowly, cinematic"},
        {"type": "image_url", "image_url": {"url": "asset://%s" % args.asset},
         "role": "reference_image"},
    ]
    body = {
        "model": args.model, "content": content, "resolution": "480p",
        "ratio": "9:16", "duration": 4, "generate_audio": False,
        "watermark": False, "seed": 0, "return_last_frame": False,
    }
    url = "%s/api/v3/contents/generations/tasks" % _base_url()
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={
        "Authorization": "Bearer %s" % _token(), "Content-Type": "application/json"})
    try:
        with _opener().open(req, timeout=120.0) as resp:
            print("  HTTP %s\n  %s" % (resp.status, resp.read().decode("utf-8", "replace")[:800]))
    except urllib.error.HTTPError as e:
        txt = e.read().decode("utf-8", "replace")
        print("  HTTP %s\n  %s" % (e.code, txt[:800]))
        if "InputImageSensitiveContentDetected" in txt:
            print("\n  >>> 结论：素材引用【仍然】被真人检测拦截 —— 路线 3 不可行。")
        else:
            print("\n  >>> 注意：错误不是真人检测，请按上面原文判断。")


def cmd_run_all(args):
    print("### 1/4 建 AIGC 素材组")
    gid = cmd_create_group(argparse.Namespace(name=args.group_name, type="AIGC"))
    print("\n### 2/4 上传素材（公网 URL）")
    aid = cmd_create_asset(argparse.Namespace(
        group=gid, name=args.asset_name, url=args.url, type="Image", max_wait=args.max_wait))
    if not aid:
        sys.exit("素材创建失败，终止")
    print("\n### 3/4 确认素材可用")
    cmd_list_assets(argparse.Namespace(group=gid))
    print("\n### 4/4 视频探针（asset:// 引用）")
    cmd_probe_video(argparse.Namespace(asset=aid, model=args.model))
    print("\n完成。group=%s asset=%s" % (gid, aid))


def main():
    p = argparse.ArgumentParser(description="火山方舟素材资产接口测试")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list-groups"); sp.set_defaults(func=cmd_list_groups)
    sp = sub.add_parser("create-group")
    sp.add_argument("--name", required=True); sp.add_argument("--type", default="AIGC")
    sp.set_defaults(func=cmd_create_group)
    sp = sub.add_parser("create-asset")
    sp.add_argument("--group", required=True); sp.add_argument("--name", required=True)
    sp.add_argument("--url", required=True); sp.add_argument("--type", default="Image")
    sp.add_argument("--max-wait", type=float, default=180.0)
    sp.set_defaults(func=cmd_create_asset)
    sp = sub.add_parser("list-assets"); sp.add_argument("--group", required=True)
    sp.set_defaults(func=cmd_list_assets)
    sp = sub.add_parser("get-asset"); sp.add_argument("--asset", required=True)
    sp.set_defaults(func=cmd_get_asset)
    sp = sub.add_parser("probe-video")
    sp.add_argument("--asset", required=True)
    sp.add_argument("--model", default="doubao-seedance-2-0-fast-260128")
    sp.set_defaults(func=cmd_probe_video)
    sp = sub.add_parser("run-all")
    sp.add_argument("--url", required=True)
    sp.add_argument("--group-name", default="probe-aigc-group")
    sp.add_argument("--asset-name", default="probe-asset-1")
    sp.add_argument("--model", default="doubao-seedance-2-0-fast-260128")
    sp.add_argument("--max-wait", type=float, default=180.0)
    sp.set_defaults(func=cmd_run_all)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
