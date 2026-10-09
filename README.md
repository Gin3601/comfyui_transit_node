# CYAI Seedance 视频生成节点（cyai.club 中转站）

在 ComfyUI 里调用 CYAI 中转站（`https://www.cyai.club`）的 `doubao-seedance` 系列视频模型。
接口为火山方舟 Ark 原生格式（`/api/v3/contents/generations/tasks`）。

## 安装

### 方式一：ComfyUI Manager（推荐）

1. 打开 ComfyUI Manager → **Custom Nodes Manager**；
2. 搜索 `CYAI Seedance`，或用 **Install via Git URL** 粘贴本仓库地址；
3. 装完重启 ComfyUI，之后 Manager 里点 **Update** 即可更新。

### 方式二：手动

```bash
git clone https://github.com/Gin3601/comfyui-cyai-seedance.git
```

或直接把文件夹放进 `ComfyUI/custom_nodes/comfyui_transit_node/`，然后重启 ComfyUI。

节点出现在 `CYAI/Seedance` 分类下：

- **CYAI Seedance 视频生成 (cyai.club)** —— 生成视频，输出 VIDEO 对象
- **CYAI 图像合并 (多图参考)** —— 把多张参考图归一成统一尺寸的 IMAGE batch

## 输出

节点输出 **ComfyUI `VIDEO` 对象**（不是字符串路径），可直接接：
- 预览节点（画布上直接播放）
- VHS（Video Helper Suite）的保存/转码节点

## 参数

| 参数 | 说明 |
|---|---|
| `api_key` | 中转站 Key（`sk-xxx`） |
| `base_url` | 默认 `https://www.cyai.club` |
| `submit_url` | 提交任务的 POST 地址，默认 `/api/v3/contents/generations/tasks` |
| `poll_url_template` | 查询任务的 GET 地址模板，默认 `/api/v3/contents/generations/tasks/{task_id}`，`{task_id}` 自动替换 |
| `model` | `doubao-seedance-2-0-260128` / `2-0-fast-260128` / `2-0-mini-260615` / `2-5-260628` |
| `prompt` | 提示词 |
| `duration` | 视频时长（秒），4-30。2.0 系列最大 10s，2.5 最大 30s |
| `resolution` | 480p / 720p / 1080p（fast 系列不支持 1080p） |
| `ratio` | 16:9 / 4:3 / 1:1 / 3:4 / 9:16 / 21:9 / adaptive |
| `generate_audio` | 是否生成声音 |
| `watermark` | 是否加 AI 水印 |
| `seed` | 随机种子（-1 = 随机） |
| `return_last_frame` | 是否返回最后一帧 |

> `submit_url` / `poll_url_template` 支持相对路径（拼在 `base_url` 后）或完整 URL。
> 中转站换接口地址时，直接在节点上改这两个字段即可，无需改代码重装。

可选输入（IMAGE）：

| 输入口 | 说明 |
|---|---|
| `first_frame` | 首帧图（图生视频，取 batch 第一张） |
| `last_frame` | 尾帧图（首尾帧模式） |
| `reference_images` | 多张参考图（建议 1-4 张），与首尾帧互斥 |

## 三种玩法

| 玩法 | 接法 | 说明 |
|---|---|---|
| 文生视频 | 三个图口都不接 | 纯文字生成 |
| 图生视频 / 首尾帧 | 接 `first_frame`（+ 可选 `last_frame`） | **ratio 必须设 adaptive** |
| 多图参考 | 接 `reference_images`（多张） | 融合多图生成 |

## 备注

- 视频 URL 为火山 TOS 临时签名链接，**24 小时有效**，节点会自动下载并包装成 VIDEO 对象。
- 任务记录仅可查询最近 7 天。
- 消耗计费依据为 `usage.completion_tokens`。
- **余额无法通过该 key 经 API 查询**（`/api/user/self` 返回 invalid access token），请到 cyai.club 网页后台查看剩余额度。
- 仅使用 ComfyUI 自带依赖（`torch`、`PIL`、`numpy`、`comfy_api` 和标准库 `urllib`），无需额外安装包。
