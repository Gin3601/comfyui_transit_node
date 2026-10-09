# CYAI Seedance 视频生成节点（中转站）

在 ComfyUI 里调用中转站的 `doubao-seedance` 系列视频模型。
接口为火山方舟 Ark 原生格式（`/api/v3/contents/generations/tasks`）。

## 安装

### 方式一：ComfyUI Manager（推荐）

1. 打开 ComfyUI Manager → **Custom Nodes Manager**；
2. 搜索 `CYAI Seedance`，或用 **Install via Git URL** 粘贴本仓库地址；
3. 装完重启 ComfyUI，之后 Manager 里点 **Update** 即可更新。

### 方式二：手动

```bash
git clone https://github.com/Gin3601/comfyui_transit_node.git
```

或直接把文件夹放进 `ComfyUI/custom_nodes/comfyui_transit_node/`，然后重启 ComfyUI。

节点分类：

- **`CYAI/Seedance`**：
  - **CYAI Seedance 视频生成（中转站）** —— 生成视频，输出 VIDEO 对象。**写实人物**打开 `upload_images` 即可自动转素材绕开真人检测
  - **CYAI 图像合并** —— 把多张参考图归一成统一尺寸的 IMAGE batch（纯合并，不含上传）
- **`CYAI/Image`**：
  - **CYAI 图像生成（中转站）** —— 生成图像，输出 IMAGE 对象（recipe 驱动，默认火山方舟 doubao-seedream 同步接口）。输出可直接接视频节点的 `first_frame` / `reference_images`，做「文生图 → 图生视频」
- **`CYAI/Asset`**（火山方舟素材资产接口）：
  - **CYAI 素材组** —— 创建 AIGC 虚拟人像组 / 查询素材组列表 / 查询详情
  - **CYAI 创建真人认证会话** —— 输出 H5Link（本人活体认证）+ BytedToken
  - **CYAI 查询认证结果** —— 轮询拿到真人素材组 GroupId
  - **CYAI 上传素材** —— 单独上传（接本地图或 URL），输出 `asset_ids`；需要单独用时才接它

## 输出

节点输出 **ComfyUI `VIDEO` 对象**（不是字符串路径），可直接接：
- 预览节点（画布上直接播放）
- VHS（Video Helper Suite）的保存/转码节点

## 参数

| 参数 | 说明 |
|---|---|
| `api_key` | 中转站 Key（`sk-xxx`） |
| `base_url` | 中转站地址（默认已预填，接入哪家就改哪家） |
| `submit_url` | 提交任务的 POST 地址，默认 `/api/v3/contents/generations/tasks` |
| `poll_url_template` | 查询任务的 GET 地址模板，默认 `/api/v3/contents/generations/tasks/{task_id}`，`{task_id}` 自动替换 |
| `model` | 模型名（可自由输入任意模型；常见 `doubao-seedance-2-0-260128` / `2-0-fast-260128` / `2-0-mini-260615` / `2-5-260628`） |
| `prompt` | 提示词 |
| `duration` | 视频时长（秒），4-30。2.0 系列最大 10s，2.5 最大 30s |
| `resolution` | 480p / 720p / 1080p（fast 系列不支持 1080p） |
| `ratio` | 16:9 / 4:3 / 1:1 / 3:4 / 9:16 / 21:9 / adaptive |
| `generate_audio` | 是否生成声音 |
| `watermark` | 是否加 AI 水印 |
| `seed` | 随机种子（-1 = 随机） |
| `return_last_frame` | 是否返回最后一帧 |
| `recipe` | 协议配方（可选，见下「通用配方」）。填内置名 `ark_seedance`，或贴 recipe JSON / 填文件路径接入其它视频 API；留空 = 用上方显式的提交/轮询/字段路径参数 |

> `submit_url` / `poll_url_template` 支持相对路径（拼在 `base_url` 后）或完整 URL。
> 中转站换接口地址时，直接在节点上改这两个字段即可，无需改代码重装。
> 填了 `recipe` 后，配方的鉴权 / 提交地址 / 请求体 / 字段路径 / 状态词会覆盖上方显式参数（配方未提供的字段回退到显式参数）。

可选输入（IMAGE）：

| 输入口 | 说明 |
|---|---|
| `first_frame` | 首帧图（图生视频，取 batch 第一张） |
| `last_frame` | 尾帧图（首尾帧模式） |
| `reference_images` | 多张参考图（建议 1-4 张），与首尾帧互斥 |

可选输入（STRING）：

| 输入口 | 说明 |
|---|---|
| `asset_ids` | 已上传的素材 ID（逗号分隔），按 `asset://<id>` 引用。真人素材必须走这条路，直接传图会触发真人脸隐私拦截 |

## 通用配方（recipe）—— 接入任意视频 API

视频节点默认按火山方舟 Ark 原生格式工作（`提交任务 → 轮询 → 下载` 骨架）。`recipe` 字段把
「鉴权 / 提交地址 / 请求体 / 响应字段路径 / 状态词」这些**供应商差异**抽成一份 JSON 配方，
接入别家视频 API 时只需填配方，不用改代码。

### 配方字段

| 键 | 说明 |
|---|---|
| `name` / `display_name` | 配方名（任意） |
| `async` | 是否轮询。`false` = 同步接口，提交响应直接带视频地址 |
| `auth.headers` | 鉴权头模板，`{api_key}` 会被替换（如 `Bearer {api_key}`） |
| `submit.method` / `submit.url` | 提交任务的方法与地址模板（`{base_url}` 会被替换） |
| `submit.body` | 请求体 JSON 字符串模板（见下方占位符） |
| `poll_url_template` | 轮询地址模板（`{task_id}` 运行时替换） |
| `content_items` | content 数组各角色的 item 模板：`text` / `first_frame` / `last_frame` / `reference_image` / `asset` |
| `task_id_path` / `status_path` / `video_url_path` / `error_path` | 响应字段路径（点路径，`\|` 分隔多候选） |
| `done_statuses` / `fail_statuses` | 状态词数组 |

### 请求体占位符

body 模板里用占位符拼 JSON，节点按类型替换：

- **字符串**（写时加引号）：`{prompt}` `{model}` `{resolution}` `{ratio}`
- **数字**（不加引号）：`{duration}` `{seed}`
- **布尔**（不加引号）：`{generate_audio}` `{watermark}` `{return_last_frame}`
- **content 数组**：`{content}`（由 `content_items` 按实际接的图 / 素材组装）

### 示例：接入一个 OpenAI 兼容的视频接口

```json
{
  "name": "openai_video",
  "async": true,
  "auth": {"headers": {"Authorization": "Bearer {api_key}", "Content-Type": "application/json"}},
  "submit": {"method": "POST", "url": "{base_url}/v1/video/generations",
             "body": "{\"model\": {model}, \"prompt\": {prompt}, \"seconds\": {duration}}"},
  "poll_url_template": "{base_url}/v1/video/generations/{task_id}",
  "content_items": {"text": "{\"type\": \"text\", \"text\": {prompt}}"},
  "task_id_path": "id",
  "status_path": "status",
  "video_url_path": "video_url",
  "done_statuses": ["completed"],
  "fail_statuses": ["failed"]
}
```

把这段 JSON 原样贴进节点 `recipe` 字段即可。内置 `ark_seedance` 配方即当前默认行为的等价声明。

### 图像生成配方

**CYAI 图像生成** 节点复用同一套配方机制，但默认走**同步接口**（提交即返回图片），内置配方 `ark_seedream`（火山方舟 doubao-seedream）。图像配方与视频配方的差异字段：

| 键 | 说明 |
|---|---|
| `async` | 默认 `false`（同步）。少数图像接口是任务制，设 `true` 并提供 `poll_url_template` / `task_id_path` / `status_path` |
| `result_path` | 图片所在数组的字段路径，默认 `data`（`data[].url` 或 `data[].b64_json`） |
| `image_field` | 图生图参考字段名，默认 `image`（换接口时改成对方字段名，如 `init_image`） |

图像节点请求体占位符：`{prompt}` `{model}` `{size}` `{seed}` `{watermark}` `{response_format}` `{n}`。图生图时节点会自动按 `image_field` 注入参考图（单图 = 字符串，多图 = 数组）。

图像节点输出 `IMAGE`，可直接接视频节点的 `first_frame` / `reference_images`，无需转换：

```
[CYAI 图像生成] ──IMAGE──→ [CYAI Seedance 视频生成].first_frame → 图生视频
```

> 素材 / 真人认证那套（`CYAI 上传素材` 等）是火山方舟私有资产能力，不属于通用配方层，仍单独使用。

## 三种玩法

| 玩法 | 接法 | 说明 |
|---|---|---|
| 文生视频 | 三个图口都不接 | 纯文字生成 |
| 图生视频 / 首尾帧 | 接 `first_frame`（+ 可选 `last_frame`） | **ratio 必须设 adaptive** |
| 多图参考（动漫/二次元） | 接 `reference_images`（多张） | 融合多图生成 |
| **写实人物（AI 漫剧）** | 接 `reference_images` + **打开 `upload_images`** | **一键模式**，节点自动转素材绕开真人检测 |

## 写实人物（AI 漫剧）—— 一键模式

写实人脸会被上游真人检测拦（base64 ❌、公网 URL ❌，只有素材引用能过）。以前要手动接线绕，
现在**打开一个开关就行**：

```
[加载图像] ─┐
            ├→ [CYAI 图像合并] ──images──→ [CYAI Seedance 视频生成] → [保存视频]
[加载图像] ─┘                              ↑ 打开 upload_images，填 api_key
```

**只要做两步**：
1. 视频节点 **`upload_images` 打开**
2. 填 **`api_key`**

节点内部自动：把 `reference_images` / `first_frame` 收到的图 → 上传成素材 → 改用 `asset://` 引用。
**不需要接 `asset_ids` 那根线。**

其它可调项（都有默认值，一般不用管）：`image_host`（图床）、`asset_group_id` / `asset_name`（素材组与名称，留空自动）、
`upload_max_side` / `upload_quality`（上传图尺寸与质量）。

## 真人认证 / 素材流程

`doubao-seedance` 不允许把真人面容直接以 base64 传进视频任务，会报
`InputImageSensitiveContentDetected.PrivacyInformation`（`may contain real person`）。
真人素材必须走火山方舟素材资产接口的真人认证流程：

1. **CYAI 创建真人认证会话** → 输出 `H5Link` 与 `BytedToken`；把 H5Link 发给本人，在手机/浏览器完成活体认证。
2. **CYAI 查询认证结果**（输入上一步的 `BytedToken`）→ 轮询拿到真人素材组 `GroupId`。
3. **CYAI 上传素材**（输入 `GroupId` + 素材的公网 URL）→ 轮询到 `Active`，输出 `asset_id`。
4. 在**视频节点**的 `asset_ids` 填上一步的 `asset_id`（逗号分隔多个），即可引用该真人素材生成视频。

> AI 生成的**虚拟人像**（非真人）不受此限制：直接用 **CYAI 上传素材**（`group_id` 留空会自动建 AIGC 组）上传，再填 `asset_ids`，无需真人认证；也可以照旧接 `reference_images` 传 base64。

注意：

- `CreateAsset` 需要**公网 URL**（平台服务端可访问的 http/https），不支持 base64，也不支持内网/本地路径。ComfyUI 本地生成的图需先发布到一个公网可访问的地址（对象存储/CDN）再上传。
- 素材创建是异步的，节点会自动轮询 `GetAsset` 直到 `Active` 或 `Failed`。
- 真人认证的 H5 活体校验发生在 ComfyUI 之外，需本人配合操作。

## 备注

- 视频 URL 为火山 TOS 临时签名链接，**24 小时有效**，节点会自动下载并包装成 VIDEO 对象。
- 任务记录仅可查询最近 7 天。
- 消耗计费依据为 `usage.completion_tokens`。
- 余额无法通过该 key 经 API 查询，请到中转站网页后台查看剩余额度。
- 仅使用 ComfyUI 自带依赖（`torch`、`PIL`、`numpy`、`comfy_api` 和标准库 `urllib`），无需额外安装包。
