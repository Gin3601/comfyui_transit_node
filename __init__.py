# -*- coding: utf-8 -*-
from .nodes import (
    CYAiSeedanceVideo,
    CYAiImageBatch,
    CYAiAssetGroup,
    CYAiCreateVerifySession,
    CYAiGetVerifyResult,
    CYAiCreateAsset,
)
# from .nodes import CYAiSeedanceUsage  # 消耗查询节点：余额 API 查不到，暂时禁用

NODE_CLASS_MAPPINGS = {
    "CYAI_Seedance_Video": CYAiSeedanceVideo,
    # "CYAI_Seedance_Usage": CYAiSeedanceUsage,
    "CYAI_ImageBatch": CYAiImageBatch,
    "CYAI_AssetGroup": CYAiAssetGroup,
    "CYAI_CreateVerifySession": CYAiCreateVerifySession,
    "CYAI_GetVerifyResult": CYAiGetVerifyResult,
    "CYAI_CreateAsset": CYAiCreateAsset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CYAI_Seedance_Video": "CYAI Seedance 视频生成 (cyai.club)",
    # "CYAI_Seedance_Usage": "CYAI Seedance 消耗查询 (cyai.club)",
    "CYAI_ImageBatch": "CYAI 图像合并 (多图参考)",
    "CYAI_AssetGroup": "CYAI 素材组 (创建/查询)",
    "CYAI_CreateVerifySession": "CYAI 创建真人认证会话",
    "CYAI_GetVerifyResult": "CYAI 查询认证结果 (真人素材组)",
    "CYAI_CreateAsset": "CYAI 上传素材 (自动建组/轮询激活)",
}
