# -*- coding: utf-8 -*-
from .nodes import CYAiSeedanceVideo, CYAiImageBatch
# from .nodes import CYAiSeedanceUsage  # 消耗查询节点：余额 API 查不到，暂时禁用

NODE_CLASS_MAPPINGS = {
    "CYAI_Seedance_Video": CYAiSeedanceVideo,
    # "CYAI_Seedance_Usage": CYAiSeedanceUsage,
    "CYAI_ImageBatch": CYAiImageBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CYAI_Seedance_Video": "CYAI Seedance 视频生成 (cyai.club)",
    # "CYAI_Seedance_Usage": "CYAI Seedance 消耗查询 (cyai.club)",
    "CYAI_ImageBatch": "CYAI 图像合并 (多图参考)",
}
