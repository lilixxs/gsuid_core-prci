"""RAG模块基础功能 - 共享常量和工具函数"""

import json
import uuid
import hashlib
import threading
from typing import Final, Union

from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import AsyncQdrantClient

from gsuid_core.logger import logger
from gsuid_core.data_store import AI_CORE_PATH
from gsuid_core.ai_core.configs.ai_config import ai_config, rerank_model_config, local_embedding_config

# ============== 向量库配置 ==============
DIMENSION: Final[int] = 512

# Embedding模型相关
EMBEDDING_MODEL_NAME: Final[str] = local_embedding_config.get_config("embedding_model_name").data
MODELS_CACHE = AI_CORE_PATH / "models_cache"
DB_PATH = AI_CORE_PATH / "local_qdrant_db"

# Reranker模型相关
RERANK_MODELS_CACHE = AI_CORE_PATH / "rerank_models_cache"
RERANKER_MODEL_NAME: Final[str] = rerank_model_config.get_config("rerank_model_name").data

# ============== Collection名称 ==============
TOOLS_COLLECTION_NAME: Final[str] = "bot_tools"
KNOWLEDGE_COLLECTION_NAME: Final[str] = "knowledge"
IMAGE_COLLECTION_NAME: Final[str] = "image"


# ============== 配置开关（动态读取，避免模块加载时配置文件不存在导致默认值错误） ==============
def is_enable_ai() -> bool:
    return ai_config.get_config("enable").data


def is_enable_rerank() -> bool:
    return ai_config.get_config("enable_rerank").data


embedding_model: "Union[TextEmbedding, None]" = None
client: "Union[AsyncQdrantClient, None]" = None
# 全局 Sparse Embedding 模型（懒加载，线程安全）
_sparse_model = None
_sparse_model_lock = threading.Lock()


def _get_sparse_model():
    """隐患三修复：添加线程锁防止并发初始化模型"""
    global _sparse_model

    if not is_enable_ai():
        return

    if _sparse_model is None:
        with _sparse_model_lock:
            # 双重检查锁定
            if _sparse_model is None:
                try:
                    _sparse_model = SparseTextEmbedding(
                        model_name="Qdrant/bm25",
                        cache_dir=str(MODELS_CACHE),
                        threads=2,
                    )
                except Exception as e:
                    logger.warning(f"🧠 [Memory] SparseTextEmbedding 初始化失败: {e}")
    return _sparse_model


def init_embedding_model():
    """初始化Embedding模型和Qdrant客户端"""
    global embedding_model, client

    if not is_enable_ai():
        return

    # 防止重复初始化，导致Qdrant文件锁冲突
    if client is not None:
        return

    embedding_model = TextEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        cache_dir=str(MODELS_CACHE),
        threads=2,
    )
    client = AsyncQdrantClient(path=str(DB_PATH))


def get_point_id(id_str: str) -> str:
    """生成向量化存储的唯一ID

    使用UUID5和DNS命名空间生成确定性的UUID，
    相同id_str始终生成相同的UUID，确保幂等性。

    Args:
        id_str: 唯一标识符字符串

    Returns:
        唯一的UUID字符串
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str))


def calculate_hash(content: dict) -> str:
    """计算内容字典的MD5哈希

    用于检测内容是否有变更，支持知识库增量更新判断。
    排序键以确保相同内容产生相同的哈希值。

    Args:
        content: 要计算哈希的内容字典

    Returns:
        MD5哈希值（32位十六进制字符串）
    """
    json_str = json.dumps(content, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(json_str.encode("utf-8")).hexdigest()
