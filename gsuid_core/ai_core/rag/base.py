"""RAG模块基础功能 - 共享常量和工具函数"""

import os
import json
import uuid
import hashlib
import zipfile
import tempfile
import threading
from typing import Final, Union
from pathlib import Path

import httpx
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import AsyncQdrantClient
from huggingface_hub import constants as hf_constants, snapshot_download

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


# ============== 模型HF仓库映射 ==============
def _get_embedding_hf_repo(model_name: str) -> str:
    """根据embedding模型名称获取对应的HuggingFace仓库名

    特别处理：只要文件名中包含 bge-small-zh-v1.5，就使用 Qdrant/bge-small-zh-v1.5
    """
    if "bge-small-zh-v1.5" in model_name:
        return "Qdrant/bge-small-zh-v1.5"
    return model_name


EMBEDDING_HF_REPO: Final[str] = _get_embedding_hf_repo(EMBEDDING_MODEL_NAME)
SPARSE_HF_REPO: Final[str] = "Qdrant/bm25"
RERANKER_HF_REPO: Final[str] = RERANKER_MODEL_NAME  # BAAI/bge-reranker-base


# ============== 配置开关（动态读取，避免模块加载时配置文件不存在导致默认值错误） ==============
def is_enable_ai() -> bool:
    return ai_config.get_config("enable").data


def is_enable_rerank() -> bool:
    return ai_config.get_config("enable_rerank").data


def _get_hf_endpoint() -> str:
    """获取HuggingFace服务器地址"""
    return ai_config.get_config("hf_endpoint").data


def _format_size(size_bytes: int) -> str:
    """将字节数格式化为人类可读的大小"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


async def _download_and_extract_zip(base_url: str, tag: str, zip_name: str, target_dir: Path) -> bool:
    """从资源库下载zip文件并解压到目标目录（流式下载，带进度日志）

    Args:
        base_url: 资源库基础URL
        tag: 资源站标签
        zip_name: zip文件名（不含扩展名），如 "models_cache"
        target_dir: 解压目标目录

    Returns:
        True 表示成功下载并解压，False 表示失败
    """
    zip_url = f"{base_url}/ai_core/{zip_name}.zip"
    logger.info(f"🧠 [RAG] 尝试从资源库下载 {zip_name}.zip: {tag} {zip_url}")

    tmp_path = None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            async with client.stream("GET", zip_url) as response:
                if response.status_code != 200:
                    logger.warning(f"🧠 [RAG] 资源库下载 {zip_name}.zip 失败，HTTP状态码: {response.status_code}")
                    return False

                total_size = int(response.headers.get("content-length", 0))
                if total_size > 0:
                    logger.info(f"🧠 [RAG] {zip_name}.zip 文件大小: {_format_size(total_size)}")

                # 流式写入临时文件
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=".zip")
                downloaded = 0
                last_log_bytes = 0
                log_interval = 5 * 1024 * 1024  # 每5MB打印一次进度

                with os.fdopen(tmp_fd, "wb") as tmp_file:
                    async for chunk in response.aiter_bytes(chunk_size=65536):  # type: ignore
                        if chunk:
                            tmp_file.write(chunk)
                            downloaded += len(chunk)

                            # 定期打印下载进度
                            if downloaded - last_log_bytes >= log_interval:
                                if total_size > 0:
                                    progress = downloaded / total_size * 100
                                    logger.info(
                                        f"🧠 [RAG] {zip_name}.zip 下载进度: "
                                        f"{_format_size(downloaded)} / {_format_size(total_size)} ({progress:.1f}%)"
                                    )
                                else:
                                    logger.info(f"🧠 [RAG] {zip_name}.zip 已下载: {_format_size(downloaded)}")
                                last_log_bytes = downloaded

                if downloaded == 0:
                    logger.warning(f"🧠 [RAG] 资源库下载 {zip_name}.zip 失败，内容为空")
                    return False

                logger.info(f"🧠 [RAG] {zip_name}.zip 下载完成: {_format_size(downloaded)}，开始解压...")

        # 解压到父目录，因为zip内部已包含同名文件夹（如 models_cache/models_cache）
        parent_dir = target_dir.parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(tmp_path, "r") as zf:
            zf.extractall(parent_dir)

        logger.success(f"🧠 [RAG] 资源库 {zip_name}.zip 解压完成: {tag} -> {target_dir}")
        return True

    except Exception as e:
        logger.warning(f"🧠 [RAG] 资源库下载 {zip_name}.zip 失败: {e}")
        return False
    finally:
        # 清理临时文件
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


async def _try_download_from_resource_lib() -> bool:
    """尝试从资源库下载模型缓存zip包

    Returns:
        True 表示成功，False 表示失败
    """
    from gsuid_core.utils.download_resource.download_core import check_speed

    try:
        tag, base_url = await check_speed()
        if not base_url:
            logger.warning("🧠 [RAG] 资源库测速失败，无法获取可用资源站")
            return False
    except Exception as e:
        logger.warning(f"🧠 [RAG] 资源库测速异常: {e}")
        return False

    # 下载 models_cache.zip
    models_ok = await _download_and_extract_zip(base_url, tag, "models_cache", MODELS_CACHE)
    if not models_ok:
        return False

    return True


def _get_dir_size(path: Path) -> int:
    """递归计算目录总大小（字节）"""
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except OSError:
        pass
    return total


def _is_models_cache_valid() -> bool:
    """检查模型缓存是否已存在且有效

    通过检查 models--Qdrant--bge-small-zh-v1.5 文件夹是否存在且大小超过 88MB 来判断。
    """
    embedding_model_dir = MODELS_CACHE / "models--Qdrant--bge-small-zh-v1.5"
    if not embedding_model_dir.is_dir():
        return False

    dir_size = _get_dir_size(embedding_model_dir)
    min_size = 88 * 1024 * 1024  # 88MB
    if dir_size < min_size:
        logger.info(f"🧠 [RAG] Embedding模型缓存目录存在但不完整: {_format_size(dir_size)} < {_format_size(min_size)}")
        return False

    logger.info(f"🧠 [RAG] 模型缓存已存在，大小: {_format_size(dir_size)}，跳过下载")
    return True


async def pre_download_models():
    """提前下载所有模型到缓存目录

    优先从资源库下载zip包并解压，如果失败则回退到HuggingFace下载。

    下载三个模型：
    1. Embedding模型: Qdrant/bge-small-zh-v1.5 -> MODELS_CACHE
    2. Sparse模型: Qdrant/bm25 -> MODELS_CACHE
    3. Reranker模型: BAAI/bge-reranker-base -> RERANK_MODELS_CACHE
    """
    if not is_enable_ai():
        return

    # 检查模型缓存是否已存在
    if _is_models_cache_valid():
        return

    # 优先尝试从资源库下载zip包
    logger.info("🧠 [RAG] 优先尝试从资源库下载模型缓存...")
    resource_ok = await _try_download_from_resource_lib()
    if resource_ok:
        logger.success("🧠 [RAG] 资源库模型缓存下载完成，跳过HuggingFace下载")
        return

    logger.info("🧠 [RAG] 资源库下载失败，回退到HuggingFace下载...")

    hf_endpoint = _get_hf_endpoint()
    # 设置HF_ENDPOINT环境变量，并同步更新huggingface_hub.constants.ENDPOINT
    # 因为huggingface_hub在模块导入时就缓存了ENDPOINT值，仅修改os.environ不会生效
    old_endpoint = os.environ.get("HF_ENDPOINT")
    old_hf_constant = getattr(hf_constants, "ENDPOINT", None)
    os.environ["HF_ENDPOINT"] = hf_endpoint
    hf_constants.ENDPOINT = hf_endpoint.rstrip("/")
    logger.info(f"🧠 [RAG] HuggingFace 端点已设置: HF_ENDPOINT={hf_constants.ENDPOINT}")

    try:
        # 下载Embedding模型
        logger.info(f"🧠 [RAG] 预下载Embedding模型: {EMBEDDING_HF_REPO}")
        snapshot_download(
            repo_id=EMBEDDING_HF_REPO,
            cache_dir=str(MODELS_CACHE),
        )
        logger.info("🧠 [RAG] Embedding模型预下载完成")

        # 下载Sparse模型
        logger.info(f"🧠 [RAG] 预下载Sparse模型: {SPARSE_HF_REPO}")
        snapshot_download(
            repo_id=SPARSE_HF_REPO,
            cache_dir=str(MODELS_CACHE),
        )
        logger.info("🧠 [RAG] Sparse模型预下载完成")

        # 下载Reranker模型（如果启用了rerank）
        if is_enable_rerank():
            logger.info(f"🧠 [RAG] 预下载Reranker模型: {RERANKER_HF_REPO}")
            snapshot_download(
                repo_id=RERANKER_HF_REPO,
                cache_dir=str(RERANK_MODELS_CACHE),
            )
            logger.info("🧠 [RAG] Reranker模型预下载完成")
    except Exception as e:
        logger.warning(f"🧠 [RAG] 模型预下载失败，将在使用时尝试加载: {e}")
    finally:
        # 恢复原来的HF_ENDPOINT和huggingface_hub常量
        if old_endpoint is not None:
            os.environ["HF_ENDPOINT"] = old_endpoint
        elif "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]
        if old_hf_constant is not None:
            hf_constants.ENDPOINT = old_hf_constant


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
                        local_files_only=True,
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
        local_files_only=True,
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
