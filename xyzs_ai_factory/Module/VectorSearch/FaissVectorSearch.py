import faulthandler
import gc
import logging
from typing import List

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from xyzs_ai_factory.Module.BaseFactory.VectorSearch import IVectorSearchModule

# 启用故障处理程序，以便在发生致命错误时能够转储回溯，帮助调试
faulthandler.enable()
# 使用当前模块名初始化日志
logger = logging.getLogger(__name__)
# 设置日志记录器的最低处理级别为DEBUG
logger.setLevel(logging.DEBUG)


class FaissVectorSearch(IVectorSearchModule):
    """此类提供使用Faiss和句子嵌入模型处理文本相似性搜索的服务。文本存储需调用者自行操作

        它的功能包括：
        1. 加载预训练的句子嵌入模型生成文本嵌入。
        2. 创建并管理Faiss索引以存储和搜索文本嵌入。
        3. 支持将新的文本嵌入追加到索引中，并能执行相似度搜索。

        Attributes:
            model_path (str): 模型路径。
            device (str): 模型运行的设备（"cpu" 或 "cuda"）。
            __indexDB (faiss.Index): 用于存储和查询文本嵌入的Faiss索引。
            __model (SentenceTransformer): 用于生成文本嵌入的模型。
            __embedding_dim (int): 嵌入的维度，通常是512或768。
        """

    def __init__(self):
        self.model_path = None
        self.faiss_file = None
        self.device = "cpu"
        self.__indexDB = None  # Faiss索引数据库
        self.__model = None  # 句子嵌入模型
        self.__embedding_dim = None  # 嵌入向量的维度

    def init(self, **kwargs):
        self.model_path = kwargs.get("model_path", None)  # 模型路径
        self.faiss_file = kwargs.get("faiss_file", None)  # Faiss文件路径

        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # 根据环境选择计算设备
        self.__indexDB = None  # Faiss索引数据库
        self.__model = None  # 句子嵌入模型
        self.__embedding_dim = None  # 嵌入向量的维度

        if self.model_path:
            self.__load_model()
        if self.faiss_file:
            self.set_file(self.faiss_file)

        logger.info(f"model_path: {self.model_path} device: {self.device}")

    def close(self) -> bool:
        """释放模型和索引占用的内存资源

        该方法会删除模型和Faiss索引，并清理GPU缓存（如果使用CUDA）。
        调用此函数后，再次使用服务时需要重新加载模型和初始化索引。

        Returns:
            bool: 如果成功释放资源返回True，否则返回False。
        """
        try:
            # 释放模型资源
            if self.__model is not None:
                del self.__model
                self.__model = None
                logger.info("模型资源已释放")

            # 释放索引资源
            if self.__indexDB is not None:
                del self.__indexDB
                self.__indexDB = None
                logger.info("Faiss索引已释放")

            # 清理GPU缓存
            if self.device == "cuda":
                torch.cuda.empty_cache()
                logger.info("CUDA缓存已清理")

            # 强制垃圾回收
            gc.collect()
            return True
        except Exception as e:
            logger.error(f"释放内存资源失败: {e}")
            return False

    def search(self, **kwargs):
        """
        使用Faiss索引进行相似度搜索
        :param kwargs: 查询参数，如 {"query": "电动车充电", "top_k": 5}
        :return:
        """
        try:
            if not self.__load_model() or not self.__init_faiss():
                return None

            query = kwargs.get("query", "")
            k = kwargs.get("top_k", 3)

            # 生成查询文本的嵌入向量
            query_embed = self.__model.encode([query], normalize_embeddings=True)

            # 使用Faiss索引进行相似度搜索
            distances, indices = self.__indexDB.search(query_embed, k)

            return distances, indices
        except Exception as e:
            logger.error(f"搜索失败：{e}")
            return None

    # 以下方法为Faiss管理的实现，用于追加文本嵌入，可以用于其他模块中使用

    def __load_model(self) -> bool:
        """懒加载句子嵌入模型

        该方法在需要时加载预训练的句子嵌入模型。加载完成后，模型将被移至指定的计算设备（CPU 或 CUDA）。

        Returns:
            bool: 如果加载成功，则返回True，否则返回False。
        """
        try:
            if not self.__model:
                # 加载句子嵌入模型
                self.__model = SentenceTransformer(self.model_path, local_files_only=True)
                # 获取嵌入向量的维度（通常为512或768）
                self.__embedding_dim = self.__model.get_sentence_embedding_dimension()
            return True
        except Exception as e:
            logger.error(f"加载模型失败：{self.model_path} \r\n {e}")
        return False

    def __init_faiss(self) -> bool:
        """初始化Faiss索引

        一般情况下Faiss如果不指定faiss文件来加载到内存，则自行创建内存来进行存储索引

        Returns:
            bool: 如果索引创建成功，则返回True，否则返回False。
        """
        try:
            if not self.__indexDB:
                if self.__embedding_dim is None:
                    logger.error("嵌入维度未初始化")
                    return False

                # 如果没有现有的索引，则创建新的索引
                """
                IndexFlatIP: 如果数据量小（<10,000条）,计算余弦相似度，非常适合语义匹配，(必须)normalize_to_unit_vector=True
                IndexIVFFlat: 如果数据量中等（10,000到100,000条）, 提供速度和准确性的平衡。
                IndexIVFPQ：如果数据量大（>100,000条），压缩向量，节省内存，适用于大规模数据，同时保持合理的搜索速度。
                """
                # self.__indexDB = faiss.IndexFlatIP(self.__embedding_dim)

                # 创建基础索引（如 IndexFlatIP）
                base_index = faiss.IndexFlatIP(self.__embedding_dim)
                # 用 IndexIDMap 包裹基础索引以支持自定义 ID
                self.__indexDB = faiss.IndexIDMap(base_index)

                return True

            return True
        except Exception as e:
            logger.error(f"创建Faiss索引失败：{self.model_path} \r\n {e}")
        return False

    def append(self, texts: List[str], ids: List[int] = None) -> bool:
        """追加文本嵌入（可选带ID）

        该方法生成文本的嵌入向量，并将其追加到Faiss索引中。

        Args:
            texts (List[str]): 要追加的文本列表。
            ids (List[int], optional): 每个文本的自定义ID列表。默认使用add()自动生成。

        Returns:
            bool: 如果追加成功，则返回True，否则返回False。
        """
        try:
            if not self.__load_model() or not self.__init_faiss():
                return False

            # 生成文本嵌入向量
            embeddings = self.__model.encode(texts, normalize_to_unit_vector=True)
            embeddings = np.ascontiguousarray(embeddings.astype('float32'))

            if ids is None:
                self.__indexDB.add(embeddings)
            else:
                if len(texts) != len(ids):
                    logger.error("文本数量与ID数量不一致")
                    return False
                ids_array = np.array(ids, dtype='int64')
                self.__indexDB.add_with_ids(embeddings, ids_array)

            return True
        except Exception as e:
            logger.error(f"追加文本嵌入发生错误：{e}")
            return False

    def set_file(self, faiss_file_path: str) -> bool:
        """设置Faiss索引文件，扩展使用，或分布式时布局使用

        加载faiss文件到内存中使用，需要调用者初始化FaissService时调用

        Returns:
            bool: 如果索引创建成功，则返回True，否则返回False。
        """
        try:
            if faiss_file_path:
                # 从文件中读取现有的Faiss索引
                self.__indexDB = faiss.read_index(faiss_file_path)
                return True
            return True
        except Exception as e:
            logger.error(f"创建Faiss索引失败：{self.model_path} \r\n {e}")
        return False

    def save_file(self, save_file_path) -> bool:
        """将Faiss索引保存到文件

        该方法将创建好的Faiss索引保存到指定路径的文件中。

        Args:
            save_file_path (str): 保存索引文件的路径。

        Returns:
            bool: 如果保存成功，则返回True，否则返回False。
        """
        try:
            if not self.__load_model() or not self.__init_faiss():
                return False

            if self.__indexDB:
                # 将Faiss索引写入文件
                faiss.write_index(self.__indexDB, save_file_path)

            return True
        except Exception as e:
            logger.error(f"保存Faiss索引失败：{self.model_path} \r\n {e}")
            return False
