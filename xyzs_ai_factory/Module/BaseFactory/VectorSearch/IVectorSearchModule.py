from abc import ABC, abstractmethod


class IVectorSearchModule(ABC):
    """
    向量检索模块接口
    """

    @abstractmethod
    def init(self, **kwargs):
        """初始化模块"""
        pass

    @abstractmethod
    def close(self):
        """释放模型资源"""
        pass

    @abstractmethod
    def search(self, **kwargs):
        """
        使用向量检索模型进行相似度搜索
        :param kwargs: 查询参数，如 {"query": "电动车充电", "top_k": 5}
        :return: 检索结果列表，每个元素是包含相关信息的字典（如文本、得分等）
        """
        pass
