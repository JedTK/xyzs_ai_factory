from abc import ABC, abstractmethod
from typing import Dict, Any, List


class IFAQSearchModule(ABC):
    """
    FAQ问答模块接口
    """

    @abstractmethod
    def init(self, **kwargs):
        """初始化模块"""
        pass

    @abstractmethod
    def search(self, **kwargs):
        """
        :param kwargs: 参数，如{"id": "1234567890", "question": "电动车充电"}
        :return: 检索结果列表，每个元素是包含相关信息的字典（如文本、得分等）
        """
        pass
