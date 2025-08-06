from abc import ABC, abstractmethod
from typing import Dict, Any

import os
from threading import Thread, Lock
from typing import Generator, Tuple, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from xyzs_py import JsonUtil

from xyzs_py.XLogs import XLogs

logger = XLogs(__name__)


class ILLMLocalModule(ABC):
    @abstractmethod
    def init(self, config: dict):
        """初始化模块"""
        pass

    # region 对话补全
    @abstractmethod
    def completions_sync(self,
                         model_name: str,
                         prompt: str,
                         **kwargs) -> str:
        """
        对话补全 同步模式
        :param model_name: 模型名称
        :param prompt: 提示信息
        :param kwargs: 其他参数
        :return: 补全结果
        """
        pass

    @abstractmethod
    async def completions_async(
            self,
            model_name: str,
            message: str,
            timeout: Optional[int] = 600,
            **kwargs,
    ) -> Generator[str, None, None]:
        """
        对话补全 异步模式

        对话补全 同步模式
        :param model_name: 模型名称
        :param message: 提示信息
        :param timeout: 流式响应超时时间(秒)
        :param kwargs: 其他参数
        :return: 补全结果
        """
        pass
    # endregion
