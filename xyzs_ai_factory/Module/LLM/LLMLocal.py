from abc import abstractmethod

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

from xyzs_py.XLogs import XLogs

from xyzs_ai_factory.Module.BaseFactory.LLM import ILLMLocalModule

logger = XLogs(__name__)


class LLMLocal(ILLMLocalModule):
    """大语言模型管理：模型列表、加载模型、对话补全、后继功能待添加"""

    # 扩展默认生成参数
    DEFAULT_GEN_CONFIG = {
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "do_sample": True,
        "num_beams": 1,
        "early_stopping": False,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 0,
    }

    def __init__(self):
        self.model_path = None
        # 判断是否可以使用GPU加速
        if torch.backends.mps.is_available():
            self.device = "mps"  # 在苹果M系列芯片上，可以使用GPU加速
        elif torch.cuda.is_available():
            self.device = "cuda"  # NVIDIA显卡加速
        else:
            self.device = "cpu"  # 无法使用GPU加速
        self.model_lock = Lock()
        self.loaded_models = {}

    @abstractmethod
    def init(self, **kwargs):
        """初始化模块"""
        self.model_path = kwargs.get("model", "")
        pass

    @abstractmethod
    def completions_sync(self, **kwargs):
        """
        :param kwargs: 查询参数
        :return: 检索结果列表，每个元素是包含相关信息的字典（如文本、得分等）
        """
        pass

    # region 辅助函数
    def _prepare_inputs(self, tokenizer, prompt: str) -> torch.Tensor:
        """自动处理输入截断，返回模型输入"""
        return tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

    # endregion

    # region 模型管理
    def models(self) -> list:
        """获取所有模型列表"""
        try:
            array = [
                d
                for d in os.listdir(self.model_path)
                if os.path.isdir(os.path.join(self.model_path, d))
                   and os.path.exists(os.path.join(self.model_path, d, "config.json"))
            ]
            return array
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            return []

    def _load_model_components(self, model_path: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """加载模型组件"""
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=True,
            padding_side="left"
        )
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        if self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                sdpa=False
            )
        elif self.device == "mps":
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,  # MPS 支持 float16 或 float32
            ).to(self.device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
            ).to(self.device)

        # 进入评估模式
        model.eval()
        return tokenizer, model

    def model_load(self, model_name: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """加载特定模型"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        with self.model_lock:
            if model_name in self.loaded_models:
                return self.loaded_models[model_name]

            try:
                model_path = os.path.join(self.model_path, model_name)
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model directory not found: {model_path}")

                logger.info(f"Loading model: {model_name}")
                tokenizer, model = self._load_model_components(model_path)
                self.loaded_models[model_name] = (tokenizer, model)
                logger.info(f"Model loaded: {model_name}")
                return tokenizer, model
            except Exception as e:
                logger.error(f"Model loading failed: {model_name} - {str(e)}")
                self.loaded_models.pop(model_name, None)
                raise RuntimeError(f"Failed to load model: {model_name}") from e

    def load_all_models(self, model_names: list):
        """加载所有模型"""
        for model_name in model_names:
            self.model_load(model_name)
        return True

    # endregion

    # region 对话补全
    @abstractmethod
    def completions_sync(self,
                         model_name: str,
                         prompt: str,
                         **kwargs
                         ) -> str:
        """
        对话补全 同步模式
        :param model_name: 模型名称
        :param prompt: 提示信息
        :param kwargs: 其他参数
        :return: 补全结果
        """
        tokenizer, model = self.model_load(model_name)
        inputs = self._prepare_inputs(tokenizer, prompt)

        try:
            config = {**self.DEFAULT_GEN_CONFIG, **kwargs}

            # 处理特殊参数
            if config.get("pad_token_id") is None:
                config["pad_token_id"] = tokenizer.eos_token_id

            outputs = model.generate(
                **inputs,
                **config
            )
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise

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
        try:
            tokenizer, model = self.model_load(model_name)
            inputs = self._prepare_inputs(tokenizer, message)

            logger.info(f"completions_async : {model_name} - {message}")

            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                timeout=timeout,
                decode_kwargs={"skip_special_tokens": True}
            )

            config = {
                **self.DEFAULT_GEN_CONFIG,
                **kwargs,
                **inputs,
                "streamer": streamer,
                "pad_token_id": tokenizer.eos_token_id
            }

            # 启动生成线程
            generation_thread = Thread(target=model.generate, kwargs=config)
            generation_thread.start()

            # 流式返回结果
            for token in streamer:
                yield token

        except Exception as e:
            logger.error(f"Stream generation failed: {str(e)}")
            yield f"[ERROR] Generation failed: {str(e)}"
    # endregion
