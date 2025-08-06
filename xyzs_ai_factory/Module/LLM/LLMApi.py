import asyncio
import json
import os
import time
import uuid
from datetime import datetime
from typing import Union, List, Dict, Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from xyzs_ai_factory.entity.OpenAIConfigEntity import OpenAIConfigEntity
from xyzs_ai_factory.service.OpenAIConfigService import OpenAIConfigService
from xyzs_ai_factory.utils.OllamaAPI import OllamaAPI
from xyzs_py.XLogs import XLogs

# 日志记录器初始化，用于记录调试和错误日志
logger = XLogs(__name__)


class LLMApi:
    """
    处理与 OpenAI 和 OllamaAPI 交互的服务，支持创建聊天补全请求，并提供流式响应和静态响应。
    """

    def __init__(self):
        """
        初始化 AIAPIService，加载配置服务
        """
        # 初始化配置服务，用于获取 AI 配置信息
        self.config_service = OpenAIConfigService()

    def get_openai_client(self, **kwargs) -> OpenAI:
        """
        获取 OpenAI 客户端实例
        :param kwargs: OpenAI 配置实体对象，包含 base_url 和 api_key
        :return: OpenAI 客户端实例
        :raises HTTPException: 如果客户端初始化失败，抛出 500 错误
        """
        try:
            base_url = kwargs.get("base_url", "")
            api_key = kwargs.get("api_key", "")
            return OpenAI(base_url=base_url, api_key=api_key)
        except Exception as e:
            logger.error(f"OpenAI客户端初始化失败: {str(e)}")
            raise HTTPException(status_code=500, detail="OpenAI客户端初始化失败")

    def get_ollama_client(self, config: OpenAIConfigEntity) -> OllamaAPI:
        """
        获取 OllamaAPI 客户端实例
        :param config: 配置实体，包含 base_url 信息
        :return: OllamaAPI 客户端实例
        :raises HTTPException: 如果客户端初始化失败，抛出 500 错误
        """
        try:
            return OllamaAPI(base_url=config.base_url)
        except Exception as e:
            logger.error(f"OllamaAPI客户端初始化失败: {str(e)}")
            raise HTTPException(status_code=500, detail="OllamaAPI客户端初始化失败")

    async def _create_openai_completion(
            self,
            config: OpenAIConfigEntity,
            model: str,
            messages: List[Dict[str, str]],
            stream: bool = False,
            options: Dict[str, Any] = None
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        """
        使用 OpenAI API 创建聊天补全请求
        :param config: OpenAI 配置实体对象
        :param model: 使用的模型名称
        :param messages: 用户与 AI 之间的消息列表
        :param stream: 是否启用流式响应
        :param options: 其他可选的配置选项
        :return: OpenAI 补全响应（如果 stream=False，则是完整的 ChatCompletion，否则是流式响应）
        :raises RuntimeError: 如果调用失败，抛出运行时异常
        """
        client = self.get_openai_client(config)
        try:
            # 异步调用 OpenAI API，使用 asyncio.to_thread 来避免阻塞
            return await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
                messages=messages,
                stream=stream,
                **{k: v for k, v in options.items() if v is not None}  # 仅传递非None值
            )
        except Exception as e:
            logger.error(f"OpenAI API调用失败: {str(e)}")
            raise RuntimeError("OpenAI API调用失败")

    async def _create_ollama_completion(
            self,
            config: OpenAIConfigEntity,
            model: str,
            messages: List[Dict[str, str]],
            stream: bool = False,
            options: Dict[str, Any] = None
    ) -> Union[Any, Any]:
        """
        使用 OllamaAPI 创建聊天补全请求
        :param config: Ollama 配置实体对象
        :param model: 使用的模型名称
        :param messages: 用户与 AI 之间的消息列表
        :param stream: 是否启用流式响应
        :param options: 其他可选的配置选项
        :return: OllamaAPI 响应结果（如果 stream=False，则是完整响应，否则是流式响应）
        :raises RuntimeError: 如果调用失败，抛出运行时异常
        """
        client = self.get_ollama_client(config)
        try:
            # 异步调用 Ollama API
            return await asyncio.to_thread(
                client.chat,
                model=model,
                messages=messages,
                stream=stream,
                **{k: v for k, v in options.items() if v is not None}  # 仅传递非None值
            )
        except Exception as e:
            logger.error(f"OllamaAPI API调用失败: {str(e)}")
            raise RuntimeError("OllamaAPI API调用失败")

    async def create_completion(
            self,
            config_code: str,
            model: str,
            messages: List[Dict[str, str]],
            stream: bool = False,
            options: Dict[str, Any] = None
    ):
        """
        根据配置创建聊天补全请求，支持 OpenAI 和 OllamaAPI
        :param config_code: 配置代码，用于查找对应的 API 配置
        :param model: 使用的模型名称
        :param messages: 用户与 AI 之间的消息列表
        :param stream: 是否启用流式响应
        :param options: 其他可选的配置选项
        :return: 对应服务的响应数据（流式或静态）
        :raises HTTPException: 如果配置获取失败或类型不支持，抛出错误
        """
        # 获取并验证配置
        try:
            config = self.config_service.get_config(config_code)
            if not config or not all([config.base_url, config.type_code]):
                raise ValueError("配置无效或缺失必要字段")
        except Exception as e:
            logger.error(f"获取配置失败: {str(e)}")
            raise HTTPException(status_code=500, detail="获取配置失败")

        logger.info(
            f"AI API请求: {config.base_url} model={model},stream={stream}, messages={messages}, options={options}")

        # 根据 type_code 选择服务
        if config.type_code == 'OpenAI':
            _response = await self._create_openai_completion(config, model, messages, stream, options)
            if stream:
                return self.__stream_openai(_response)
            else:
                content = _response.choices[0].message.content
                return self.generate_non_stream_response(content, model)

        elif config.type_code == 'OllamaAPI':
            _response = await self._create_ollama_completion(config, model, messages, stream, options)
            if stream:
                return self.__stream_ollama(_response)
            else:
                # 提取 OllamaAPI 响应中的内容
                content = _response.get("response", "") if isinstance(_response, dict) else str(_response)
                return self.generate_non_stream_response(content, model)

        else:
            raise HTTPException(status_code=400, detail=f"不支持的type_code: {config.type_code}")

    def __stream_openai(self, stream_response: Stream[ChatCompletionChunk]) -> StreamingResponse:
        """
        将 OpenAI 的流式响应转换为 SSE 格式
        :param stream_response: OpenAI 的流式响应
        :return: 流式响应的 StreamingResponse 对象
        """

        async def stream_generator():
            # 遍历流式响应并实时生成数据
            for chunk in stream_response:
                if chunk.choices[0].delta.content:
                    data = {
                        "id": chunk.id,
                        "object": "chat.completion.chunk",
                        "created": chunk.created,
                        "model": chunk.model,
                        "choices": [{
                            "index": chunk.choices[0].index,
                            "delta": {"content": chunk.choices[0].delta.content},
                            "finish_reason": chunk.choices[0].finish_reason
                        }]
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    await asyncio.sleep(0)
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    def __stream_ollama(self, stream_response: Any) -> StreamingResponse:
        """
        将 Ollama 的流式响应转换为 SSE 格式
        :param stream_response: Ollama 的流式响应
        :return: 流式响应的 StreamingResponse 对象
        """

        async def stream_generator():
            stream_id = f"cmpl-{uuid.uuid4().hex[:8]}"
            model_name = None

            # 遍历 Ollama 的响应，逐个处理 chunk
            for chunk in stream_response:
                if not model_name and "model" in chunk:
                    model_name = chunk["model"]

                content = chunk.get("response", "") or chunk.get("message", {}).get("content", "")
                finish_reason = "stop" if chunk.get("done", False) else None
                created_at = chunk.get("created_at", int(time.time()))

                # 如果 created_at 是字符串格式，尝试解析为时间戳
                if isinstance(created_at, str):
                    try:
                        created = int(datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp())
                    except (ValueError, TypeError):
                        created = int(time.time())
                else:
                    created = created_at

                # 如果有内容或完成标志，构建流数据
                if content or finish_reason:
                    data = {
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name or "unknown",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": finish_reason
                        }]
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    await asyncio.sleep(0)
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    def generate_non_stream_response(self, content: str, model: str) -> Dict[str, Any]:
        """
        生成静态响应
        :param content: 响应内容
        :param model: 使用的模型名称
        :return: 静态响应的字典格式
        """
        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "message": {"role": "assistant", "content": content},
                "index": 0,
                "finish_reason": "stop"
            }]
        }

    def generate_stream_response(self, content: str, model: str) -> StreamingResponse:
        """
        生成模拟的流式响应，用于模拟OpenAI流式生成的效果
        :param content: 生成的响应内容
        :param model: 使用的模型名称
        :return: StreamingResponse对象，用于流式传输数据
        """

        async def stream_generator():
            # 将内容按空格分割成多个token进行模拟流式生成
            tokens = content.split()
            for index, token in enumerate(tokens):
                # 每次生成一个token并返回给前端
                data = {
                    "id": f"cmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": index,
                        "delta": {"content": token + " "},  # 模拟流式响应的token
                        "finish_reason": None  # 生成尚未结束
                    }]
                }
                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(0.05)  # 模拟延迟
            yield "data: [DONE]\n\n"  # 模拟流式生成结束

        # 返回StreamingResponse，告知前端数据是流式传输的
        return StreamingResponse(stream_generator(), media_type="text/event-stream")


class AuthService:
    """
    处理API请求的授权服务，验证API密钥
    """

    def __init__(self):
        """
        从环境变量中获取 API 密钥
        """
        self.api_key = os.getenv("API_KEY", "")

    def verify_token(self, authorization: str) -> str:
        """
        验证API请求的授权token
        :param authorization: 请求头中的 Authorization 字段
        :return: 返回 API 密钥
        :raises HTTPException: 如果验证失败，抛出 HTTPException
        """
        # 检查Authorization字段是否存在并以"Bearer "开头
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="未授权")

        # 提取API密钥并验证
        api_key = authorization.split(" ")[1]
        if api_key != self.api_key:
            raise HTTPException(status_code=401, detail="无效的API密钥")

        return api_key
