import json

import requests


class OllamaAPI:
    def __init__(self, base_url="http://localhost:11434", api_key=None):
        """
        初始化 OllamaAPI 类

        Args:
            base_url (str): Ollama API 的基础 URL，默认为 http://localhost:11434
            api_key (str, optional): API 密钥（Ollama 默认不需要，但为扩展性保留）
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key

    def _make_request(self, method, endpoint, data=None, stream=False):
        """
        内部方法：发送 HTTP 请求

        Args:
            method (str): HTTP 方法，如 'GET', 'POST', 'DELETE'
            endpoint (str): API 端点
            data (dict, optional): 请求数据
            stream (bool): 是否使用流式输出

        Returns:
            dict or generator: 根据 stream 参数返回响应数据或流式生成器

        Raises:
            Exception: 如果请求失败，抛出带有错误信息的异常
        """
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, stream=stream)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, data=json.dumps(data), stream=stream)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, headers=headers)
            else:
                raise ValueError(f"不支持的 HTTP 方法: {method}")

            response.raise_for_status()

            if stream:
                def stream_generator():
                    for line in response.iter_lines():
                        if line:
                            yield json.loads(line.decode('utf-8'))

                return stream_generator()
            else:
                return response.json() if response.content else {}
        except requests.exceptions.RequestException as e:
            raise Exception(f"API 请求失败: {str(e)}")

    def list_models(self):
        """
        获取本地已安装的模型列表

        Returns:
            list: 模型名称列表
        """
        return self._make_request('GET', '/api/tags').get("models", [])

    def pull_model(self, model_name):
        """
        拉取指定模型

        Args:
            model_name (str): 模型名称

        Returns:
            dict: 拉取结果
        """
        data = {"name": model_name}
        return self._make_request('POST', '/api/pull', data)

    def delete_model(self, model_name):
        """
        删除指定模型

        Args:
            model_name (str): 模型名称

        Returns:
            dict: 删除结果
        """
        return self._make_request('DELETE', f'/api/delete/{model_name}')

    def show_model_info(self, model_name):
        """
        显示模型的详细信息

        Args:
            model_name (str): 模型名称

        Returns:
            dict: 模型信息
        """
        data = {"name": model_name}
        return self._make_request('POST', '/api/show', data)

    def generate(self, model, prompt, stream=False, **kwargs):
        """
        调用生成文本的 API

        Args:
            model (str): 使用的模型名称
            prompt (str): 输入的提示文本
            stream (bool): 是否使用流式输出
            **kwargs: 其他可选参数，如 temperature, max_tokens 等

        Returns:
            str or generator: 如果 stream=False，返回完整文本；否则返回流式生成器
        """
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            **kwargs
        }
        if stream:
            return self._make_request('POST', '/api/generate', data, stream=True)
        else:
            result = self._make_request('POST', '/api/generate', data)
            return result.get("response", "")

    def chat(self, model, messages, stream=False, **kwargs):
        """
        调用聊天模式的 API

        Args:
            model (str): 使用的模型名称
            messages (list): 消息列表，格式为 [{"role": "user", "content": "文本"}, ...]
            stream (bool): 是否使用流式输出
            **kwargs: 其他可选参数，如 temperature, max_tokens 等

        Returns:
            str or generator: 如果 stream=False，返回完整回复；否则返回流式生成器
        """
        data = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        if stream:
            return self._make_request('POST', '/api/chat', data, stream=True)
        else:
            result = self._make_request('POST', '/api/chat', data)
            return result.get("message", {}).get("content", "")

    def embeddings(self, model, prompt):
        """
        生成文本的嵌入向量

        Args:
            model (str): 使用的模型名称
            prompt (str): 输入的文本

        Returns:
            list: 嵌入向量
        """
        data = {
            "model": model,
            "prompt": prompt
        }
        result = self._make_request('POST', '/api/embeddings', data)
        return result.get("embedding", [])
