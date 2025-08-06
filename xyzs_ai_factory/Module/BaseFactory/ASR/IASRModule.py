from abc import ABC, abstractmethod


class IASRModule(ABC):
    """
    ASR（Automatic Speech Recognition，自动语音识别）模块接口
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
    def transcribe(self, audio_file, **kwargs):
        """
        非流式识别
        :param audio_file: 音频文件路径
        :param kwargs:   附加参数
        :return:
        """
        pass

    @abstractmethod
    def transcribe_stream(self, audio_frame: bytes, **kwargs):
        """
        流式识别（适用于WebSocket等场景）
        :param audio_frame: 音频帧数据
        :param kwargs: 附加参数
        """
        pass
