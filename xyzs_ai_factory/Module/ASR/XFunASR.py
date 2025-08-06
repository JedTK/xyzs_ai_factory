from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

from xyzs_ai_factory.Module import ModuleUtil
from xyzs_ai_factory.Module.BaseFactory.ASR import IASRModule


class XFunASR(IASRModule):
    def __init__(self):
        self.model = None

    def init(self, **kwargs):
        """
        初始化ASR模型

        可以支持的调用方式1：
            asr.init(config={
                "model": "xxx",
                "device": "cuda",
                "trust_remote_code": True,
            })

        可以支持的调用方式2：
        asr.init(
            model="xxx",
            device="cuda",
            trust_remote_code=True
        )

        :param kwargs: 参数
        """
        kwargs = ModuleUtil.normalize_model_kwargs(**kwargs)
        model_id = kwargs.pop("model", None)
        if not model_id:
            raise ValueError("必须提供[model]参数")

        self.model = AutoModel(model=model_id, disable_update=True, **kwargs)

    def close(self):
        """释放模型资源"""
        self.model = None

    def transcribe(self, audio_file, **kwargs):
        """识别整段音频（非流式）"""
        kwargs = ModuleUtil.normalize_model_kwargs(**kwargs)
        res = self.model.generate(input=audio_file, **kwargs)
        text = rich_transcription_postprocess(res[0]["text"])
        return text

    def transcribe_stream(self, audio_frame: bytes, **kwargs):
        """分帧流式识别"""



