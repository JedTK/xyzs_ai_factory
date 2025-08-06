from xyzs_ai_factory.Module.BaseFactory.ASR.IASRModule import IASRModule
from xyzs_ai_factory.Module.BaseFactory.ModuleFactory import ModuleFactory


class ASRFactory(ModuleFactory[IASRModule]):
    """
    ASR（Automatic Speech Recognition，自动语音识别）模块工厂类
    """

    def _init_instance(self, instance: IASRModule, **kwargs):
        instance.init(**kwargs)
