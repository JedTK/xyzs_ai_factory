from xyzs_ai_factory.Module.BaseFactory.LLM import ILLApiMModule
from xyzs_ai_factory.Module.BaseFactory.ModuleFactory import ModuleFactory


class LLMApiFactory(ModuleFactory[ILLApiMModule]):
    """
    大语言模型模块工厂类
    """

    def _init_instance(self, instance: ILLApiMModule, **kwargs):
        instance.init(**kwargs)
