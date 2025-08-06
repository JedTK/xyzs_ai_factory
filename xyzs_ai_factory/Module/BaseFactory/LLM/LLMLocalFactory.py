from xyzs_ai_factory.Module.BaseFactory.LLM import ILLMLocalModule
from xyzs_ai_factory.Module.BaseFactory.ModuleFactory import ModuleFactory


class LLMLocalFactory(ModuleFactory[ILLMLocalModule]):
    """
    大语言模型模块工厂类
    """

    def _init_instance(self, instance: ILLMLocalModule, **kwargs):
        instance.init(**kwargs)
