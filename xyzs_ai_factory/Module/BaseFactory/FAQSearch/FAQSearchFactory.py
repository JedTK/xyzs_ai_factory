from xyzs_ai_factory.Module.BaseFactory.FAQSearch.IFAQSearchModule import IFAQSearchModule
from xyzs_ai_factory.Module.BaseFactory.ModuleFactory import ModuleFactory


class FAQSearchFactory(ModuleFactory[IFAQSearchModule]):
    """
    FAQ搜索模块工厂类
    """

    def _init_instance(self, instance: IFAQSearchModule, **kwargs):
        instance.init(**kwargs)
