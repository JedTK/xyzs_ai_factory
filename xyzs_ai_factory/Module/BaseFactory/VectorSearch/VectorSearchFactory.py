from xyzs_ai_factory.Module.BaseFactory.ModuleFactory import ModuleFactory
from xyzs_ai_factory.Module.BaseFactory.VectorSearch import IVectorSearchModule


class VectorSearchFactory(ModuleFactory[IVectorSearchModule]):
    """
    向量检索模块工厂类
    """

    def _init_instance(self, instance: IVectorSearchModule, **kwargs):
        instance.init(**kwargs)
