from abc import ABC, abstractmethod
from typing import Dict, Type, Generic, TypeVar

T = TypeVar('T')  # 模块实例类型


class ModuleFactory(ABC, Generic[T]):
    """模块工厂类，用于创建模块实例"""

    def __init__(self):
        self._registry: Dict[str, Type[T]] = {}  # name -> 实现类
        self._instances: Dict[str, T] = {}  # key -> 实例对象

    def register(self, module_name: str, cls: Type[T]):
        """注册某类模块类型"""
        self._registry[module_name] = cls

    def create(self, module_name: str, **kwargs) -> T:
        """创建新模块实例（不缓存）"""
        if module_name not in self._registry:
            raise ValueError(f"[ModuleFactory] 未注册模块类型: {module_name}")
        instance = self._registry[module_name]()
        self._init_instance(instance, **kwargs)
        return instance

    def init_instance(self, module_key: str = None, module_name: str = None, **kwargs):
        """创建命名实例并缓存"""
        if module_key is None:
            module_key = module_name
        instance = self.create(module_name, **kwargs)
        self._instances[module_key] = instance

    def get_instance(self, module_key: str) -> T:
        """获取已初始化的命名实例"""
        if module_key not in self._instances:
            raise ValueError(f"[ModuleFactory] 未找到模块实例: {module_key}")
        return self._instances[module_key]

    def has(self, module_key: str) -> bool:
        return module_key in self._instances

    @abstractmethod
    def _init_instance(self, instance: T, **kwargs):
        """子类实现：模块初始化逻辑"""
        pass
