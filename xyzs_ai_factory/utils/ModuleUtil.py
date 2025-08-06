class ModuleUtil:

    @staticmethod
    def normalize_model_kwargs(**kwargs) -> dict:
        """
        标准化模型参数：
        - 支持 'config','params','param' 参数合并
        - 最终返回一个干净的 kwargs，包含 model 之外的参数
        """

        def merge_nested(key: str):
            value = kwargs.get(key)
            if value and isinstance(value, dict):
                kwargs.update(value)
                kwargs.pop(key, None)

        merge_nested("config")
        merge_nested("params")
        merge_nested("param")

        return kwargs
