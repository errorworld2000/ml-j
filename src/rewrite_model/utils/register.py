class Registry:
    def __init__(self, name=None):
        """
        初始化注册器。

        Args:
            name (str): 注册器名称（可选，仅用于描述）。
        """
        self.name = name
        self._dict = {}  # 用于存储注册的对象

    def register(self, target=None, name=None):
        """
        注册一个对象。

        Args:
            target (callable): 要注册的对象（函数或类）。
            name (str): 注册名（可选，默认为函数/类的名称）。

        Returns:
            callable: 返回注册的对象本体。
        """
        if target is None:
            # 如果没有直接传入目标，返回装饰器模式
            return lambda t: self.register(t, name)

        if not callable(target):
            raise TypeError(
                "Only callable objects (functions or classes) can be registered."
            )

        register_name = name or target.__name__  # 如果未指定名称，使用对象名称
        if register_name in self._dict:
            print(
                f"\033[33mWARNING: '{register_name}' is already registered in '{self.name}'.\033[0m"
            )

        self._dict[register_name] = target
        return target

    def get(self, name):
        """
        获取注册的对象。

        Args:
            name (str): 注册名。

        Returns:
            callable: 注册的对象。
        """
        if name not in self._dict:
            raise KeyError(f"'{name}' is not found in '{self.name}'.")
        return self._dict[name]

    def __call__(self, target=None, name=None):
        """
        支持通过装饰器方式注册。
        """
        return self.register(target, name)

    def __contains__(self, name):
        return name in self._dict

    def __getitem__(self, name):
        return self.get(name)

    def __setitem__(self, key, value):
        self.register(value, name=key)

    def __repr__(self):
        return f"Registry(name={self.name}, items={list(self._dict.keys())})"

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()


# DATASET_REGISTRY = Registry("dataset")  # 用于注册和管理数据集。
# ARCH_REGISTRY = Registry("arch")  # 用于注册和管理模型架构。
# MODEL_REGISTRY = Registry("model")  # 用于注册具体模型实例。
# LOSS_REGISTRY = Registry("loss")  # 用于注册损失函数。
# METRIC_REGISTRY = Registry("metric")  # 用于注册评估指标。
# OPTIMIZER_REGISTRY = Registry("optimizer")  # 用于注册各种优化器。
# SCHEDULER_REGISTRY = Registry("scheduler")  # 用于注册学习率调度器。
# PREPROCESSOR_REGISTRY = Registry("preprocessor")  # 用于注册数据预处理步骤。
# CALLBACK_REGISTRY = Registry("callback")  # 用于注册回调函数。
# LOGGER_REGISTRY = Registry("logger")  # 用于注册日志记录工具。

MODELS = Registry("models")
BACKBONES = Registry("backbones")
DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")
LOSSES = Registry("losses")
OPTIMIZERS = Registry("optimizers")
