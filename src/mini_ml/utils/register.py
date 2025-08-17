import logging

logger = logging.getLogger(__name__)


class Registry:
    """注册器类，用于管理和注册各种对象（如模型、数据集等）。"""

    def __init__(self, name: str):
        """
        初始化注册器。

        Args:
            name (str): 注册器名称。
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Registry name must be a non-empty string.")
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
            logger.warning(
                "'%s' is already registered in '%s'", register_name, self.name
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

    __call__ = register

    def __getitem__(self, name):
        return self.get(name)

    def __setitem__(self, key, value):
        self.register(value, name=key)

    def __contains__(self, name):
        return name in self._dict

    def __repr__(self):
        return f"Registry(name={self.name}, items={list(self._dict.keys())})"

    def keys(self):
        """
        方法返回注册器中所有注册对象的名称列表。

        Returns:
            _type_: 返回注册对象名称的列表。
        """
        return self._dict.keys()

    def values(self):
        """
        获取注册器中所有注册对象的值。

        Returns:
            _type_: 返回注册对象值的列表。
        """
        return self._dict.values()

    def items(self):
        """
        获取注册器中所有注册对象的键值对。

        Returns:
            _type_: 返回注册对象键值对的列表。
        """
        return self._dict.items()


# METRIC_REGISTRY = Registry("metric")  # 用于注册评估指标。
# PREPROCESSOR_REGISTRY = Registry("preprocessor")  # 用于注册数据预处理步骤。
# CALLBACK_REGISTRY = Registry("callback")  # 用于注册回调函数。
# LOGGER_REGISTRY = Registry("logger")  # 用于注册日志记录工具。

SCHEDULERS = Registry("schedulers")  # 用于注册学习率调度器。
MODELS = Registry("models")  # 用于注册具体模型实例。
ARCH = Registry("arch")  # 用于注册和管理模型架构。
HEADERS = Registry("headers")  # 用于注册模型头部。
BACKBONES = Registry("backbones")  # 用于注册模型骨干网络。
DATASETS = Registry("datasets")  # 用于注册数据集。
TRANSFORMS = Registry("transforms")  # 用于注册数据转换。
LOSSES = Registry("losses")  # 用于注册损失函数。
OPTIMIZERS = Registry("optimizers")  # 用于注册优化器。
