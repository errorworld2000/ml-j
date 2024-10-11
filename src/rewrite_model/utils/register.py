class Register(dict):
    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._dict = dict()  # 创建一个字典用于保存注册的可调用对象

    def register(self, target):
        def add_item(key, value):
            if key in self._dict:  # 如果 key 已经存在
                print(f"\033[34m"
                      f"WARNING: {value.__name__} 已经存在!"
                      f"\033[0m")

            # 进行注册，将 key 和 value 添加到字典中
            self[key] = value
            return value

        # 传入的 target 可调用 --> 没有给注册名 --> 传入的函数名或类名作为注册名
        if callable(target):  # key 为函数/类的名称; value 为函数/类本体
            return add_item(key=target.__name__, value=target)
        else:  # 传入的 target 不可调用 --> 抛出异常
            raise TypeError("\033[31mOnly support callable object, e.g. function or class\033[0m")
        
    def __call__(self, target):
        return self.register(target)

    def __setitem__(self, key, value):  # 将键值对添加到 _dict 字典中
        self._dict[key] = value

    def __getitem__(self, key):  # 从 _dict 字典中获取注册的可调用对象
        return self._dict[key]

    def __contains__(self, key):  # 检查给定的注册名是否存在于 _dict 字典中
        return key in self._dict

    def __str__(self):  # 返回 _dict 字典的字符串表示
        return str(self._dict)

    def keys(self):  # 返回 _dict 字典中的所有键
        return self._dict.keys()

    def values(self):  # 返回 _dict 字典中的所有值
        return self._dict.values()

    def items(self):  # 返回 _dict 字典中的所有键值对
        return self._dict.items()
    
DATASET_REGISTRY = Registry('dataset')  # 用于注册和管理数据集。
ARCH_REGISTRY = Registry('arch')        # 用于注册和管理模型架构。
MODEL_REGISTRY = Registry('model')      # 用于注册具体模型实例。
LOSS_REGISTRY = Registry('loss')        # 用于注册损失函数。
METRIC_REGISTRY = Registry('metric')    # 用于注册评估指标。
OPTIMIZER_REGISTRY = Registry('optimizer')  # 用于注册各种优化器。
SCHEDULER_REGISTRY = Registry('scheduler')  # 用于注册学习率调度器。
PREPROCESSOR_REGISTRY = Registry('preprocessor')  # 用于注册数据预处理步骤。
CALLBACK_REGISTRY = Registry('callback')  # 用于注册回调函数。
LOGGER_REGISTRY = Registry('logger')      # 用于注册日志记录工具。

