"""
这个模块定义了一个Config类，用于加载和管理应用程序的配置信息。

Config类实现了单例模式，确保在应用程序的生命周期中只有一个配置实例。
它可以从YAML格式的配置文件中读取配置数据，并提供访问和修改这些数据的接口。
"""

from typing import Optional, Dict, Any
import yaml
from .logger import setup_logger

logger = setup_logger(__name__)


class Config:
    """
    Config类用于管理应用程序的配置信息。

    该类实现了单例模式，确保在整个应用程序中只有一个Config实例。
    配置数据从YAML文件中加载，并可以通过类方法获取和设置。
    """

    _instance: Optional["Config"] = None
    _config_data: Dict[str, Any] = {}

    def __new__(cls, file_path: str = "config.yml"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_file(file_path)
        return cls._instance

    @classmethod
    def get_instance(cls, file_path: str = "config.yml") -> "Config":
        """Get the singleton instance of Config."""
        return cls(file_path)

    def _load_file(self, file_path: str) -> None:
        """Load the configuration file and update _config_data."""
        logger.info("Loading configuration file: %s", file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self._config_data = yaml.safe_load(f) or {}
                logger.info("Configuration loaded successfully.")
        except FileNotFoundError:
            logger.warning(
                "Configuration file not found: %s. Using default config.", file_path
            )
            self._config_data = {}
        except yaml.YAMLError as exc:
            logger.error("Error in YAML file: %s", exc)
            self._config_data = {}

    def reload(self, file_path: str = "config.yml") -> None:
        """Reload configuration from the specified file."""
        logger.info("Reloading configuration file: %s", file_path)
        self._load_file(file_path)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get a configuration value with a default if the key is missing."""
        return self._config_data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to configuration values."""
        try:
            return self._config_data[key]
        except KeyError:
            logger.error("Key '%s' not found in configuration.", key)
            raise

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-like setting of configuration values."""
        self._config_data[key] = value
