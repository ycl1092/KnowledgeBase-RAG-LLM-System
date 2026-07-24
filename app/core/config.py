"""
YAML 配置加载器

从 config/rag.yaml 加载配置，支持环境变量覆盖。
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """应用配置，从 YAML + 环境变量加载"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        # 找到项目根目录
        self.ROOT_DIR = Path(__file__).resolve().parent.parent.parent
        config_path = self.ROOT_DIR / "config" / "rag.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw: dict = yaml.safe_load(f)

        # 按层级展开成扁平属性
        self._raw = raw
        self._flatten(raw)

        # 环境变量优先级高于 YAML
        self.LLM_API_KEY = os.getenv("LLM_API_KEY", "")
        self.LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")

    def _flatten(self, d: dict, prefix: str = ""):
        for k, v in d.items():
            key = f"{prefix}{k}".upper() if prefix else k.upper()
            if isinstance(v, dict):
                self._flatten(v, f"{key}_")
            else:
                setattr(self, key, v)

    def get(self, path: str, default: Any = None) -> Any:
        """按点号路径取值，如 get('chroma.collection_name')"""
        parts = path.split(".")
        val = self._raw
        for p in parts:
            if isinstance(val, dict):
                val = val.get(p)
            else:
                return default
        return val if val is not None else default


settings = Settings()
