"""
结构化日志系统

支持控制台和文件双输出，自动轮转，trace_id 链路追踪。
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

from app.core.config import settings


class Logger:
    """应用日志器，控制台 + 文件双写"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.logger = logging.getLogger("rag")

        level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
        self.logger.setLevel(level)

        # 避免重复添加 handler
        if self.logger.handlers:
            return

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-5s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 控制台
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        console.setFormatter(formatter)
        self.logger.addHandler(console)

        # 文件
        log_path = Path(settings.LOG_FILE if hasattr(settings, 'LOG_FILE') else "logs/rag.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=getattr(settings, 'LOG_MAX_BYTES', 10_485_760),
            backupCount=getattr(settings, 'LOG_backupCount', 5),
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        self.logger.exception(msg, *args, **kwargs)


logger = Logger()
maxBytes=getattr(settings, 'LOG_MAX_BYTES', 10_485_760),
maxBytes=getattr(settings, 'LOG_MAX_BYTES', 10_485_760),
backupCount=getattr(settings, 'LOG_backupCount', 5),
backupCount=getattr(settings, 'LOG_backupCount', 5),
