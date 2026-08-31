"""AI 客户端管理器的公共基础能力。"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TypeVar, cast


ClientT = TypeVar("ClientT")


class BaseClientManager:
    """通过双重检查锁安全地创建并缓存客户端单例。"""

    @classmethod
    def _get_or_create(
        cls,
        instance_name: str,
        lock: threading.Lock,
        factory: Callable[[], ClientT],
    ) -> ClientT:
        """返回已有实例，或通过工厂方法创建并缓存一个新实例。

        第一次检查是不加锁的快速路径，客户端已经创建后可以直接返回；
        只有首次创建时才进入锁。进入锁后必须再次检查，因为等待锁期间
        可能已有其他线程完成了实例创建。

        ```text
        第一次检查实例
              │
              ├── 已存在 ────────────────> 直接返回
              │
              └── 不存在
                    │
                    ▼
                  获取锁
                    │
                    ▼
              第二次检查实例
                    │
                    ├── 已存在 ──────────> 返回其他线程创建的实例
                    │
                    └── 不存在
                          │
                          ▼
                    factory() 创建实例
                          │
                          ▼
                    缓存实例并返回
        ```

        工厂方法抛出异常时不会写入缓存，后续调用仍可重新尝试创建。
        """
        instance = getattr(cls, instance_name, None)
        if instance is not None:
            return cast(ClientT, instance)

        with lock:
            instance = getattr(cls, instance_name, None)
            if instance is not None:
                return cast(ClientT, instance)

            instance = factory()
            setattr(cls, instance_name, instance)
            return instance
