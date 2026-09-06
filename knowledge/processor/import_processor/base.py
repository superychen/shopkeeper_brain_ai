"""
导入流程节点基类

定义统一的节点接口规范，提供通用功能

调用走向：LangGraph invoke → 节点实例(state) → BaseNode.__call__
→ 子类 process(state) → 返回状态更新给图。业务步骤写在 process 中，
所有节点共用这里的开始/结束/失败日志；失败抛出后，图不会执行下一节点。
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Optional
import logging

from knowledge.processor.import_processor.config import ImportConfig, get_config
from knowledge.processor.import_processor.exceptions import ImportProcessError

T = TypeVar("T")  # 泛型状态类型


class BaseNode(ABC):
    """
    导入流程节点基类

    所有节点类都应继承此基类，实现 process 方法。
    基类提供统一的日志、任务追踪和错误处理。

    使用示例:
        class MyNode(BaseNode):
            name = "my_node"

            def process(self, state):
                # 实现具体逻辑
                return state

        # 作为 LangGraph 节点使用
        node = MyNode()
        workflow.add_node("my_node", node)
    """

    name: str = "base_node"  # 节点名称，子类应覆盖

    def __init__(self, config: Optional[ImportConfig] = None):
        """
        初始化节点

        Args:
            config: 配置对象，默认使用全局配置
        """
        self.config = config or get_config()
        self.logger = logging.getLogger(f"import.{self.name}")

    def __call__(self, state: T) -> T:
        """
        节点执行入口

        LangGraph 调用节点时会调用此方法。
        提供统一的日志输出、任务追踪和异常处理。

        Args:
            state: 图状态字典

        Returns:
            更新后的状态字典

        Raises:
            ImportProcessError: 节点执行失败时抛出
        """
        try:
            # 1. 开始准备执行节点
            task_id = state.get("task_id", "") if isinstance(state, dict) else ""
            self.logger.info("--- %s 开始: task_id=%s ---", self.name, task_id)

            # 2. 动态调用具体子类的实现；类似 Java 的模板方法模式。
            # 返回值可能是整个状态，也可能仅含变更字段，由 LangGraph 合并。
            result = self.process(state)

            # 3. 执行节点成功
            self.logger.info(f"--- {self.name} 完成 ---")

            return result
        except ImportProcessError as e:
            # 已分类的业务异常保留原类型，方便 LangGraph 上层按错误类别处理。
            self.logger.error("%s 执行失败: task_id=%s, error_type=%s", self.name, task_id, type(e).__name__)
            raise
        except Exception as e:
            # 未分类的底层异常统一包装；cause 保存原异常供诊断，日志只记录类型。
            self.logger.error("%s 执行失败: task_id=%s, error_type=%s", self.name, task_id, type(e).__name__)
            raise ImportProcessError(
                message="节点执行失败",
                node_name=self.name,
                cause=e
            )

    @abstractmethod
    def process(self, state: T) -> T:
        """
        节点核心处理逻辑

        子类必须实现此方法。

        Args:
            state: 图状态字典

        Returns:
            更新后的状态字典
        """
        pass

    def log_step(self, step_name: str, message: str = ""):
        """
        记录步骤日志

        Args:
            step_name: 步骤名称
            message: 附加信息
        """
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)


# 配置日志格式
def setup_logging(level: int = logging.INFO):
    """
    配置导入流程日志

    CLI 启动时调用一次。import.<节点名> 日志继承此格式，使任务经过各节点的
    时间顺序可见；应用若已经配置 logging，basicConfig 默认不会覆盖它。

    Args:
        level: 日志级别
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
