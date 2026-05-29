"""
工具运行时上下文 - 存放不适合放入 LangGraph state 的环境配置。

git_provider 和 output_dir 不可序列化，不放入图 state，
通过此模块级变量在 agent 启动时注入一次。
"""


class ToolContext:
    """运行时环境配置，agent 启动时初始化。"""
    git_provider = None
    output_dir: str = ""


def init_context(git_provider, output_dir: str):
    """初始化工具运行时上下文。在 agent 启动时调用一次。"""
    ToolContext.git_provider = git_provider
    ToolContext.output_dir = output_dir
