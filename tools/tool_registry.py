"""
工具注册表 - 收集所有 @tool 并创建 LangGraph ToolNode。

各 @tool 定义在各自的工具文件中。
运行时环境配置（git_provider, output_dir）通过 context.init_context() 注入。
图 state 中的数据（diff_files, mr_id, source_branch）通过 InjectedState 自动注入。
"""
from langgraph.prebuilt import ToolNode

from ut_agent.tools.context import init_context
from ut_agent.tools.clone_branch import clone_mr_source_branch
from ut_agent.tools.commit_push import commit_and_push_to_mr
from ut_agent.tools.fetch_pipeline import check_pipeline_result


def init_tool_context(git_provider, output_dir: str, **kwargs):
    """
    初始化工具运行时上下文。在 agent 启动时调用一次。

    参数:
        git_provider: pr-agent 的 git provider 实例
        output_dir: workspace 根目录
    """
    init_context(git_provider, output_dir)


# 工具列表
ALL_TOOLS = [
    clone_mr_source_branch,
    commit_and_push_to_mr,
    check_pipeline_result,
]


def create_tool_node() -> ToolNode:
    """创建 LangGraph ToolNode，包含所有已注册的工具。"""
    return ToolNode(ALL_TOOLS)
