"""
clone_branch 工具 - 将 MR 源分支浅克隆到 workspace。

通过 git clone --branch <source_branch> --depth 1 将仓库源分支下载到
workspace/mr_{id}/repo/ 目录，供后续 UT 生成使用。
"""
import os
import subprocess
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from ut_agent.tools.context import ToolContext


def clone_source_branch(git_provider, output_dir: str, mr_id: int, source_branch: str) -> str:
    """
    浅克隆 MR 源分支到 workspace。

    参数:
        git_provider: pr-agent 的 git provider 实例（需要有 _prepare_clone_url_with_token）
        output_dir: workspace 根目录
        mr_id: MR 编号
        source_branch: 源分支名

    返回:
        成功: 克隆目标目录路径
        失败: 以 "ERROR:" 开头的错误信息
    """
    repo_dir = os.path.join(output_dir, f"mr_{mr_id}", "repo")

    # 如果目录已存在且有 .git，跳过重复克隆
    if os.path.isdir(os.path.join(repo_dir, ".git")):
        return f"{repo_dir} (已存在，跳过克隆)"

    os.makedirs(repo_dir, exist_ok=True)

    # 获取带 token 的 clone URL
    repo_url = git_provider.get_git_repo_url(git_provider.pr_url)
    if not repo_url:
        return "ERROR: 无法获取仓库 URL"

    clone_url = git_provider._prepare_clone_url_with_token(repo_url)
    if not clone_url:
        return f"ERROR: 无法生成带认证信息的 clone URL (repo: {repo_url})"

    # 浅克隆指定分支
    cmd = [
        "git", "clone",
        "--branch", source_branch,
        "--depth", "1",
        "--single-branch",
        "--recurse-submodules",
        "--shallow-submodules",
        clone_url,
        repo_dir,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            return f"ERROR: git clone 失败 (exit={result.returncode}): {stderr}"
    except subprocess.TimeoutExpired:
        return "ERROR: git clone 超时 (300s)"
    except Exception as e:
        return f"ERROR: git clone 异常: {e}"

    # 确保子模块已初始化（兼容 .gitmodules 存在但 clone 时未拉取的场景）
    gitmodules_path = os.path.join(repo_dir, ".gitmodules")
    if os.path.isfile(gitmodules_path):
        try:
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive", "--depth", "1"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except Exception:
            pass  # 非致命，submodule 可能已在 clone 时拉取

    return repo_dir


@tool
def clone_mr_source_branch(state: Annotated[dict, InjectedState]) -> str:
    """将 MR 的源分支浅克隆到本地 workspace。

    执行 git clone --depth 1 将源分支代码下载到 workspace/mr_{id}/repo/ 目录。
    无需参数，自动使用当前 MR 的源分支和仓库信息。
    如果目录已存在则跳过。

    返回: 克隆目标目录路径，或错误描述。
    """
    git_provider = ToolContext.git_provider
    output_dir = ToolContext.output_dir
    mr_id = state["mr_id"]
    source_branch = state["source_branch"]

    return clone_source_branch(git_provider, output_dir, mr_id, source_branch)
