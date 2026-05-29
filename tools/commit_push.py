"""
commit_and_push 工具 - 将生成的测试文件提交并推送到 MR 源分支。

在克隆的仓库中添加生成的测试代码文件，commit 后 push 到远端源分支。
"""
import logging
import os
import subprocess
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from ut_agent.tools.context import ToolContext

logger = logging.getLogger("ut_agent")


def commit_and_push(
    repo_dir: str,
    patch_files: list[str],
    source_branch: str,
    mr_id: int,
    author_name: str = "UT Agent",
    author_email: str = "ut-agent@noreply.local",
) -> str:
    """
    将生成的测试文件复制到仓库目录、commit 并 push 到源分支。

    参数:
        repo_dir: 克隆的仓库本地路径
        patch_files: 生成的测试代码文件绝对路径列表
        source_branch: MR 源分支名
        mr_id: MR 编号
        author_name: commit 作者名
        author_email: commit 作者邮箱

    返回:
        成功: "OK: pushed N files to {branch}"
        失败: "ERROR: ..." 错误信息
    """
    if not os.path.isdir(os.path.join(repo_dir, ".git")):
        return f"ERROR: {repo_dir} 不是有效的 git 仓库"

    if not patch_files:
        return "ERROR: 没有需要提交的文件"

    # 过滤只提交测试代码文件（排除中间计划/日志文件）
    test_files_to_add = []
    abs_repo = os.path.abspath(repo_dir)
    for fpath in patch_files:
        if not os.path.isfile(fpath):
            logger.warning(f"[upload] 文件不存在，跳过: {fpath}")
            continue
        abs_fpath = os.path.abspath(fpath)
        if not abs_fpath.startswith(abs_repo):
            logger.warning(f"[upload] 文件不在 repo 目录内，跳过: {fpath}")
            continue
        rel_path = os.path.relpath(abs_fpath, abs_repo)
        test_files_to_add.append(rel_path)

    if not test_files_to_add:
        return "ERROR: 过滤后没有可提交的测试文件"

    # 配置 git 用户信息
    _run_git(repo_dir, ["config", "user.name", author_name])
    _run_git(repo_dir, ["config", "user.email", author_email])

    # git add
    for rel_path in test_files_to_add:
        ret = _run_git(repo_dir, ["add", rel_path])
        if ret.startswith("ERROR:"):
            return ret
    logger.info(f"[upload] git add 完成: {len(test_files_to_add)} 个文件")

    # git commit
    commit_msg = f"[UT Agent] MR !{mr_id}: 自动生成单元测试\n\n添加 {len(test_files_to_add)} 个测试文件"
    ret = _run_git(repo_dir, ["commit", "-m", commit_msg])
    if ret.startswith("ERROR:"):
        # 如果没有变更需要提交（文件内容相同），不算错误
        if "nothing to commit" in ret:
            return "OK: 无新变更需要提交"
        return ret

    # 获取 commit hash 并展示，方便审查
    commit_hash = _run_git(repo_dir, ["rev-parse", "HEAD"])
    logger.info(f"[upload] git commit 完成, commit={commit_hash}")
    print(f"[UT Agent] Commit: {commit_hash}")

    # git push
    ret = _run_git(repo_dir, ["push", "origin", source_branch])
    if ret.startswith("ERROR:"):
        return ret
    logger.info(f"[upload] git push 完成: {source_branch}")

    return f"OK: pushed {len(test_files_to_add)} files to {source_branch}, commit={commit_hash}"


def _run_git(repo_dir: str, args: list[str]) -> str:
    """在 repo_dir 中执行 git 命令。"""
    cmd = ["git"] + args
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            return f"ERROR: git {args[0]} 失败 (exit={result.returncode}): {stderr}"
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return f"ERROR: git {args[0]} 超时 (120s)"
    except Exception as e:
        return f"ERROR: git {args[0]} 异常: {e}"


@tool
def commit_and_push_to_mr(state: Annotated[dict, InjectedState]) -> str:
    """将已生成的测试代码文件提交并推送到 MR 源分支。

    自动从当前状态获取仓库路径、patch 文件列表和源分支信息。
    执行 git add/commit/push 将测试代码推送到远端。

    返回: 成功/失败描述。
    """
    repo_dir = state.get("repo", "")
    if " (" in repo_dir:
        repo_dir = repo_dir.split(" (")[0]
    generated_patches = state.get("generated_patches") or []
    source_branch = state["source_branch"]
    mr_id = state["mr_id"]

    return commit_and_push(
        repo_dir=repo_dir,
        patch_files=generated_patches,
        source_branch=source_branch,
        mr_id=mr_id,
    )
