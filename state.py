from typing import Literal, TypedDict, Optional


# tool_request 可选工具枚举
ToolName = Literal[
    "clone_mr_source_branch",  # 克隆 MR 源分支到本地
    "commit_and_push_to_mr",   # 提交并推送测试代码到源分支
    "check_pipeline_result",   # 等待并获取流水线结果
]
Task = Literal[
    "collect_mr_info",   # 收集 MR 信息
    "clone_repo",        # 克隆源分支到本地
    "analyze_diff",      # 分析 diff，识别需要测试的变更
    "generate_test_plan",  # 生成测试计划
    "generate_patch",    # 生成 UT 代码 patch
    "validate_plan",     # 校验测试计划完成度
    "fix_issues",        # 问题修复
    "upload_to_gitlab",  # 将 patch 推送/提交到 GitLab
    "check_pipeline",    # 等待并获取流水线结果
    "verify_pipeline",   # 验证流水线结果(PASS/FAIL)
    "plan_fix",          # 制定修复计划
]


class UTAgentState(TypedDict):
    """UT Agent 的 LangGraph 状态定义。"""
    # 输入：MR 信息
    pr_url: str
    title: str
    author: str
    mr_id: int
    source_branch: str
    target_branch: str
    diff_files: list[dict]  # [{filename, patch, edit_type, language}]
    # 状态时间戳，每次状态变更时更新
    timestamp: str
    # 规划与执行状态
    goal: Optional[str]              # 最终目标
    task: Optional[Task]              # 当前任务（固定阶段）
    sub_task: Optional[str]          # 当前子任务
    current_action: Optional[str]    # 当前正在执行的动作
    next_action: Optional[str]       # 下一个计划动作
    tool_request: Optional[ToolName]     # 申请调用的工具名称（固定枚举）
    # 本地资源
    repo: Optional[str]              # 克隆到本地的仓库目录路径
    clone_attempts: int              # 克隆重试次数
    # 分析结果
    diff_analysis: Optional[str]     # 分析 JSON（<=5个文件时直接存 state）
    diff_analysis_dir: Optional[str] # 分析结果落盘目录（>5个文件时按批次存文件）
    # 测试计划
    test_plan: Optional[str]         # 测试计划 JSON
    test_plan_path: Optional[str]    # 测试计划落盘路径
    # Patch 生成与校验
    generated_patches: Optional[list[str]]  # 已生成的 patch 文件路径列表
    pending_cases: Optional[str]     # 未完成的测试用例差异清单 JSON
    plan_valid: Optional[bool]       # 计划是否全部完成
    patch_iterations: int            # patch 生成迭代次数
    # 流水线反馈
    commit_sha: Optional[str]        # push 后的 commit hash
    pipeline_feedback: Optional[str] # 流水线结果 JSON（覆盖率+失败日志）
    # 验证与修复
    verification_verdict: Optional[str]  # Verifier 输出 JSON (PASS/FAIL + 详情)
    fix_plan: Optional[str]          # Planner 修复计划 JSON
    fix_iterations: int              # 修复迭代次数
    # 输出
    response: Optional[str]
