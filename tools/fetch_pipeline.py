"""
fetch_pipeline 工具 - 获取 GitLab 流水线覆盖率和失败 job 日志。

在 UT Agent push 代码后，等待 CI 流水线完成，然后拉取：
1. 整体覆盖率（如果有）
2. 指定 job 的失败日志（build_release_arm64 / x86_64_ut_coverage_check）

=== 设计说明 ===

流水线运行需要时间，本工具采用轮询策略：
- 初始等待 60s（流水线通常需要排队）
- 每 30s 查询一次流水线状态
- 最大等待 20 分钟（可配置）
- 只关注 push 后触发的最新流水线

只关注以下 job:
- build_release_arm64: ARM64 构建
- x86_64_ut_coverage_check: x86 单元测试覆盖率检查
"""
import asyncio
import logging
import time
from typing import Annotated, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from ut_agent.tools.context import ToolContext

logger = logging.getLogger("ut_agent")

# 关注的 job 名称
TARGET_JOBS = ["build_release_arm64", "x86_64_ut_coverage_check"]

# 轮询参数
INITIAL_WAIT_SECONDS = 60       # 首次等待时间（让流水线开始运行）
POLL_INTERVAL_SECONDS = 30      # 轮询间隔
MAX_WAIT_SECONDS = 20 * 60      # 最大等待时间 20 分钟
JOB_LOG_TAIL_LINES = 80         # 尾部保留行数（总结区）
JOB_LOG_CONTEXT_LINES = 15      # 每个错误段落前后上下文行数

# 日志中的错误/失败模式（用于从全量日志提取关键段落）
# 注意：匹配时使用 re.IGNORECASE，所有 pattern 大小写不敏感。
# 方括号为可选（\[?\s* ... \s*\]?），有无括号均可匹配。
ERROR_PATTERNS = [
    r"\[?\s*failed\s*\]?",                 # [FAILED] / FAILED / Failed 等
    r"\[?\s*fatal\s*\]?",                  # [FATAL] / FATAL / Fatal 等
    r"failed.*test.*failed",               # CTest summary: N tests failed
    r"\berror\s*:",                        # error: / Error: / ERROR:（含 gcc/clang 小写 error:）
    r"undefined reference",                # 链接错误
    r"fatal\s+error",                      # fatal error（编译致命错误）
    r"❌.*test.*fail",                     # 自定义脚本失败标记
    r"package had test failures",          # colcon test failures
    r"assert.*fail",                       # assertion 失败（assert failed / ASSERT FAIL 等）
    r"Segmentation fault|SIGSEGV|SIGABRT", # 崩溃信号
]


def fetch_pipeline_feedback(
    commit_sha: str,
    project_id: Optional[str] = None,
) -> dict:
    """
    根据 commit SHA 获取对应流水线的覆盖率和失败 job 日志。

    参数:
        commit_sha: 触发流水线的 commit hash
        project_id: GitLab 项目路径/ID（不传则从 ToolContext 获取）

    返回:
        {
            "status": "success" | "timeout" | "error",
            "pipeline_id": int,
            "pipeline_status": str,
            "coverage": float | None,
            "failed_jobs": [
                {
                    "name": str,
                    "status": str,
                    "log_tail": str,  # 最后 N 行日志
                }
            ],
            "message": str,  # 人类可读总结
        }
    """
    git_provider = ToolContext.git_provider
    if not git_provider:
        return {"status": "error", "message": "ERROR: git_provider 未初始化"}

    gl = git_provider.gl
    proj_id = project_id or git_provider.id_project

    try:
        project = gl.projects.get(proj_id)
    except Exception as e:
        return {"status": "error", "message": f"ERROR: 获取项目失败: {e}"}

    logger.info(f"[pipeline] 等待 commit {commit_sha[:8]} 的流水线...")
    print(f"[UT Agent] 等待流水线运行，commit={commit_sha[:8]}...")

    # 初始等待，让流水线有时间被创建
    time.sleep(INITIAL_WAIT_SECONDS)

    start_time = time.time()
    pipeline = None

    while (time.time() - start_time) < MAX_WAIT_SECONDS:
        # 查找与此 commit 关联的流水线
        pipelines = project.pipelines.list(sha=commit_sha, order_by="id", sort="desc", per_page=1)
        if pipelines:
            pipeline = pipelines[0]
            # 刷新状态
            pipeline = project.pipelines.get(pipeline.id)
            status = pipeline.status

            logger.info(f"[pipeline] Pipeline #{pipeline.id} 状态: {status}")
            print(f"[UT Agent] Pipeline #{pipeline.id} 状态: {status}")

            # 终态判断
            if status in ("success", "failed", "canceled", "skipped"):
                break

        # 未完成，继续等待
        time.sleep(POLL_INTERVAL_SECONDS)

    # 超时检查
    if pipeline is None:
        return {
            "status": "timeout",
            "pipeline_id": None,
            "pipeline_status": None,
            "coverage": None,
            "failed_jobs": [],
            "message": f"超时 ({MAX_WAIT_SECONDS}s): 未找到 commit {commit_sha[:8]} 对应的流水线",
        }

    # 刷新 pipeline 最终状态
    pipeline = project.pipelines.get(pipeline.id)
    if pipeline.status not in ("success", "failed", "canceled", "skipped"):
        return {
            "status": "timeout",
            "pipeline_id": pipeline.id,
            "pipeline_status": pipeline.status,
            "coverage": getattr(pipeline, "coverage", None),
            "failed_jobs": [],
            "message": f"超时 ({MAX_WAIT_SECONDS}s): Pipeline #{pipeline.id} 仍在运行 ({pipeline.status})",
        }

    # 获取覆盖率
    coverage = None
    raw_coverage = getattr(pipeline, "coverage", None)
    if raw_coverage is not None:
        try:
            coverage = float(raw_coverage)
        except (TypeError, ValueError):
            coverage = None

    # 收集 pipeline 及其下游（父子 pipeline）的所有 job
    # GitLab 的 parent-child pipeline 中，父 pipeline 只有 trigger bridges，
    # 真正的 job 在 downstream pipeline 里，必须递归展开。
    def _collect_all_pipelines(p, depth=0, visited=None):
        if visited is None:
            visited = set()
        if p.id in visited or depth > 3:
            return []
        visited.add(p.id)
        result = [p]
        try:
            bridges = p.bridges.list(get_all=True, per_page=100)
        except Exception as e:
            logger.warning(f"[pipeline] 获取 bridges 失败 (pipeline #{p.id}): {e}")
            return result
        for b in bridges:
            ds = getattr(b, "downstream_pipeline", None)
            if not ds:
                continue
            ds_id = ds.get("id") if isinstance(ds, dict) else getattr(ds, "id", None)
            if not ds_id or ds_id in visited:
                continue
            try:
                ds_pipeline = project.pipelines.get(ds_id)
                logger.info(f"[pipeline] 发现 downstream pipeline #{ds_id} (来自 bridge {b.name})")
                result.extend(_collect_all_pipelines(ds_pipeline, depth + 1, visited))
            except Exception as e:
                logger.warning(f"[pipeline] 获取 downstream pipeline #{ds_id} 失败: {e}")
        return result

    all_pipelines = _collect_all_pipelines(pipeline)
    if len(all_pipelines) > 1:
        logger.info(f"[pipeline] 共收集到 {len(all_pipelines)} 个相关 pipeline (含 downstream): "
                    f"{[p.id for p in all_pipelines]}")

    # 等待所有 pipeline 的 job 达到终态（pipeline 可能先于部分 job 进入终态）
    JOB_TERMINAL_STATES = {"success", "failed", "canceled", "skipped", "manual"}
    job_wait_start = time.time()
    JOB_WAIT_MAX = 300  # 最多再等 5 分钟让 job 结束
    while (time.time() - job_wait_start) < JOB_WAIT_MAX:
        all_terminal = True
        for p in all_pipelines:
            jobs_list = p.jobs.list(per_page=100, get_all=True)
            for job in jobs_list:
                if job.status not in JOB_TERMINAL_STATES:
                    all_terminal = False
                    logger.info(f"[pipeline] 等待 job 结束: {job.name} (status={job.status}, pipeline=#{p.id})")
                    break
            if not all_terminal:
                break
        if all_terminal:
            break
        time.sleep(POLL_INTERVAL_SECONDS)

    # 获取目标 job 的状态和失败日志（聚合所有相关 pipeline）
    failed_jobs = []
    coverage_from_job = None   # 从 x86_64_ut_coverage_check 日志提取的覆盖率
    coverage_threshold = None  # 从日志提取的阈值
    ut_coverage_job_id = None  # x86_64_ut_coverage_check 的 job id（无论 success/failed 都记录）
    all_jobs_info = []  # 记录所有 job 用于诊断
    jobs = []
    for p in all_pipelines:
        jobs.extend(p.jobs.list(per_page=100, get_all=True))
    logger.info(f"[pipeline] 获取到 {len(jobs)} 个 job (跨 {len(all_pipelines)} 个 pipeline)")
    for job in jobs:
        all_jobs_info.append(f"{job.name}({job.status})")
        # 模糊匹配目标 job（大小写不敏感子串匹配）
        matched_target = None
        job_name_lower = job.name.lower()
        for target in TARGET_JOBS:
            if target.lower() in job_name_lower:
                matched_target = target
                break

        if matched_target is None:
            # 非目标 job：仅标记名字与状态，不拉日志、不参与失败判定
            if job.status == "failed":
                failed_jobs.append({
                    "name": job.name,
                    "status": job.status,
                    "log_tail": "",
                    "is_target": False,
                })
                logger.info(f"[pipeline] 非目标 job 失败 (仅标记，不参与判定): {job.name}")
            continue

        if job.status == "failed":
            # 获取 job 日志
            log_tail = _get_job_log_tail(project, job.id, JOB_LOG_TAIL_LINES)
            failed_jobs.append({
                "name": job.name,
                "status": job.status,
                "log_tail": log_tail,
                "is_target": True,
            })
            logger.info(f"[pipeline] 目标 job 失败: {job.name} (id={job.id})")
            if "x86_64_ut_coverage_check" in job_name_lower:
                ut_coverage_job_id = job.id
        elif job.status == "success" and "x86_64_ut_coverage_check" in job_name_lower:
            ut_coverage_job_id = job.id
            # job 成功时从日志提取覆盖率数据
            cov_info = _extract_coverage_from_job(project, job.id)
            if cov_info:
                coverage_from_job = cov_info.get("coverage")
                coverage_threshold = cov_info.get("threshold")
                logger.info(f"[pipeline] 覆盖率: {coverage_from_job}%, 阈值: {coverage_threshold}%")
        elif job.status != "success":
            # canceled / skipped 等非成功状态也记录，并尝试提取日志
            log_tail = ""
            if job.status in ("canceled", "failed"):
                log_tail = _get_job_log_tail(project, job.id, JOB_LOG_TAIL_LINES)
            failed_jobs.append({
                "name": job.name,
                "status": job.status,
                "log_tail": log_tail,
                "is_target": True,
            })
            logger.info(f"[pipeline] 目标 job 异常: {job.name} (status={job.status}, id={job.id})")

    # 打印所有 job 信息用于诊断
    logger.info(f"[pipeline] 所有 job: {', '.join(all_jobs_info)}")
    if pipeline.status == "failed" and not failed_jobs:
        logger.warning(f"[pipeline] 流水线失败但未匹配到失败 job! 所有 job 状态: {all_jobs_info}")

    # 最终覆盖率：优先用从 job 日志提取的，其次用 pipeline 级别的
    final_coverage = coverage_from_job if coverage_from_job is not None else coverage

    # 构建总结消息
    message = _build_summary(pipeline, final_coverage, failed_jobs)
    print(f"[UT Agent] Pipeline #{pipeline.id} 完成: {pipeline.status}, 覆盖率={final_coverage}%")

    return {
        "status": "success",
        "pipeline_id": pipeline.id,
        "pipeline_status": pipeline.status,
        "coverage": final_coverage,
        "coverage_threshold": coverage_threshold,
        "ut_coverage_job_id": ut_coverage_job_id,
        "failed_jobs": failed_jobs,
        "message": message,
    }


def _extract_coverage_from_job(project, job_id: int) -> Optional[dict]:
    """从 x86_64_ut_coverage_check job 日志中提取覆盖率和阈值。

    匹配格式:
        Coverage: 0.00%
        Threshold: 80.0%
        Total changed lines: 60
        Covered changed lines: 0
    """
    import re

    try:
        job = project.jobs.get(job_id)
        trace = job.trace()
        if isinstance(trace, bytes):
            trace = trace.decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"[pipeline] 获取 job {job_id} 日志失败: {e}")
        return None

    result = {}

    # 提取 Coverage: XX.XX%
    cov_match = re.search(r"Coverage:\s*([\d.]+)%", trace)
    if cov_match:
        try:
            result["coverage"] = float(cov_match.group(1))
        except ValueError:
            pass

    # 提取 Threshold: XX.X%
    thr_match = re.search(r"Threshold:\s*([\d.]+)%", trace)
    if thr_match:
        try:
            result["threshold"] = float(thr_match.group(1))
        except ValueError:
            pass

    # 提取行数信息
    total_match = re.search(r"Total changed lines:\s*(\d+)", trace)
    covered_match = re.search(r"Covered changed lines:\s*(\d+)", trace)
    if total_match:
        result["total_lines"] = int(total_match.group(1))
    if covered_match:
        result["covered_lines"] = int(covered_match.group(1))

    return result if result else None


def _get_job_log_tail(project, job_id: int, tail_lines: int) -> str:
    """获取 job 日志：提取错误相关段落 + 尾部总结。

    策略：
    1. 全量获取日志
    2. 用正则匹配错误行，提取每个错误行前后 context 行
    3. 拼接去重的错误段落 + 日志尾部（总结区）
    4. 总长度上限 500 行，避免 token 爆炸
    """
    import re

    try:
        job = project.jobs.get(job_id)
        trace = job.trace()
        if isinstance(trace, bytes):
            trace = trace.decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"[pipeline] 获取 job {job_id} 日志失败: {e}")
        return f"[获取日志失败: {e}]"

    lines = trace.splitlines()
    total = len(lines)

    if total == 0:
        return "[日志为空]"

    # 编译错误模式
    pattern = re.compile("|".join(ERROR_PATTERNS), re.IGNORECASE)

    # 找出所有匹配错误的行号
    error_line_indices = []
    for i, line in enumerate(lines):
        if pattern.search(line):
            error_line_indices.append(i)

    # 提取错误段落（每个错误行 ± context）
    context = JOB_LOG_CONTEXT_LINES
    error_segments = []
    used_ranges = []  # 避免重叠

    for idx in error_line_indices:
        start = max(0, idx - context)
        end = min(total, idx + context + 1)

        # 如果和上一个范围重叠，合并
        if used_ranges and start <= used_ranges[-1][1]:
            used_ranges[-1] = (used_ranges[-1][0], end)
        else:
            used_ranges.append((start, end))

    for start, end in used_ranges:
        segment = lines[start:end]
        header = f"--- [日志 L{start+1}-L{end}] ---"
        error_segments.append(header + "\n" + "\n".join(segment))

    # 尾部总结（最后 tail_lines 行）
    tail_start = max(0, total - tail_lines)
    # 避免和已提取的错误段落重复
    if used_ranges and tail_start < used_ranges[-1][1]:
        tail_start = used_ranges[-1][1]
    tail_section = lines[tail_start:] if tail_start < total else []

    # 拼接结果
    parts = []
    if error_segments:
        parts.append(f"=== 错误段落 ({len(used_ranges)} 处) ===")
        parts.extend(error_segments)
    if tail_section:
        parts.append(f"\n=== 日志尾部 (L{tail_start+1}-L{total}) ===")
        parts.append("\n".join(tail_section))

    result = "\n".join(parts)

    # 上限 500 行
    result_lines = result.splitlines()
    if len(result_lines) > 500:
        result = "\n".join(result_lines[:500]) + "\n\n... [截断，共 {} 行]".format(len(result_lines))

    return result


def _build_summary(pipeline, coverage: Optional[float], failed_jobs: list[dict]) -> str:
    """构建人类可读的流水线反馈摘要。"""
    parts = [f"Pipeline #{pipeline.id} 状态: {pipeline.status}"]

    if coverage is not None:
        parts.append(f"覆盖率: {coverage}%")
    else:
        parts.append("覆盖率: 无数据")

    if failed_jobs:
        target_failed = [fj for fj in failed_jobs if fj.get("is_target", True)]
        non_target_failed = [fj for fj in failed_jobs if not fj.get("is_target", True)]
        if target_failed:
            parts.append(f"目标 job 失败 ({len(target_failed)}):")
            for fj in target_failed:
                parts.append(f"  - {fj['name']}: {fj['status']}")
        if non_target_failed:
            parts.append(f"非目标 job 失败 ({len(non_target_failed)}, 仅标记不参与判定):")
            for fj in non_target_failed:
                parts.append(f"  - {fj['name']}: {fj['status']}")
    elif pipeline.status == "failed":
        parts.append("流水线失败，但未匹配到具体失败 job（可能 job 名称不在监控列表中）")
    else:
        parts.append("目标 job 全部通过")

    return "\n".join(parts)


async def async_fetch_pipeline_feedback(
    commit_sha: str,
    project_id: Optional[str] = None,
) -> dict:
    """异步版本 - 在 asyncio 事件循环中运行轮询（避免阻塞）。"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fetch_pipeline_feedback, commit_sha, project_id)


@tool
def check_pipeline_result(state: Annotated[dict, InjectedState]) -> str:
    """等待 CI 流水线完成并获取覆盖率和失败 job 日志。

    自动从当前状态获取 commit_sha，轮询 GitLab 流水线直到完成。
    返回结构化的流水线结果（覆盖率、失败 job 名称和错误日志）。

    返回: JSON 格式的流水线反馈结果。
    """
    import json

    commit_sha = state.get("commit_sha", "")
    if not commit_sha:
        return json.dumps({"status": "error", "message": "ERROR: 无 commit_sha"}, ensure_ascii=False)

    result = fetch_pipeline_feedback(commit_sha)
    return json.dumps(result, ensure_ascii=False)
