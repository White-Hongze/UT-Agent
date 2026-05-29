"""
基于 LangGraph 的 UT Agent。

接收 MR 信息（标题、描述、分支、diff），生成单元测试分析评论并回发到 MR 上。
"""
import json
import logging
import os
import subprocess
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from datetime import datetime, timezone

from ut_agent.state import UTAgentState
from ut_agent.tools.context import ToolContext
from ut_agent.tools.clone_branch import clone_source_branch
from ut_agent.tools.commit_push import commit_and_push
from ut_agent.tools.fetch_pipeline import fetch_pipeline_feedback
from ut_agent.prompt import load_prompt
from ut_agent.llm import call_llm, call_llm_with_continuation
from ut_agent.config import TEST_MODE as _CFG_TEST_MODE


# workspace 默认在 ut_agent 包目录下的 workspace/ 子目录
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
UT_WORKSPACE = os.environ.get("UT_AGENT_WORKSPACE", os.path.join(_PACKAGE_DIR, "workspace"))
os.makedirs(UT_WORKSPACE, exist_ok=True)

# 日志配置：同时输出到控制台和文件
LOG_DIR = os.path.join(UT_WORKSPACE, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "ut_agent.log")

logger = logging.getLogger("ut_agent")
logger.setLevel(logging.DEBUG)


# ──────────────────────────────────────────────────────────────────────────────
# 彩色日志 Formatter（控制台）
# ──────────────────────────────────────────────────────────────────────────────

class ColorFormatter(logging.Formatter):
    """为不同节点/级别设置终端颜色的 Formatter。"""

    # ANSI 颜色码
    RESET = "\033[0m"
    BOLD = "\033[1m"
    # 级别颜色
    LEVEL_COLORS = {
        logging.DEBUG:    "\033[36m",     # 青色
        logging.INFO:     "\033[32m",     # 绿色
        logging.WARNING:  "\033[33m",     # 黄色
        logging.ERROR:    "\033[31m",     # 红色
        logging.CRITICAL: "\033[1;31m",   # 粗体红色
    }
    # 节点/阶段颜色（根据日志消息中的关键词）
    NODE_COLORS = {
        "collect_mr_info":  "\033[36m",   # 青色
        "clone_repo":       "\033[34m",   # 蓝色
        "analyze_diff":     "\033[35m",   # 紫色
        "generate_test_plan": "\033[35m", # 紫色
        "generate_patch":   "\033[33m",   # 黄色
        "validate_plan":    "\033[34m",   # 蓝色
        "upload_to_gitlab": "\033[36m",   # 青色
        "check_pipeline":   "\033[34m",   # 蓝色
        "verify_pipeline":  "\033[32m",   # 绿色
        "plan_fix":         "\033[31m",   # 红色
        "Verifier: PASS":   "\033[1;32m", # 粗体绿色
        "Verifier: FAIL":   "\033[1;31m", # 粗体红色
        "Commit:":          "\033[1;36m", # 粗体青色
    }

    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record):
        # 先用父类格式化
        msg = super().format(record)

        # 匹配节点颜色
        for keyword, color in self.NODE_COLORS.items():
            if keyword in record.getMessage():
                return f"{color}{msg}{self.RESET}"

        # 按级别着色
        color = self.LEVEL_COLORS.get(record.levelno, "")
        return f"{color}{msg}{self.RESET}"


# 控制台 handler（彩色）
# 通过环境变量 NO_COLOR=1 可禁用彩色（适配 docker logs 重定向场景）
_use_color = os.environ.get("NO_COLOR", "") == ""
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
if _use_color:
    _console_handler.setFormatter(ColorFormatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
else:
    _console_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))

# 文件 handler（追加模式）
_file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))

if not logger.handlers:
    logger.addHandler(_console_handler)
    logger.addHandler(_file_handler)


MAX_CLONE_ATTEMPTS = 3
MAX_PATCH_ITERATIONS = 3
MAX_FIX_ITERATIONS = 5          # 流水线失败后最大修复轮次
BATCH_SIZE = 5
MIN_COVERAGE_THRESHOLD = 80.0   # 覆盖率合格线 (%)

# 测试模式：跳过 LLM 环节，直接生成 dummy 文件测试推送链路
# 优先读取环境变量，其次读取 settings.toml 中 [agent].test_mode
TEST_MODE = os.environ.get("UT_AGENT_TEST_MODE", "0") == "1" or _CFG_TEST_MODE


def collect_mr_info(state: UTAgentState) -> dict:
    """收集并确认 MR 信息，设置初始状态。"""
    logger.info(f"[UT Agent] === Task: collect_mr_info ===")
    logger.info(f"[UT Agent] MR: !{state['mr_id']} | 标题: {state['title']} | 作者: {state['author']}")
    logger.info(f"[UT Agent] 分支: {state['source_branch']} -> {state['target_branch']}")
    logger.info(f"[UT Agent] 变更文件数: {len(state['diff_files'])}")
    result = {
        "task": "collect_mr_info",
        "current_action": "收集 MR 基本信息",
        "next_action": "克隆源分支",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(f"[UT Agent] State更新: {result}")
    return result


def clone_repo(state: UTAgentState) -> dict:
    """克隆 MR 源分支到本地 workspace。"""
    logger.info(f"[UT Agent] === Task: clone_repo ===")
    git_provider = ToolContext.git_provider
    output_dir = ToolContext.output_dir
    mr_id = state["mr_id"]
    source_branch = state["source_branch"]
    attempts = state.get("clone_attempts", 0) + 1
    logger.info(f"[UT Agent] 克隆分支: {source_branch} | 第{attempts}次尝试 | 输出目录: {output_dir}")

    result = clone_source_branch(git_provider, output_dir, mr_id, source_branch)

    if result.startswith("ERROR:"):
        state_update = {
            "task": "clone_repo",
            "current_action": f"克隆失败 (第{attempts}次): {result}",
            "next_action": "重试克隆" if attempts < MAX_CLONE_ATTEMPTS else "终止",
            "repo": None,
            "clone_attempts": attempts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.error(f"[UT Agent] 克隆失败: {result}")
        logger.info(f"[UT Agent] State更新: {state_update}")
        return state_update

    state_update = {
        "task": "clone_repo",
        "current_action": "克隆完成",
        "next_action": "分析 diff",
        "repo": result,
        "clone_attempts": attempts,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(f"[UT Agent] 克隆成功: {result}")
    logger.info(f"[UT Agent] State更新: {state_update}")
    return state_update


def route_after_clone(state: UTAgentState) -> str:
    """克隆后路由：成功->analyze_diff，失败且未超限->重试，超限->END。测试模式直接跳到 generate_patch。"""
    if state.get("repo"):
        if TEST_MODE:
            return "generate_patch"
        return "analyze_diff"
    if state.get("clone_attempts", 0) < MAX_CLONE_ATTEMPTS:
        return "clone_repo"
    return END


def _build_diff_content(diff_files: list[dict]) -> str:
    """将 diff_files 组装为 prompt 中的 diff 详情文本。"""
    sections = []
    for f in diff_files:
        filename = f["filename"]
        language = f.get("language", "unknown")
        patch = f.get("patch", "")
        section = f"### {filename} ({language})\n\n```diff\n{patch}\n```"
        sections.append(section)
    return "\n\n".join(sections) if sections else "无 diff 内容。"


def _build_file_list(diff_files: list[dict]) -> str:
    """构建文件列表摘要。"""
    lines = []
    for f in diff_files:
        filename = f["filename"]
        edit_type = f.get("edit_type", "UNKNOWN")
        language = f.get("language", "unknown")
        lines.append(f"- `{filename}` ({edit_type}, {language})")
    return "\n".join(lines) if lines else "无文件变更。"


async def _analyze_batch(batch: list[dict], state: UTAgentState) -> str:
    """对一批 diff_files 调用 LLM 分析，返回 JSON 字符串。"""
    system_prompt = load_prompt("analyze_diff_system")
    user_template = load_prompt("analyze_diff_user")

    user_prompt = user_template.format(
        title=state["title"],
        author=state["author"],
        mr_id=state["mr_id"],
        source_branch=state["source_branch"],
        target_branch=state["target_branch"],
        file_count=len(batch),
        file_list=_build_file_list(batch),
        diff_content=_build_diff_content(batch),
    )

    return await call_llm(system=system_prompt, user=user_prompt)


def _get_analysis_dir(mr_id: int) -> str:
    """获取分析结果落盘目录。"""
    output_dir = ToolContext.output_dir
    analysis_dir = os.path.join(output_dir, f"mr_{mr_id}", "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    return analysis_dir


async def analyze_diff(state: UTAgentState) -> dict:
    """分析 MR diff：<=5个文件直接返回，>5个文件按批次落盘。分析结果发布为 MR 评论。"""
    logger.info(f"[UT Agent] === Task: analyze_diff ===")
    all_diff_files = state["diff_files"]
    # 过滤掉已删除的文件，删除文件无需生成 UT
    diff_files = [f for f in all_diff_files if f.get("edit_type", "").upper() not in ("DELETED", "DELETE")]
    if len(diff_files) < len(all_diff_files):
        skipped = len(all_diff_files) - len(diff_files)
        logger.info(f"[UT Agent] 跳过 {skipped} 个已删除文件，不进行 UT 分析")
    mr_id = state["mr_id"]
    logger.info(f"[UT Agent] 文件数: {len(diff_files)} | 批次大小: {BATCH_SIZE}")

    if len(diff_files) <= BATCH_SIZE:
        logger.info(f"[UT Agent] 单批次模式，直接调用 LLM")
        result = await _analyze_batch(diff_files, state)
        logger.info(f"[UT Agent] LLM 分析完成，结果长度: {len(result)} chars")
        _publish_analysis_comment(result, mr_id, len(diff_files))
        logger.info(f"[UT Agent] 评论已发布到 MR")
        state_update = {
            "task": "analyze_diff",
            "current_action": f"分析完成 ({len(diff_files)} 个文件)",
            "next_action": "生成测试计划",
            "diff_analysis": result,
            "diff_analysis_dir": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(f"[UT Agent] State更新: task={state_update['task']}, action={state_update['current_action']}")
        return state_update
    else:
        analysis_dir = _get_analysis_dir(mr_id)
        total_batches = (len(diff_files) + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(f"[UT Agent] 多批次模式，共 {total_batches} 批 | 落盘目录: {analysis_dir}")
        all_results = []

        for i in range(0, len(diff_files), BATCH_SIZE):
            batch = diff_files[i:i + BATCH_SIZE]
            batch_idx = i // BATCH_SIZE + 1
            logger.info(f"[UT Agent] 处理批次 {batch_idx}/{total_batches} ({len(batch)} 个文件)")
            result = await _analyze_batch(batch, state)
            logger.info(f"[UT Agent] 批次 {batch_idx} LLM 完成，结果长度: {len(result)} chars")
            all_results.append(result)

            output_path = os.path.join(analysis_dir, f"batch_{batch_idx:03d}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)
            logger.info(f"[UT Agent] 批次 {batch_idx} 已落盘: {output_path}")

        combined = "\n\n---\n\n".join(
            f"**Batch {i+1}/{total_batches}:**\n\n```json\n{r}\n```"
            for i, r in enumerate(all_results)
        )
        _publish_analysis_comment(combined, mr_id, len(diff_files), total_batches)
        logger.info(f"[UT Agent] 评论已发布到 MR（{total_batches} 批次合并）")

        state_update = {
            "task": "analyze_diff",
            "current_action": f"分析完成 ({len(diff_files)} 个文件, {total_batches} 批次已落盘)",
            "next_action": "生成测试计划",
            "diff_analysis": None,
            "diff_analysis_dir": analysis_dir,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(f"[UT Agent] State更新: task={state_update['task']}, action={state_update['current_action']}")
        return state_update


def _repair_truncated_plan_json(json_str: str) -> str:
    """
    尝试修复被截断的测试计划 JSON。

    策略：找到最后一个完整的 } 或 ] 闭合点，从该处截断，
    然后补全缺失的 ] 和 } 使 JSON 合法。
    """
    # 逐字符回退，找到最后一个看起来完整的 test_case 结尾
    # test_case 对象以 } 结尾，数组以 ] 结尾
    # 我们找到最后一个 "}," 或 "}" 后面跟换行的位置
    import re

    # 找到最后一个完整的 test_case 对象结束位置
    # 模式: "}\n" 后面可能跟空格和逗号
    last_complete = -1
    brace_positions = [m.start() for m in re.finditer(r'\}\s*,?\s*\n', json_str)]
    if brace_positions:
        last_complete = brace_positions[-1] + 1  # 保留 }

    if last_complete <= 0:
        # 无法修复，返回原样
        return json_str

    # 从最后一个完整位置截断
    truncated = json_str[:last_complete]

    # 计算未闭合的括号
    open_braces = truncated.count("{") - truncated.count("}")
    open_brackets = truncated.count("[") - truncated.count("]")

    # 补全：先关闭 ]，再关闭 }
    suffix = ""
    # 交替闭合（JSON 结构通常是 }]}）
    # 简单策略：先补 ] 再补 }
    for _ in range(open_brackets):
        suffix += "\n]"
    for _ in range(open_braces):
        suffix += "\n}"

    repaired = truncated + suffix
    return repaired


def _split_test_plan(plan_json_str: str, output_dir: str, max_suites_per_part: int = 5) -> int:
    """
    将测试计划按 test_suite 切片，每个分片最多包含 max_suites_per_part 个 suite。

    每个分片保留完整的头信息（test_file 的 path, source_file, language, framework,
    includes, mocks, fixtures），只是 test_suites 不同。

    分片文件命名：plan_part_01.json, plan_part_02.json, ...

    返回:
        分片数量
    """
    import shutil

    os.makedirs(output_dir, exist_ok=True)
    # 清理旧分片
    for f in os.listdir(output_dir):
        if f.startswith("plan_part_") and f.endswith(".json"):
            os.remove(os.path.join(output_dir, f))

    try:
        plan = json.loads(plan_json_str)
    except (json.JSONDecodeError, TypeError):
        # 解析失败，整体存为单个分片
        with open(os.path.join(output_dir, "plan_part_01.json"), "w", encoding="utf-8") as f:
            f.write(plan_json_str)
        return 1

    # 顶层信息（plan_summary, flexibility_notes 等）
    top_level_keys = {k: v for k, v in plan.items() if k != "test_files"}
    test_files = plan.get("test_files", [])

    # 收集所有 (test_file_header, suite) 对
    all_suites = []
    for tf in test_files:
        # 头信息 = test_file 中除了 test_suites 之外的所有字段
        header = {k: v for k, v in tf.items() if k != "test_suites"}
        for suite in tf.get("test_suites", []):
            all_suites.append((header, suite))

    if not all_suites:
        with open(os.path.join(output_dir, "plan_part_01.json"), "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        return 1

    # 按 max_suites_per_part 切分
    part_idx = 0
    for i in range(0, len(all_suites), max_suites_per_part):
        chunk = all_suites[i:i + max_suites_per_part]
        part_idx += 1

        # 按 test_file header 分组（同一个 test_file 的 suites 放在一起）
        file_groups = {}
        for header, suite in chunk:
            key = header.get("path", "")
            if key not in file_groups:
                file_groups[key] = {"header": header, "suites": []}
            file_groups[key]["suites"].append(suite)

        # 构建分片 JSON
        part_plan = dict(top_level_keys)
        part_plan["part_index"] = part_idx
        part_plan["total_suites_in_part"] = len(chunk)
        part_plan["test_files"] = []
        for key, group in file_groups.items():
            tf_entry = dict(group["header"])
            tf_entry["test_suites"] = group["suites"]
            part_plan["test_files"].append(tf_entry)

        part_path = os.path.join(output_dir, f"plan_part_{part_idx:02d}.json")
        with open(part_path, "w", encoding="utf-8") as f:
            json.dump(part_plan, f, ensure_ascii=False, indent=2)

    logger.info(f"[UT Agent] 切分结果: {len(all_suites)} 个 suite → {part_idx} 个分片")
    return part_idx


def _get_project_context(repo_path: str) -> str:
    """尝试从克隆的仓库中获取项目上下文（CMakeLists.txt、package.xml 等）。"""
    context_files = ["CMakeLists.txt", "package.xml", "setup.py", "pyproject.toml"]
    context_parts = []
    for fname in context_files:
        fpath = os.path.join(repo_path, fname)
        if os.path.isfile(fpath):
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read(2000)
                context_parts.append(f"### {fname}\n```\n{content}\n```")
            except Exception:
                pass
    return "\n\n".join(context_parts) if context_parts else "无额外项目上下文。"


async def generate_test_plan(state: UTAgentState) -> dict:
    """基于 diff 分析结果生成可执行的测试计划。"""
    logger.info(f"[UT Agent] === Task: generate_test_plan ===")
    mr_id = state["mr_id"]
    repo_path = state.get("repo", "") or ""

    diff_analysis = state.get("diff_analysis")
    if not diff_analysis and state.get("diff_analysis_dir"):
        analysis_dir = state["diff_analysis_dir"]
        parts = []
        if os.path.isdir(analysis_dir):
            for fname in sorted(os.listdir(analysis_dir)):
                fpath = os.path.join(analysis_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    parts.append(f.read())
        diff_analysis = "\n".join(parts)
        logger.info(f"[UT Agent] 从 {analysis_dir} 加载了 {len(parts)} 个批次分析结果")

    if not diff_analysis:
        logger.error("[UT Agent] 无 diff 分析结果，无法生成测试计划")
        return {
            "task": "generate_test_plan",
            "current_action": "失败：无分析结果",
            "next_action": "终止",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    project_context = _get_project_context(repo_path) if repo_path else "无额外项目上下文。"
    logger.info(f"[UT Agent] 项目上下文长度: {len(project_context)} chars")

    system_prompt = load_prompt("generate_test_plan_system")
    user_template = load_prompt("generate_test_plan_user")
    user_prompt = user_template.format(
        title=state["title"],
        author=state["author"],
        mr_id=state["mr_id"],
        source_branch=state["source_branch"],
        target_branch=state["target_branch"],
        repo_path=repo_path or "未克隆",
        diff_analysis=diff_analysis,
        project_context=project_context,
    )

    logger.info(f"[UT Agent] 调用 LLM 生成测试计划...")
    result = await call_llm_with_continuation(
        system=system_prompt,
        user=user_prompt,
        max_tokens=32000,
        max_continuations=3,
    )
    logger.info(f"[UT Agent] 测试计划生成完成，长度: {len(result)} chars")

    # 清理 markdown 代码围栏
    cleaned_result = result.strip()
    if cleaned_result.startswith("```"):
        first_newline = cleaned_result.find("\n")
        if first_newline != -1:
            cleaned_result = cleaned_result[first_newline + 1:]
        if cleaned_result.rstrip().endswith("```"):
            cleaned_result = cleaned_result.rstrip()[:-3].rstrip()

    # 校验 JSON 完整性，如果不完整则尝试修复
    try:
        json.loads(cleaned_result)
        logger.info(f"[UT Agent] 测试计划 JSON 校验通过")
    except json.JSONDecodeError as e:
        logger.warning(f"[UT Agent] 测试计划 JSON 不完整: {e}")
        # 尝试补全：找到最后一个完整的 test_case 对象，截断后闭合
        cleaned_result = _repair_truncated_plan_json(cleaned_result)
        try:
            json.loads(cleaned_result)
            logger.info(f"[UT Agent] JSON 修复成功，修复后长度: {len(cleaned_result)} chars")
        except json.JSONDecodeError as e2:
            logger.error(f"[UT Agent] JSON 修复失败: {e2}")

    output_dir = ToolContext.output_dir
    plan_dir = os.path.join(output_dir, f"mr_{mr_id}")
    os.makedirs(plan_dir, exist_ok=True)
    plan_path = os.path.join(plan_dir, "test_plan.json")
    with open(plan_path, "w", encoding="utf-8") as f:
        f.write(cleaned_result)
    logger.info(f"[UT Agent] 测试计划已落盘: {plan_path}")

    # 切片：每 5 个 test suite 一个分片文件，保留头信息
    plan_parts_dir = os.path.join(plan_dir, "plan_parts")
    part_count = _split_test_plan(cleaned_result, plan_parts_dir, max_suites_per_part=5)
    logger.info(f"[UT Agent] 测试计划已切分为 {part_count} 个分片")

    _publish_test_plan_comment(result, mr_id)

    state_update = {
        "task": "generate_test_plan",
        "current_action": "测试计划生成完成",
        "next_action": "生成测试代码",
        "test_plan": cleaned_result,
        "test_plan_path": plan_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(f"[UT Agent] State更新: task={state_update['task']}")
    return state_update


# ============ Copilot CLI 代码生成 ============

COPILOT_TIMEOUT = 600  # 10 分钟超时

# 受保护的文件模式：Copilot 不应修改这些文件中的非测试 target
PROTECTED_FILE_PATTERNS = ["CMakeLists.txt", "*.cmake"]


def _restore_protected_files(repo_dir: str) -> list[str]:
    """检查被 Copilot 修改的 CMakeLists.txt，如果引用了不存在的源文件则恢复原版。

    返回被拦截的错误描述列表（空列表表示全部通过）。
    """
    import fnmatch
    import re
    violations = []
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=repo_dir, capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return violations
        modified_files = result.stdout.strip().split("\n")
        for fpath in modified_files:
            if not fpath:
                continue
            basename = os.path.basename(fpath)
            if basename != "CMakeLists.txt":
                continue
            # 读取修改后的 CMakeLists.txt，检查引用的源文件是否存在
            full_path = os.path.join(repo_dir, fpath)
            if not os.path.isfile(full_path):
                continue
            cmake_dir = os.path.dirname(full_path)
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            # 匹配 .cpp/.cc/.cxx/.c 文件引用
            source_refs = re.findall(r'[\s(]([^\s()]*\.(?:cpp|cc|cxx|c))\b', content)
            has_missing = False
            missing_files = []
            for src in source_refs:
                src_path = os.path.join(cmake_dir, src) if not os.path.isabs(src) else os.path.join(repo_dir, src)
                if not os.path.isfile(src_path):
                    missing_files.append(src)
                    has_missing = True
            if has_missing:
                violation_msg = f"{fpath} 引用了不存在的源文件: {missing_files}"
                logger.warning(f"[UT Agent] CMakeLists.txt {violation_msg}")
                logger.warning(f"[UT Agent] 恢复被错误修改的 CMakeLists.txt: {fpath}")
                subprocess.run(
                    ["git", "checkout", "--", fpath],
                    cwd=repo_dir, capture_output=True, timeout=10
                )
                violations.append(violation_msg)
            else:
                logger.info(f"[UT Agent] CMakeLists.txt 修改通过验证，保留: {fpath}")
    except Exception as e:
        logger.warning(f"[UT Agent] 检查 CMakeLists.txt 时出错: {e}")
    return violations


def _build_copilot_prompt(test_plan: str, pending_cases: str | None, repo_dir: str, mr_id: int, iteration: int, languages: set[str] | None = None, fix_plan: str | None = None) -> str:
    """构造传给 Copilot CLI 的 prompt，从模板文件加载并按语言拼接专项规范。"""
    if fix_plan:
        # 修复模式：将 fix_plan 的 evidence + instructions 作为任务描述
        fix_data = json.loads(fix_plan) if isinstance(fix_plan, str) else fix_plan
        failure_type = fix_data.get("failure_type", "unknown")
        failure_reason = fix_data.get("failure_reason", "")
        instructions = fix_data.get("instructions", "")
        evidence = fix_data.get("evidence", [])
        evidence_text = "\n".join(evidence[:20]) if isinstance(evidence, list) else str(evidence)
        task_description = f"""## CI 流水线修复模式（第 {fix_data.get('iteration', '?')} 轮）

**失败类型:** {failure_type}
**失败原因:** {failure_reason}

### 修复指令
{instructions}

### CI 错误日志（evidence）
```
{evidence_text}
```

请根据以上错误日志和修复指令，在当前仓库中修复测试代码的问题。"""
    elif pending_cases:
        task_description = f"""以下是上一轮未完成的测试用例差异清单，请补充生成这些测试：

未完成用例：
{pending_cases}"""
    else:
        task_description = f"""请根据以下测试计划，在当前仓库中生成对应的单元测试代码文件。

测试计划（JSON）：
{test_plan}"""

    # 通用规范
    system_prompt = load_prompt("generate_patch_system")

    # 按语言追加专项规范
    lang_sections = []
    if languages is None:
        languages = {"cpp", "python"}
    if languages & {"cpp", "c", "cc", "cxx", "h", "hpp"}:
        lang_sections.append(load_prompt("generate_patch_cpp"))
    if languages & {"python", "py"}:
        lang_sections.append(load_prompt("generate_patch_python"))

    full_system = system_prompt
    if lang_sections:
        full_system += "\n\n" + "\n\n".join(lang_sections)

    user_template = load_prompt("generate_patch_user")
    user_prompt = user_template.format(
        task_description=task_description,
        repo_dir=repo_dir,
        mr_id=mr_id,
        iteration=iteration,
    )

    return f"{full_system}\n\n---\n\n{user_prompt}"


def _diagnose_copilot_timeout(stdout_lines: list[str], stderr_lines: list[str], elapsed: float) -> str:
    """根据 Copilot CLI 的部分输出诊断超时原因。"""
    all_output = "\n".join(stdout_lines + stderr_lines).lower()

    # 429 / Rate limit
    if "429" in all_output or "rate limit" in all_output or "too many requests" in all_output:
        return "LLM 429 限流 - Copilot API 请求过于频繁，需要等待后重试"

    # 认证问题
    if "unauthorized" in all_output or "401" in all_output or "auth" in all_output and "fail" in all_output:
        return "认证失败 - GITHUB_TOKEN 可能过期或权限不足"

    # 网络问题
    if "timeout" in all_output or "timed out" in all_output or "connection" in all_output and ("refused" in all_output or "reset" in all_output):
        return "网络超时 - 无法连接到 Copilot API 服务"

    # Prompt 过长
    if "too long" in all_output or "context length" in all_output or "token limit" in all_output:
        return "Prompt 过长 - 超出模型 context window 限制"

    # 没有任何输出（进程完全卡死）
    if not stdout_lines:
        return "无输出 - Copilot CLI 启动后无任何响应，可能是进程挂起或 PATH 问题"

    # 有输出但一直在搜索/读文件（死循环）
    search_count = sum(1 for l in stdout_lines if "search" in l.lower() or "read" in l.lower())
    if search_count > 20:
        return f"搜索循环 - Copilot 执行了 {search_count} 次搜索/读取操作后超时，可能在查找不存在的文件"

    # 有输出但很少（可能在等待 API 响应）
    if len(stdout_lines) < 5:
        return f"API 响应慢 - 仅 {len(stdout_lines)} 行输出后超时，可能是 LLM API 响应延迟"

    return f"未知原因 - 已有 {len(stdout_lines)} 行输出，最后活动在第 {int(elapsed)}s"


def _run_copilot_generate(
    repo_dir: str,
    test_plan: str,
    pending_cases: str | None,
    mr_id: int,
    iteration: int,
    languages: set[str] | None = None,
    fix_plan: str | None = None,
) -> list[str]:
    """调用 Copilot CLI 在 repo 中生成测试代码，返回生成的文件路径列表。"""
    prompt = _build_copilot_prompt(test_plan, pending_cases, repo_dir, mr_id, iteration, languages, fix_plan=fix_plan)
    logger.info(f"[UT Agent] 调用 Copilot CLI 生成测试代码 (iter={iteration}, fix_mode={fix_plan is not None})...")
    logger.debug(f"[UT Agent] Copilot prompt 长度: {len(prompt)} chars")

    # 记录调用前的文件快照（用于对比找出新文件）
    before_files = _snapshot_test_files(repo_dir)

    cmd = [
        "copilot",
        "-p", prompt,
        "--allow-all-tools",
        "--deny-tool=shell(git push)",
        "--deny-tool=shell(git commit)",
        "--deny-tool=shell(rm)",
    ]

    try:
        # 使用 Popen 实时流式输出日志（而非 subprocess.run 阻塞等待）
        import time as _time
        proc = subprocess.Popen(
            cmd,
            cwd=repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout_lines = []
        stderr_lines = []
        start_time = _time.time()

        # 实时读取 stdout 并写入日志
        while True:
            # 检查超时
            elapsed = _time.time() - start_time
            if elapsed > COPILOT_TIMEOUT:
                proc.kill()
                # 读取已有的 stderr 用于诊断
                try:
                    remaining_stderr = proc.stderr.read()
                    if remaining_stderr:
                        stderr_lines.extend(remaining_stderr.strip().split("\n"))
                except Exception:
                    pass

                # 诊断超时原因
                timeout_reason = _diagnose_copilot_timeout(stdout_lines, stderr_lines, elapsed)
                logger.error(f"[UT Agent] Copilot CLI 超时 ({int(elapsed)}s)，已终止")
                logger.error(f"[UT Agent] 超时诊断: {timeout_reason}")
                logger.error(f"[UT Agent] 最后输出: {' | '.join(stdout_lines[-5:]) if stdout_lines else '无'}")
                if stderr_lines:
                    logger.error(f"[UT Agent] stderr: {' | '.join(stderr_lines[-3:])}")
                return []

            line = proc.stdout.readline()
            if line:
                stripped = line.rstrip()
                stdout_lines.append(stripped)
                # 实时输出关键行到日志（过滤空行和过长行）
                if stripped and len(stripped) < 500:
                    logger.info(f"[Copilot] {stripped}")
            elif proc.poll() is not None:
                # 进程已结束，读取剩余输出
                remaining = proc.stdout.read()
                if remaining:
                    stdout_lines.extend(remaining.rstrip().split("\n"))
                break

        # 读取 stderr
        stderr_output = proc.stderr.read()
        if stderr_output:
            stderr_lines = stderr_output.strip().split("\n")

        returncode = proc.returncode
        logger.info(f"[UT Agent] Copilot CLI 退出码: {returncode} (耗时 {int(elapsed)}s)")
        if stderr_lines:
            logger.warning(f"[UT Agent] Copilot stderr:\n" + "\n".join(stderr_lines[-5:]))

    except Exception as e:
        logger.error(f"[UT Agent] Copilot CLI 调用异常: {e}")
        return []

    # 安全网：恢复被 Copilot 意外修改的构建配置文件
    violations = _restore_protected_files(repo_dir)

    # 如果安全网拦截了 CMakeLists.txt 修改，带错误信息重试一次
    if violations and not hasattr(_run_copilot_generate, "_in_retry"):
        _run_copilot_generate._in_retry = True
        try:
            retry_msg = (
                "你上一次修改的 CMakeLists.txt 被回滚了，因为引用了不存在的源文件。\n"
                "具体错误:\n" + "\n".join(f"- {v}" for v in violations) + "\n\n"
                "请重新修改，确保 CMakeLists.txt 中引用的每个源文件都实际存在于仓库中。\n"
                "如果文件不存在，不要引用它。只引用你能在仓库中搜索到的文件。"
            )
            logger.info(f"[UT Agent] 安全网拦截，带错误反馈重试 Copilot CLI...")
            retry_prompt = prompt + f"\n\n---\n\n## ⚠️ 上次修改被回滚\n\n{retry_msg}"
            # 重新记录快照
            before_files = _snapshot_test_files(repo_dir)
            retry_cmd = [
                "copilot",
                "-p", retry_prompt,
                "--allow-all-tools",
                "--deny-tool=shell(git push)",
                "--deny-tool=shell(git commit)",
                "--deny-tool=shell(rm)",
            ]
            proc = subprocess.Popen(
                retry_cmd,
                cwd=repo_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout_lines = []
            start_time = _time.time()
            while True:
                elapsed = _time.time() - start_time
                if elapsed > COPILOT_TIMEOUT:
                    proc.kill()
                    logger.error(f"[UT Agent] 重试 Copilot CLI 超时 ({int(elapsed)}s)")
                    break
                line = proc.stdout.readline()
                if line:
                    stripped = line.rstrip()
                    stdout_lines.append(stripped)
                    if stripped and len(stripped) < 500:
                        logger.info(f"[Copilot-retry] {stripped}")
                elif proc.poll() is not None:
                    remaining = proc.stdout.read()
                    if remaining:
                        stdout_lines.extend(remaining.rstrip().split("\n"))
                    break
            returncode = proc.returncode
            logger.info(f"[UT Agent] 重试 Copilot CLI 退出码: {returncode} (耗时 {int(elapsed)}s)")
            # 重试后再过一次安全网
            retry_violations = _restore_protected_files(repo_dir)
            if retry_violations:
                logger.warning(f"[UT Agent] 重试后仍有安全网违规: {retry_violations}")
        finally:
            del _run_copilot_generate._in_retry

    # 对比文件快照，找出新增/修改的测试文件
    after_files = _snapshot_test_files(repo_dir)
    new_files = _diff_snapshots(before_files, after_files)

    if new_files:
        logger.info(f"[UT Agent] Copilot 生成了 {len(new_files)} 个测试文件:")
        for f in new_files:
            logger.info(f"  - {f}")
    else:
        logger.warning("[UT Agent] Copilot CLI 未生成任何新测试文件")

    return new_files


def _snapshot_test_files(repo_dir: str) -> dict[str, float]:
    """遍历 repo 目录，记录所有文件的 mtime（用于对比找出 Copilot 新增的文件）。"""
    snapshot = {}
    if not os.path.isdir(repo_dir):
        return snapshot
    for root, _dirs, files in os.walk(repo_dir):
        # 跳过 .git 目录
        if "/.git" in root or "\\.git" in root:
            continue
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                snapshot[fpath] = os.path.getmtime(fpath)
            except OSError:
                pass
    return snapshot


def _diff_snapshots(before: dict[str, float], after: dict[str, float]) -> list[str]:
    """对比两次快照，返回新增或修改的文件路径列表（测试文件 + CMakeLists.txt）。"""
    test_extensions = {".cpp", ".cc", ".cxx", ".h", ".hpp", ".py"}
    test_indicators = {"test_", "_test.", "_test_", "Test", "tests/", "test/"}
    new_or_modified = []
    for fpath, mtime in after.items():
        if fpath not in before or mtime > before[fpath]:
            basename = os.path.basename(fpath)
            # CMakeLists.txt 修改也需要追踪（已通过安全网验证）
            if basename == "CMakeLists.txt":
                new_or_modified.append(fpath)
                continue
            # 只保留看起来是测试文件的
            ext = os.path.splitext(fpath)[1].lower()
            if ext not in test_extensions:
                continue
            path_lower = fpath.replace("\\", "/").lower()
            is_test = any(ind.lower() in path_lower for ind in test_indicators)
            if is_test:
                new_or_modified.append(fpath)
    return sorted(new_or_modified)


# ============ End Copilot CLI ============


async def generate_patch(state: UTAgentState) -> dict:
    """根据测试计划（或差异清单中的未完成用例）生成 UT 代码 patch。"""
    logger.info(f"[UT Agent] === Task: generate_patch ===")
    mr_id = state["mr_id"]
    iteration = state.get("patch_iterations", 0) + 1
    pending_cases = state.get("pending_cases")

    # 检查是否处于修复模式
    fix_plan = state.get("fix_plan")
    if fix_plan:
        logger.info(f"[UT Agent] 修复模式: 使用 fix_plan 作为 Copilot 输入")

    if TEST_MODE:
        # 测试模式：在 repo 内生成一个 dummy hello world 测试文件
        logger.info(f"[UT Agent] [TEST_MODE] 生成 dummy 测试文件")
        repo_dir = state.get("repo", "")
        if " (" in repo_dir:
            repo_dir = repo_dir.split(" (")[0]
        dummy_dir = os.path.join(repo_dir, "tests", "ut_agent_test")
        os.makedirs(dummy_dir, exist_ok=True)
        dummy_file = os.path.join(dummy_dir, "test_hello_world.cpp")
        with open(dummy_file, "w", encoding="utf-8") as f:
            f.write('''#include <iostream>\n#include <gtest/gtest.h>\n\nTEST(UTAgentSmokeTest, HelloWorld) {\n    std::string msg = "hello world";\n    EXPECT_FALSE(msg.empty());\n    EXPECT_EQ(msg, "hello world");\n    std::cout << msg << std::endl;\n}\n\nint main(int argc, char **argv) {\n    ::testing::InitGoogleTest(&argc, argv);\n    return RUN_ALL_TESTS();\n}\n''')
        logger.info(f"[UT Agent] [TEST_MODE] 已生成: {dummy_file}")
        generated_patches = [dummy_file]
    else:
        if pending_cases:
            logger.info(f"[UT Agent] 第{iteration}次迭代，处理未完成用例差异清单")
        else:
            logger.info(f"[UT Agent] 首次生成 patch，基于完整测试计划")

        repo_dir = state.get("repo", "")
        if " (" in repo_dir:
            repo_dir = repo_dir.split(" (")[0]

        # 从 diff_files 中提取涉及的语言
        languages = set()
        for f in state.get("diff_files", []):
            lang = f.get("language", "").lower()
            if lang:
                languages.add(lang)

        # 如果是首次生成且有分片，逐片调用 Copilot（追加模式）
        output_dir = ToolContext.output_dir
        plan_parts_dir = os.path.join(output_dir, f"mr_{mr_id}", "plan_parts")
        generated_patches = []

        if fix_plan:
            # 修复模式：直接使用 fix_plan 调用 Copilot
            generated_patches = _run_copilot_generate(
                repo_dir=repo_dir,
                test_plan=state.get("test_plan", ""),
                pending_cases=None,
                mr_id=mr_id,
                iteration=iteration,
                languages=languages or None,
                fix_plan=fix_plan,
            )
        elif not pending_cases and os.path.isdir(plan_parts_dir):
            part_files = sorted([
                f for f in os.listdir(plan_parts_dir)
                if f.startswith("plan_part_") and f.endswith(".json")
            ])
            if part_files:
                logger.info(f"[UT Agent] 分片模式: {len(part_files)} 个分片待处理")
                for idx, part_file in enumerate(part_files, 1):
                    part_path = os.path.join(plan_parts_dir, part_file)
                    with open(part_path, "r", encoding="utf-8") as f:
                        part_plan = f.read()
                    logger.info(f"[UT Agent] 处理分片 {idx}/{len(part_files)}: {part_file} ({len(part_plan)} chars)")
                    new_files = _run_copilot_generate(
                        repo_dir=repo_dir,
                        test_plan=part_plan,
                        pending_cases=None,
                        mr_id=mr_id,
                        iteration=iteration,
                        languages=languages or None,
                    )
                    generated_patches.extend(new_files)
                    logger.info(f"[UT Agent] 分片 {idx} 完成: 生成 {len(new_files)} 个文件，累计 {len(generated_patches)} 个")
            else:
                # 无分片文件，回退到整体模式
                generated_patches = _run_copilot_generate(
                    repo_dir=repo_dir,
                    test_plan=state.get("test_plan", ""),
                    pending_cases=pending_cases,
                    mr_id=mr_id,
                    iteration=iteration,
                    languages=languages or None,
                )
        else:
            # pending_cases 模式或无分片目录
            generated_patches = _run_copilot_generate(
                repo_dir=repo_dir,
                test_plan=state.get("test_plan", ""),
                pending_cases=pending_cases,
                mr_id=mr_id,
                iteration=iteration,
                languages=languages or None,
            )

        # 去重（同一个文件可能在多个分片中被追加修改）
        generated_patches = list(dict.fromkeys(generated_patches))

    # 落盘 patch 文件列表到 JSON（解决 LangGraph state 传播不可靠问题）
    output_dir = ToolContext.output_dir
    patches_manifest = os.path.join(output_dir, f"mr_{mr_id}", "generated_patches.json")
    os.makedirs(os.path.dirname(patches_manifest), exist_ok=True)
    with open(patches_manifest, "w", encoding="utf-8") as f:
        json.dump(generated_patches, f, ensure_ascii=False)
    logger.info(f"[UT Agent] patch 清单已落盘: {patches_manifest}")

    state_update = {
        "task": "generate_patch",
        "current_action": f"Patch 生成完成（第{iteration}次迭代）",
        "next_action": "校验计划完成度",
        "generated_patches": generated_patches,
        "patch_iterations": iteration,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(f"[UT Agent] State更新: iteration={iteration}, patches={len(generated_patches)}")
    return state_update


async def validate_plan(state: UTAgentState) -> dict:
    """比对已生成的 patch 与落盘的测试计划，计算差异清单（gap）。"""
    logger.info(f"[UT Agent] === Task: validate_plan ===")
    mr_id = state["mr_id"]
    iteration = state.get("patch_iterations", 0)
    output_dir = ToolContext.output_dir
    mr_dir = os.path.join(output_dir, f"mr_{mr_id}")

    # 修复模式下跳过计划校验（fix 循环的目标是修复 CI 错误，不是生成新用例）
    if state.get("fix_plan"):
        logger.info(f"[UT Agent] 修复模式，跳过计划校验，直接通过")
        return {
            "task": "validate_plan",
            "current_action": "修复模式：跳过计划校验",
            "next_action": "上传到 GitLab",
            "plan_valid": True,
            "pending_cases": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # 1. 从落盘的测试计划中提取所有用例清单
    plan_path = os.path.join(mr_dir, "test_plan.json")
    if not os.path.isfile(plan_path):
        logger.error(f"[UT Agent] 测试计划文件不存在: {plan_path}")
        return {
            "task": "validate_plan",
            "current_action": "校验失败：计划文件不存在",
            "next_action": "终止",
            "plan_valid": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    with open(plan_path, "r", encoding="utf-8") as f:
        plan_content = f.read()

    # 提取计划中所有测试用例名（作为基准清单）
    all_planned_cases = _extract_planned_cases(plan_content)
    logger.info(f"[UT Agent] 计划中共 {len(all_planned_cases)} 个测试用例")

    # 2. 读取已生成的 patch 文件，提取已实现的用例名
    generated_patches = state.get("generated_patches") or []
    completed_cases = _extract_completed_cases(generated_patches)
    logger.info(f"[UT Agent] 已完成 {len(completed_cases)} 个测试用例")

    # 3. 计算差异：计划中有但 patch 中没有的 = pending
    pending_cases = [c for c in all_planned_cases if c["name"] not in completed_cases]

    # 一致性校验：只判断「计划中的用例是否全部在 patch 中实现」
    # 阈值/优先级类放宽逻辑放到 route_after_validate，避免污染 plan_valid 语义。
    if len(all_planned_cases) == 0 and len(completed_cases) == 0:
        plan_valid = False
        logger.warning(f"[UT Agent] 计划为空且无 patch，判定为未通过")
    else:
        plan_valid = len(pending_cases) == 0

    # 4. 落盘本次校验结果
    validation_result = {
        "iteration": iteration,
        "total_planned": len(all_planned_cases),
        "total_completed": len(completed_cases),
        "completion_rate": f"{len(completed_cases)}/{len(all_planned_cases)}",
        "all_completed": plan_valid,
        "completed_cases": list(completed_cases),
        "pending_cases": pending_cases,
    }
    validation_path = os.path.join(mr_dir, f"validation_iter_{iteration}.json")
    os.makedirs(os.path.dirname(validation_path), exist_ok=True)
    with open(validation_path, "w", encoding="utf-8") as f:
        json.dump(validation_result, f, ensure_ascii=False, indent=2)
    logger.info(f"[UT Agent] 校验结果已落盘: {validation_path}")
    logger.info(f"[UT Agent] 完成率: {len(completed_cases)}/{len(all_planned_cases)} | 通过: {plan_valid}")

    state_update = {
        "task": "validate_plan",
        "current_action": "全部用例已覆盖" if plan_valid else f"存在 {len(pending_cases)} 个未覆盖用例（第{iteration}次）",
        "next_action": "上传到 GitLab" if plan_valid else "继续生成 patch",
        "plan_valid": plan_valid,
        "pending_cases": json.dumps(pending_cases, ensure_ascii=False) if pending_cases else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(f"[UT Agent] State更新: valid={plan_valid}, iteration={iteration}")
    return state_update


def _extract_planned_cases(plan_json_str: str) -> list[dict]:
    """从测试计划 JSON 中提取所有测试用例（name + priority + suite）。"""
    cases = []

    # 清理 markdown 代码围栏（LLM 可能返回 ```json ... ```）
    cleaned = plan_json_str.strip()
    if cleaned.startswith("```"):
        # 移除开头的 ```json 或 ```
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1:]
        # 移除结尾的 ```
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3].rstrip()

    try:
        plan = json.loads(cleaned)
        for test_file in plan.get("test_files", []):
            for suite in test_file.get("test_suites", []):
                suite_name = suite.get("suite_name", "")
                for tc in suite.get("test_cases", []):
                    cases.append({
                        "name": tc.get("name", ""),
                        "suite": suite_name,
                        "priority": tc.get("priority", "P1"),
                        "description": tc.get("description", ""),
                        "assertions": tc.get("assertions", []),
                    })
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        logger.warning(f"[UT Agent] 解析测试计划 JSON 失败: {e}")
    return cases


def _extract_completed_cases(patch_paths: list[str]) -> set[str]:
    """从已生成的测试代码文件中提取已实现的测试用例名（通过正则匹配测试函数）。"""
    import re
    completed = set()
    # 匹配 GTest: TEST_F(Suite, Name) / TEST(Suite, Name)
    gtest_pattern = re.compile(r"TEST(?:_F)?\s*\(\s*\w+\s*,\s*(\w+)\s*\)")
    # 匹配 pytest: def test_xxx
    pytest_pattern = re.compile(r"def\s+(test_\w+)\s*\(")

    for path in patch_paths:
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        for match in gtest_pattern.finditer(content):
            completed.add(match.group(1))
        for match in pytest_pattern.finditer(content):
            completed.add(match.group(1))
    return completed


def _check_completion_threshold(all_cases: list[dict], completed: set[str]) -> bool:
    """检查完成度是否达标：P0 全部完成 且 总完成率 >= 80%。"""
    if not all_cases:
        return True
    p0_cases = [c for c in all_cases if c["priority"] == "P0"]
    p0_completed = all(c["name"] in completed for c in p0_cases)
    total_rate = len(completed) / len(all_cases)
    return p0_completed and total_rate >= 0.8


def route_after_validate(state: UTAgentState) -> str:
    """校验后路由：
    - plan_valid（计划与 patch 完全一致）-> upload_to_gitlab
    - 未一致但已达可接受阈值（P0 全部完成且总完成率 >= 80%）-> upload_to_gitlab
    - 未一致且未超迭代上限 -> 继续 generate_patch
    - 超限兜底 -> upload_to_gitlab
    """
    if state.get("plan_valid"):
        return "upload_to_gitlab"
    # 即便不完全一致，达到可接受阈值也直接 upload
    try:
        pending = json.loads(state.get("pending_cases") or "[]")
    except (TypeError, ValueError):
        pending = []
    # 这里只关心是否达到阈值，不需要重新解析计划文件——但 _check_completion_threshold
    # 需要完整 all_cases/completed 集合，沿用 validate_plan 的落盘结果会更可靠。
    mr_id = state.get("mr_id")
    iteration = state.get("patch_iterations", 0)
    output_dir = ToolContext.output_dir
    val_path = os.path.join(output_dir, f"mr_{mr_id}", f"validation_iter_{iteration}.json")
    if os.path.isfile(val_path):
        try:
            with open(val_path, "r", encoding="utf-8") as f:
                val = json.load(f)
            total_planned = val.get("total_planned", 0)
            total_completed = val.get("total_completed", 0)
            pending_list = val.get("pending_cases", []) or []
            if total_planned > 0:
                rate = total_completed / total_planned
                p0_pending = any(c.get("priority") == "P0" for c in pending_list)
                if not p0_pending and rate >= 0.8:
                    logger.info(f"[UT Agent] 达到可接受阈值 (rate={rate:.2f}, P0 全部完成)，提前 upload")
                    return "upload_to_gitlab"
        except Exception as e:
            logger.warning(f"[UT Agent] 读取 validation 结果失败: {e}")
    if state.get("patch_iterations", 0) < MAX_PATCH_ITERATIONS:
        return "generate_patch"
    # 超限也上传已有的成果
    return "upload_to_gitlab"


def upload_to_gitlab(state: UTAgentState) -> dict:
    """将生成的测试代码文件提交并推送到 MR 源分支。"""
    logger.info(f"[UT Agent] === Task: upload_to_gitlab ===")
    mr_id = state["mr_id"]
    repo_dir = state.get("repo", "")
    source_branch = state["source_branch"]

    # 优先从落盘的 JSON 清单读取 patch 列表（解决 state 传播不可靠问题）
    output_dir = ToolContext.output_dir
    patches_manifest = os.path.join(output_dir, f"mr_{mr_id}", "generated_patches.json")
    if os.path.isfile(patches_manifest):
        with open(patches_manifest, "r", encoding="utf-8") as f:
            generated_patches = json.load(f)
        logger.info(f"[UT Agent] 从清单文件加载 {len(generated_patches)} 个 patch")
    else:
        generated_patches = state.get("generated_patches") or []
        logger.info(f"[UT Agent] 从 state 加载 {len(generated_patches)} 个 patch")

    if not repo_dir:
        logger.error("[UT Agent] 无仓库路径，无法上传")
        return {
            "task": "upload_to_gitlab",
            "current_action": "上传失败：无仓库路径",
            "next_action": "终止",
            "response": "ERROR: 无克隆仓库，无法推送测试代码",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    if not generated_patches:
        logger.warning("[UT Agent] 无已生成的 patch 文件，跳过上传")
        return {
            "task": "upload_to_gitlab",
            "current_action": "跳过：无 patch 文件",
            "next_action": "完成",
            "response": "UT Agent 完成分析和计划生成，但未产出测试代码文件。",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # 清理 repo_dir（可能带有 "(已存在，跳过克隆)" 后缀）
    if " (" in repo_dir:
        repo_dir = repo_dir.split(" (")[0]

    logger.info(f"[UT Agent] 提交 {len(generated_patches)} 个文件到 {source_branch}")
    result = commit_and_push(
        repo_dir=repo_dir,
        patch_files=generated_patches,
        source_branch=source_branch,
        mr_id=mr_id,
    )
    logger.info(f"[UT Agent] 上传结果: {result}")

    # 从结果中提取 commit hash
    commit_sha = None
    if "commit=" in result:
        commit_sha = result.split("commit=")[-1].strip()

    if result.startswith("ERROR:"):
        # 发布错误评论
        git_provider = ToolContext.git_provider
        if git_provider:
            git_provider.publish_comment(
                f"## UT Agent - 上传失败 ❌\n\n{result}"
            )
        return {
            "task": "upload_to_gitlab",
            "current_action": f"上传失败: {result}",
            "next_action": "终止",
            "response": f"ERROR: 推送测试代码失败: {result}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # 成功：发布评论
    git_provider = ToolContext.git_provider
    if git_provider:
        git_provider.publish_comment(
            f"## UT Agent - 测试代码已推送 ✅\n\n"
            f"已将 {len(generated_patches)} 个测试文件推送到分支 `{source_branch}`。\n\n"
            f"请检查新提交并运行 CI 验证。"
        )

    return {
        "task": "upload_to_gitlab",
        "current_action": "上传完成",
        "next_action": "完成",
        "commit_sha": commit_sha,
        "response": f"UT Agent 已将 {len(generated_patches)} 个测试文件推送到 {source_branch}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _publish_test_plan_comment(content: str, mr_id: int):
    """将测试计划作为评论发布到 MR。"""
    git_provider = ToolContext.git_provider
    if not git_provider:
        return
    body = (f"## UT Agent - 测试计划 📋\n\n"
            f"**MR:** !{mr_id}\n\n"
            f"```json\n{content}\n```")
    git_provider.publish_comment(body)


def check_pipeline(state: UTAgentState) -> dict:
    """等待 CI 流水线完成并获取覆盖率和失败 job 日志。"""
    logger.info(f"[UT Agent] === Task: check_pipeline ===")
    commit_sha = state.get("commit_sha")
    mr_id = state["mr_id"]

    if not commit_sha:
        logger.warning("[UT Agent] 无 commit_sha，跳过流水线检查")
        return {
            "task": "check_pipeline",
            "pipeline_feedback": None,
            "response": "跳过流水线检查：无 commit SHA",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    result = fetch_pipeline_feedback(commit_sha)

    # 落盘结果供后续使用
    import json as _json
    output_dir = ToolContext.output_dir
    feedback_path = os.path.join(output_dir, f"mr_{mr_id}", "pipeline_feedback.json")
    os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
    with open(feedback_path, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"[UT Agent] 流水线反馈已落盘: {feedback_path}")

    # 发布 MR 评论
    git_provider = ToolContext.git_provider
    if git_provider and result.get("message"):
        emoji = "✅" if result.get("pipeline_status") == "success" else "❌"
        comment = (
            f"## UT Agent - 流水线反馈 {emoji}\n\n"
            f"{result['message']}\n"
        )
        if result.get("failed_jobs"):
            for fj in result["failed_jobs"]:
                if fj.get("log_tail"):
                    comment += f"\n<details><summary>{fj['name']} 错误日志</summary>\n\n```\n{fj['log_tail']}\n```\n</details>\n"
        git_provider.publish_comment(comment)

    return {
        "task": "check_pipeline",
        "pipeline_feedback": _json.dumps(result, ensure_ascii=False),
        "response": result.get("message", "流水线检查完成"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Verifier: 结构化判定流水线结果
# ──────────────────────────────────────────────────────────────────────────────

# 失败类型枚举
FAILURE_TYPE_BUILD = "build_failure"
FAILURE_TYPE_TEST = "test_failure"
FAILURE_TYPE_COVERAGE = "coverage_insufficient"
FAILURE_TYPE_TIMEOUT = "pipeline_timeout"
FAILURE_TYPE_UNKNOWN = "unknown"


def verify_pipeline(state: UTAgentState) -> dict:
    """
    Verifier 节点：根据 check_pipeline 结果输出结构化验证结论。

    输出 verdict:
        {
            "result": "PASS" | "FAIL",
            "failure_type": str | None,
            "failure_reason": str | None,
            "evidence": list[str],  # 关键证据（日志片段/数字）
            "failed_job_names": list[str],
        }
    """
    logger.info(f"[UT Agent] === Task: verify_pipeline ===")

    feedback_raw = state.get("pipeline_feedback")
    if not feedback_raw:
        # 没有流水线反馈（可能 check_pipeline 跳过了）
        verdict = {
            "result": "PASS",
            "failure_type": None,
            "failure_reason": "无流水线反馈数据，视为通过",
            "evidence": [],
            "failed_job_names": [],
        }
        logger.info(f"[UT Agent] Verifier: PASS (无反馈数据)")
        return {
            "task": "verify_pipeline",
            "verification_verdict": json.dumps(verdict, ensure_ascii=False),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    feedback = json.loads(feedback_raw)
    pipeline_status = feedback.get("pipeline_status")
    coverage = feedback.get("coverage")
    coverage_threshold = feedback.get("coverage_threshold")  # 从 job 日志提取的阈值
    failed_jobs = feedback.get("failed_jobs", [])
    fb_status = feedback.get("status")  # success / timeout / error

    # 使用 job 报告的阈值，如果没有则使用配置的默认值
    threshold = coverage_threshold if coverage_threshold is not None else MIN_COVERAGE_THRESHOLD

    evidence = []
    # 区分目标 job 和非目标 job 的失败
    target_failed = [fj for fj in failed_jobs if fj.get("is_target", True) and fj.get("status") == "failed"]
    non_target_failed = [fj for fj in failed_jobs if not fj.get("is_target", True) and fj.get("status") == "failed"]
    failed_job_names = [fj["name"] for fj in target_failed]

    # 判定逻辑
    # 0. pipeline 整体失败但目标 job 未报失败 → 仍然算 FAIL
    if pipeline_status == "failed" and not failed_job_names:
        # 有非目标 job 失败
        if non_target_failed:
            non_target_names = [fj["name"] for fj in non_target_failed]
            log_snippets = []
            for fj in non_target_failed[:3]:  # 最多取 3 个
                snippet = (fj.get("log_tail") or "")[:1000]
                log_snippets.append(f"{fj['name']}:\n{snippet}")
            evidence.append(f"非目标 job 失败: {non_target_names}")
            evidence.extend(log_snippets)
            verdict = {
                "result": "FAIL",
                "failure_type": FAILURE_TYPE_BUILD,
                "failure_reason": f"流水线失败，失败 job: {', '.join(non_target_names)}",
                "evidence": evidence,
                "failed_job_names": non_target_names,
            }
        else:
            verdict = {
                "result": "FAIL",
                "failure_type": FAILURE_TYPE_UNKNOWN,
                "failure_reason": f"流水线状态为 failed 但未找到具体失败 job",
                "evidence": [feedback.get("message", "")],
                "failed_job_names": [],
            }
        logger.info(f"[UT Agent] Verifier: FAIL (pipeline_failed, no target job matched)")
        return {
            "task": "verify_pipeline",
            "verification_verdict": json.dumps(verdict, ensure_ascii=False),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # 1. 超时 -> FAIL (timeout)
    if fb_status == "timeout":
        verdict = {
            "result": "FAIL",
            "failure_type": FAILURE_TYPE_TIMEOUT,
            "failure_reason": feedback.get("message", "流水线超时未完成"),
            "evidence": [feedback.get("message", "")],
            "failed_job_names": [],
        }
        logger.info(f"[UT Agent] Verifier: FAIL (timeout)")
        return {
            "task": "verify_pipeline",
            "verification_verdict": json.dumps(verdict, ensure_ascii=False),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # 2. 编译失败: build_release_arm64 失败
    if "build_release_arm64" in failed_job_names:
        build_job = next(fj for fj in failed_jobs if fj["name"] == "build_release_arm64")
        log_snippet = build_job.get("log_tail") or ""
        evidence.append(f"build_release_arm64 失败日志:\n{log_snippet}")
        verdict = {
            "result": "FAIL",
            "failure_type": FAILURE_TYPE_BUILD,
            "failure_reason": "ARM64 编译失败",
            "evidence": evidence,
            "failed_job_names": failed_job_names,
        }
        logger.info(f"[UT Agent] Verifier: FAIL (build_failure)")
        return {
            "task": "verify_pipeline",
            "verification_verdict": json.dumps(verdict, ensure_ascii=False),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # 3. 测试失败: x86_64_ut_coverage_check 失败
    if "x86_64_ut_coverage_check" in failed_job_names:
        ut_job = next(fj for fj in failed_jobs if fj["name"] == "x86_64_ut_coverage_check")
        log_snippet = ut_job.get("log_tail") or ""
        evidence.append(f"x86_64_ut_coverage_check 失败日志:\n{log_snippet}")

        # 区分是测试用例失败还是覆盖率不足
        log_lower = log_snippet.lower()
        if "coverage" in log_lower and ("insufficient" in log_lower or "below" in log_lower or "不足" in log_lower):
            failure_type = FAILURE_TYPE_COVERAGE
            failure_reason = "单元测试覆盖率未达标"
        else:
            failure_type = FAILURE_TYPE_TEST
            failure_reason = "单元测试用例执行失败"

        verdict = {
            "result": "FAIL",
            "failure_type": failure_type,
            "failure_reason": failure_reason,
            "evidence": evidence,
            "failed_job_names": failed_job_names,
        }
        logger.info(f"[UT Agent] Verifier: FAIL ({failure_type})")
        return {
            "task": "verify_pipeline",
            "verification_verdict": json.dumps(verdict, ensure_ascii=False),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # 4. 流水线通过 — 还需校验覆盖率是否达标
    if pipeline_status == "success":
        if coverage is not None:
            evidence.append(f"覆盖率: {coverage}%, 阈值: {threshold}%")
            if coverage < threshold:
                verdict = {
                    "result": "FAIL",
                    "failure_type": FAILURE_TYPE_COVERAGE,
                    "failure_reason": f"覆盖率 {coverage}% 低于阈值 {threshold}%",
                    "evidence": evidence,
                    "failed_job_names": [],
                }
                logger.info(f"[UT Agent] Verifier: FAIL (coverage={coverage}% < {threshold}%)")
                return {
                    "task": "verify_pipeline",
                    "verification_verdict": json.dumps(verdict, ensure_ascii=False),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
        verdict = {
            "result": "PASS",
            "failure_type": None,
            "failure_reason": None,
            "evidence": evidence,
            "failed_job_names": [],
        }
        logger.info(f"[UT Agent] Verifier: PASS (coverage={coverage}%)")
        return {
            "task": "verify_pipeline",
            "verification_verdict": json.dumps(verdict, ensure_ascii=False),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # 5. 其他未知状态
    verdict = {
        "result": "FAIL",
        "failure_type": FAILURE_TYPE_UNKNOWN,
        "failure_reason": f"流水线状态异常: {pipeline_status}",
        "evidence": [feedback.get("message", "")],
        "failed_job_names": failed_job_names,
    }
    logger.info(f"[UT Agent] Verifier: FAIL (unknown, status={pipeline_status})")
    return {
        "task": "verify_pipeline",
        "verification_verdict": json.dumps(verdict, ensure_ascii=False),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def route_after_verify(state: UTAgentState) -> str:
    """Verifier 路由: PASS -> END, FAIL -> plan_fix (未超限) 或 END (超限)。"""
    verdict_raw = state.get("verification_verdict", "")
    fix_iters = state.get("fix_iterations", 0)

    if not verdict_raw:
        return END

    verdict = json.loads(verdict_raw)
    if verdict.get("result") == "PASS":
        return END

    # 超过修复轮次上限，终止
    if fix_iters >= MAX_FIX_ITERATIONS:
        logger.warning(f"[UT Agent] 修复轮次已达上限 ({MAX_FIX_ITERATIONS})，终止")
        return END

    return "plan_fix"


# ──────────────────────────────────────────────────────────────────────────────
# Planner: 根据失败类型制定修复计划
# ──────────────────────────────────────────────────────────────────────────────

def plan_fix(state: UTAgentState) -> dict:
    """
    Planner 节点：根据 Verifier 的 FAIL 结论制定修复计划。

    按 failure_type 分流：
    - build_failure: 定位编译错误，修复源码/CMake 问题
    - test_failure: 分析失败用例日志，修复测试代码
    - coverage_insufficient: 生成补充测试用例
    - pipeline_timeout: 报告超时，建议人工介入
    - unknown: 输出诊断报告
    """
    logger.info(f"[UT Agent] === Task: plan_fix ===")

    verdict = json.loads(state.get("verification_verdict", "{}"))
    failure_type = verdict.get("failure_type", FAILURE_TYPE_UNKNOWN)
    failure_reason = verdict.get("failure_reason", "")
    evidence = verdict.get("evidence", [])
    fix_iters = state.get("fix_iterations", 0)

    logger.info(f"[UT Agent] Planner: failure_type={failure_type}, 迭代={fix_iters+1}/{MAX_FIX_ITERATIONS}")

    # 构建修复任务描述（供 generate_patch 使用）
    fix_task = {
        "mode": "fix",  # 区别于初次生成的 "generate" 模式
        "failure_type": failure_type,
        "failure_reason": failure_reason,
        "evidence": evidence,
        "iteration": fix_iters + 1,
    }

    # 按类型生成具体指令
    if failure_type == FAILURE_TYPE_BUILD:
        fix_task["instructions"] = (
            "编译失败修复:\n"
            "1. 从 evidence 中的编译日志定位 error: 行\n"
            "2. 检查是否是新增测试代码引入的问题(头文件缺失/类型错误/链接问题)\n"
            "3. 修复测试代码中的编译错误，不要修改源码\n"
            "4. 若需修改 CMakeLists.txt，只能修改测试相关 target，且引用的所有文件必须实际存在，禁止引用不存在的文件\n"
            "5. 不允许修改已有的非测试 target（add_library/add_executable 的已有业务目标）"
        )
    elif failure_type == FAILURE_TYPE_TEST:
        fix_task["instructions"] = (
            "测试失败修复:\n"
            "1. 从 evidence 中找到 [FAILED] 的测试用例名\n"
            "2. 分析失败原因(assertion 不匹配/超时/异常)\n"
            "3. 修复测试代码中的逻辑错误\n"
            "4. 如果是 mock 不正确导致的，修正 mock 返回值或行为"
        )
    elif failure_type == FAILURE_TYPE_COVERAGE:
        fix_task["instructions"] = (
            "覆盖率不足补充:\n"
            "1. 分析当前覆盖率与目标差距\n"
            "2. 找到未覆盖的代码分支/路径\n"
            "3. 补充测试用例覆盖缺失的分支\n"
            "4. 优先覆盖错误处理路径和边界条件"
        )
    elif failure_type == FAILURE_TYPE_TIMEOUT:
        fix_task["instructions"] = (
            "流水线超时:\n"
            "无法自动修复，生成诊断报告供人工审查。\n"
            "可能原因: CI runner 负载/网络问题/死循环测试"
        )
    else:
        fix_task["instructions"] = (
            "未知错误:\n"
            "生成诊断报告，记录所有可用证据供人工审查。"
        )

    # 落盘修复计划
    output_dir = ToolContext.output_dir
    mr_id = state["mr_id"]
    fix_plan_path = os.path.join(output_dir, f"mr_{mr_id}", f"fix_plan_iter{fix_iters+1}.json")
    os.makedirs(os.path.dirname(fix_plan_path), exist_ok=True)
    with open(fix_plan_path, "w", encoding="utf-8") as f:
        json.dump(fix_task, f, ensure_ascii=False, indent=2)
    logger.info(f"[UT Agent] 修复计划已落盘: {fix_plan_path}")

    # 发布 MR 评论
    git_provider = ToolContext.git_provider
    if git_provider:
        git_provider.publish_comment(
            f"## UT Agent - 修复计划 (第 {fix_iters+1} 轮) 🔧\n\n"
            f"**失败类型:** `{failure_type}`\n"
            f"**原因:** {failure_reason}\n\n"
            f"**修复策略:**\n```\n{fix_task['instructions']}\n```"
        )

    return {
        "task": "plan_fix",
        "fix_plan": json.dumps(fix_task, ensure_ascii=False),
        "fix_iterations": fix_iters + 1,
        "current_action": f"修复计划已生成: {failure_type}",
        "next_action": "generate_patch (修复模式)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _publish_analysis_comment(content: str, mr_id: int, file_count: int, batch_count: int = None):
    """将分析结果作为评论发布到 MR。"""
    git_provider = ToolContext.git_provider
    if not git_provider:
        return

    if batch_count:
        header = (f"## UT Agent - Diff 分析报告\n\n"
                  f"**MR:** !{mr_id} | **文件数:** {file_count} | **批次:** {batch_count}\n\n")
    else:
        header = (f"## UT Agent - Diff 分析报告\n\n"
                  f"**MR:** !{mr_id} | **文件数:** {file_count}\n\n")

    if not batch_count:
        body = f"{header}```json\n{content}\n```"
    else:
        body = f"{header}{content}"

    git_provider.publish_comment(body)


def build_graph() -> CompiledStateGraph:
    """构建 UT Agent 的 LangGraph 工作流。"""
    workflow = StateGraph(UTAgentState)

    workflow.add_node("collect_mr_info", collect_mr_info)
    workflow.add_node("clone_repo", clone_repo)
    workflow.add_node("analyze_diff", analyze_diff)
    workflow.add_node("generate_test_plan", generate_test_plan)
    workflow.add_node("generate_patch", generate_patch)
    workflow.add_node("validate_plan", validate_plan)
    workflow.add_node("upload_to_gitlab", upload_to_gitlab)
    workflow.add_node("check_pipeline", check_pipeline)
    workflow.add_node("verify_pipeline", verify_pipeline)
    workflow.add_node("plan_fix", plan_fix)

    workflow.add_edge(START, "collect_mr_info")
    workflow.add_edge("collect_mr_info", "clone_repo")
    workflow.add_conditional_edges(
        "clone_repo",
        route_after_clone,
        {"analyze_diff": "analyze_diff", "generate_patch": "generate_patch", "clone_repo": "clone_repo", END: END},
    )
    workflow.add_edge("analyze_diff", "generate_test_plan")
    workflow.add_edge("generate_test_plan", "generate_patch")
    if TEST_MODE:
        # 测试模式：generate_patch 直接跳到 upload_to_gitlab
        workflow.add_edge("generate_patch", "upload_to_gitlab")
    else:
        workflow.add_edge("generate_patch", "validate_plan")
        workflow.add_conditional_edges(
            "validate_plan",
            route_after_validate,
            {"upload_to_gitlab": "upload_to_gitlab", "generate_patch": "generate_patch"},
        )
    workflow.add_edge("upload_to_gitlab", "check_pipeline")
    workflow.add_edge("check_pipeline", "verify_pipeline")
    workflow.add_conditional_edges(
        "verify_pipeline",
        route_after_verify,
        {END: END, "plan_fix": "plan_fix"},
    )
    # plan_fix 之后回到 generate_patch 执行修复
    workflow.add_edge("plan_fix", "generate_patch")

    return workflow.compile()


class UTAgent:
    """UT Agent 的高层接口。"""

    def __init__(self):
        self.graph = build_graph()

    async def run(self, mr_info: UTAgentState) -> str:
        """
        运行 UT Agent，传入 MR 信息，返回要发布为评论的响应字符串。
        """
        mr_info["timestamp"] = datetime.now(timezone.utc).isoformat()
        mr_info.setdefault("clone_attempts", 0)
        mr_info.setdefault("patch_iterations", 0)
        mr_info.setdefault("generated_patches", [])
        result = await self.graph.ainvoke(mr_info)
        return result.get("response", "ERROR: 工作流未生成响应")