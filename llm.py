"""
ut_agent LLM 调用封装 - 基于 litellm 的统一接口，使用独立模型配置。
"""
import litellm
from pr_agent.log import get_logger
from ut_agent.config import MODEL, API_KEY, BASE_URL, DEFAULT_TEMPERATURE


async def call_llm(system: str, user: str, temperature: float = None, model: str = None, max_tokens: int = 16384) -> str:
    """
    调用 LLM 获取回复。

    参数:
        system: system prompt
        user: user prompt
        temperature: 温度参数，默认从 config 读取
        model: 模型名称，默认从 ut_agent.config 读取
        max_tokens: 最大输出 token 数，默认 16384

    返回:
        LLM 回复的文本内容
    """
    if model is None:
        model = MODEL
    if temperature is None:
        temperature = DEFAULT_TEMPERATURE

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    try:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=API_KEY,
            api_base=BASE_URL,
        )
        content = response.choices[0].message.content
        # 检测是否因 token 限制被截断
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            get_logger().warning(f"LLM 输出被截断 (max_tokens={max_tokens})，finish_reason=length")
        return content
    except Exception as e:
        get_logger().error(f"LLM 调用失败: {e}")
        return f"ERROR: LLM 调用失败: {e}"


async def call_llm_with_continuation(
    system: str,
    user: str,
    temperature: float = None,
    model: str = None,
    max_tokens: int = 32000,
    max_continuations: int = 3,
) -> str:
    """
    调用 LLM 并在截断时自动续写（适用于超长 JSON 输出如测试计划）。

    当 finish_reason == "length" 时，将已有输出拼接到 messages 中，
    追加续写指令让 LLM 从断点继续输出，最多续写 max_continuations 次。

    参数:
        system: system prompt
        user: user prompt
        temperature: 温度参数
        model: 模型名称
        max_tokens: 每次调用的最大输出 token 数，默认 32000
        max_continuations: 最大续写次数，默认 3

    返回:
        拼接后的完整文本
    """
    if model is None:
        model = MODEL
    if temperature is None:
        temperature = DEFAULT_TEMPERATURE

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    full_content = ""

    for i in range(1 + max_continuations):
        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=API_KEY,
                api_base=BASE_URL,
            )
            chunk = response.choices[0].message.content or ""
            full_content += chunk
            finish_reason = response.choices[0].finish_reason

            if finish_reason != "length":
                # 正常结束
                break

            # 被截断，准备续写
            get_logger().warning(
                f"LLM 输出截断 (第 {i+1} 段, max_tokens={max_tokens})，尝试续写..."
            )
            # 将已输出内容加入上下文，追加续写指令
            messages.append({"role": "assistant", "content": chunk})
            messages.append({"role": "user", "content": "输出被截断了，请从断点处继续输出剩余的 JSON 内容（不要重复已输出的部分，直接从上次结束的位置继续）："})

        except Exception as e:
            get_logger().error(f"LLM 调用失败 (第 {i+1} 段): {e}")
            if full_content:
                break
            return f"ERROR: LLM 调用失败: {e}"

    return full_content
