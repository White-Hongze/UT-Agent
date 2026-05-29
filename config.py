"""
ut_agent 模型配置加载器 - 从 ut_agent/settings.toml 读取配置。
"""
import os
import tomllib

_SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.toml")

with open(_SETTINGS_PATH, "rb") as f:
    _cfg = tomllib.load(f)

_llm = _cfg.get("llm", {})
_agent = _cfg.get("agent", {})

# 模型配置
MODEL = _llm.get("model", "anthropic/claude-sonnet-4-5-20250929")
FALLBACK_MODEL = _llm.get("fallback_model", "anthropic/claude-haiku-4-5-20251001")
API_KEY = _llm.get("api_key", "")
BASE_URL = _llm.get("base_url", "")
DEFAULT_TEMPERATURE = _llm.get("temperature", 0.2)

# Agent 配置
TEST_MODE = _agent.get("test_mode", False)
