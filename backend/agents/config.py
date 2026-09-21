"""
VEXA M3 — LLM Configuration

Centralised LLM/model configuration for all VEXA agents.

Environment variables:
    VEXA_LLM_MODEL            LLM model string (default: gpt-4o-mini)
    VEXA_LLM_TEMPERATURE      Sampling temperature (default: 0.1)
    VEXA_LLM_API_KEY          API key (overrides provider default, e.g. OPENAI_API_KEY)
    VEXA_LLM_BASE_URL         Optional base URL for custom endpoints / proxies
    VEXA_AGENT_MAX_ITER       Max ReAct iterations per agent (default: 10)
    VEXA_DEBUG_MAX_ITER       Max debug/repair iterations in the debug loop (default: 3)
    VEXA_AGENT_VERBOSE        Enable agent step logging (default: False)

Secrets are read from environment; none are logged.
"""

from __future__ import annotations

import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """LLM and agent configuration — isolated from main Settings to keep separation clean."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="VEXA_",  # All agent settings prefixed with VEXA_
    )

    # -----------------------------------------------------------------------
    # LLM configuration
    # -----------------------------------------------------------------------
    llm_mode: str = Field(
        default="mock",
        description="Execution mode: 'mock' for deterministic tests, 'real' for live LLM.",
    )
    llm_provider: str = Field(
        default="openai",
        description="Provider mapping (e.g. 'openai', 'anthropic', 'mistral', 'gemini', 'groq').",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description=(
            "LiteLLM-compatible model string. Examples: "
            "'gpt-4o-mini', 'gpt-4o', 'anthropic/claude-3-5-haiku-20241022', "
            "'ollama/llama3.2', 'gemini/gemini-1.5-flash'"
        ),
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Lower = more deterministic.",
    )
    llm_api_key: str | None = Field(
        default=None,
        description="Override API key. If None, falls back to provider env vars (OPENAI_API_KEY etc.).",
    )
    llm_base_url: str | None = Field(
        default=None,
        description="Custom base URL for API endpoint (proxy / local model server).",
    )
    llm_max_tokens: int | None = Field(
        default=None,
        description="Maximum response token limit. None = provider default.",
    )
    llm_input_cost_per_1m: float | None = Field(
        default=None,
        description="Cost in USD per 1M input tokens. Example: 0.15 for gpt-4o-mini.",
    )
    llm_output_cost_per_1m: float | None = Field(
        default=None,
        description="Cost in USD per 1M output tokens. Example: 0.60 for gpt-4o-mini.",
    )

    # -----------------------------------------------------------------------
    # Fallback LLM (used when primary hits rate limits or quota)
    # -----------------------------------------------------------------------
    llm_fallback_provider: str | None = Field(
        default=None,
        description="Fallback provider (e.g. 'mistral'). None = no fallback.",
    )
    llm_fallback_model: str | None = Field(
        default=None,
        description="Fallback model string (e.g. 'mistral/mistral-small-latest').",
    )
    llm_fallback_api_key: str | None = Field(
        default=None,
        description="API key for the fallback provider.",
    )

    # -----------------------------------------------------------------------
    # Agent execution controls
    # -----------------------------------------------------------------------
    agent_max_iter: int = Field(
        default=10,
        ge=1,
        description="Maximum ReAct tool-use iterations per agent before stopping.",
    )
    debug_max_iter: int = Field(
        default=3,
        ge=1,
        description="Maximum debug/repair iterations in the Debugger loop.",
    )
    agent_verbose: bool = Field(
        default=False,
        description="Enable verbose step-by-step logging from CrewAI agents.",
    )


def get_agent_settings() -> AgentSettings:
    """Return an AgentSettings instance (not cached — allows test override)."""
    return AgentSettings()


class LLMFactory:
    """
    Validates configuration and instantiates the real LLM for CrewAI.
    Supports automatic failover from primary provider (Groq) to a secondary
    provider (Mistral) when rate limits or daily quota are exceeded.
    """

    @staticmethod
    def build(settings: AgentSettings | None = None) -> Any:
        """
        Build a real CrewAI LLM instance from AgentSettings.
        If a fallback provider is configured, installs a monkey-patch on
        litellm.completion that transparently switches to the fallback when
        the primary provider returns a rate-limit or quota error.
        """
        if settings is None:
            settings = get_agent_settings()

        provider = settings.llm_provider.lower()
        api_key = settings.llm_api_key

        if provider == "openai" and not api_key and not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("VEXA_LLM_API_KEY (or OPENAI_API_KEY) is not configured.")
        elif provider == "anthropic" and not api_key and not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("VEXA_LLM_API_KEY (or ANTHROPIC_API_KEY) is not configured.")
        elif provider == "mistral" and not api_key and not os.environ.get("MISTRAL_API_KEY"):
            raise ValueError("VEXA_LLM_API_KEY (or MISTRAL_API_KEY) is not configured.")
        elif provider == "gemini" and not api_key and not os.environ.get("GEMINI_API_KEY"):
            raise ValueError("VEXA_LLM_API_KEY (or GEMINI_API_KEY) is not configured.")
        elif provider == "groq" and not api_key and not os.environ.get("GROQ_API_KEY"):
            raise ValueError("VEXA_LLM_API_KEY (or GROQ_API_KEY) is not configured.")

        try:
            from crewai import LLM
            import litellm
            from litellm import completion as original_completion
            import time
            import logging

            _log = logging.getLogger(__name__)


            # Configure LiteLLM global retry count
            litellm.num_retries = 3

            # Build fallback LLM kwargs if a fallback is configured
            _fallback_kwargs: dict | None = None
            if settings.llm_fallback_model and settings.llm_fallback_provider:
                _fallback_kwargs = {
                    "model": settings.llm_fallback_model,
                    "temperature": settings.llm_temperature,
                }
                if settings.llm_fallback_api_key:
                    _fallback_kwargs["api_key"] = settings.llm_fallback_api_key
                    if settings.llm_fallback_provider == "mistral":
                        os.environ["MISTRAL_API_KEY"] = settings.llm_fallback_api_key
                _log.info(
                    "LLM fallback configured | primary=%s fallback=%s",
                    settings.llm_model,
                    settings.llm_fallback_model,
                )

            # Track if we have already switched to the fallback in this session
            _state = {"using_fallback": False}

            def rate_limit_completion(*args, **kwargs):
                """
                Wraps litellm.completion to:
                1. Retry TPM rate limits with a 20s sleep (up to 5 times).
                2. Switch to the fallback model permanently if a daily quota
                   (TPD) error is encountered and a fallback is configured.
                """
                # If already switched to fallback, rewrite model in every call
                if _state["using_fallback"] and _fallback_kwargs:
                    kwargs = {**kwargs, **_fallback_kwargs}

                retries = 5
                for attempt in range(retries):
                    try:
                        return original_completion(*args, **kwargs)
                    except Exception as e:
                        err_str = str(e).lower()
                        is_rate_limit = "rate limit" in err_str or "429" in err_str
                        is_quota = "tokens per day" in err_str or "tpd" in err_str

                        if not is_rate_limit and not is_quota:
                            raise e  # Not a rate limit — propagate immediately

                        # Daily quota exhausted → try fallback permanently
                        if is_quota and _fallback_kwargs and not _state["using_fallback"]:
                            _log.warning(
                                "Primary LLM daily quota exhausted. "
                                "Switching permanently to fallback: %s",
                                settings.llm_fallback_model,
                            )
                            _state["using_fallback"] = True
                            kwargs = {**kwargs, **_fallback_kwargs}
                            continue  # Retry immediately with fallback

                        if attempt == retries - 1:
                            raise e  # Exhausted all retries

                        # TPM rate limit → wait and retry same provider
                        wait = 20
                        _log.warning(
                            "Rate limit hit (attempt %d/%d). Sleeping %ds...",
                            attempt + 1, retries, wait,
                        )
                        time.sleep(wait)

            litellm.completion = rate_limit_completion

            # Workaround for CrewAI bug where cache_breakpoint is sent to non-Anthropic providers
            if provider != "anthropic":
                import crewai.llms.cache as _crewai_cache
                _crewai_cache.mark_cache_breakpoint = lambda msg: msg

        except ImportError:
            raise RuntimeError("crewai package is not installed.")

        kwargs: dict = {
            "model": settings.llm_model,
            "temperature": settings.llm_temperature,
            "max_retries": 3,
        }
        if settings.llm_api_key:
            kwargs["api_key"] = settings.llm_api_key
        if settings.llm_base_url:
            kwargs["base_url"] = settings.llm_base_url
        if settings.llm_max_tokens:
            kwargs["max_tokens"] = settings.llm_max_tokens

        # Create the standard LLM instance
        base_llm = LLM(**kwargs)

        # Patch the base_llm.call method directly since monkey-patching litellm
        # doesn't always work if crewai already imported it.
        original_call = base_llm.call
        
        def rate_limit_call(messages, *args, **call_kwargs):
            import time
            import logging
            
            # If already switched to fallback in previous call, we need to ensure the LLM
            # instance properties are updated, but since we can't easily change base_llm's
            # inner state safely, we just rely on the litellm patch if possible, OR
            # we mutate base_llm.model!
            if _state.get("using_fallback") and _fallback_kwargs:
                base_llm.model = _fallback_kwargs["model"]
            
            retries = 5
            for attempt in range(retries):
                try:
                    return original_call(messages, *args, **call_kwargs)
                except Exception as e:
                    err_str = str(e).lower()
                    is_rate_limit = "rate limit" in err_str or "429" in err_str
                    is_quota = "tokens per day" in err_str or "tpd" in err_str

                    if not is_rate_limit and not is_quota:
                        raise e

                    # Daily quota exhausted → try fallback permanently
                    if is_quota and _fallback_kwargs and not _state.get("using_fallback"):
                        logging.getLogger(__name__).warning(
                            "Primary LLM daily quota exhausted. "
                            "Switching permanently to fallback: %s",
                            settings.llm_fallback_model,
                        )
                        _state["using_fallback"] = True
                        base_llm.model = _fallback_kwargs["model"]
                        # CrewAI LLM uses kwargs internally, so this might be tricky,
                        # but litellm wrapper will also catch it if we are lucky.
                        continue  # Retry immediately with fallback

                    if attempt == retries - 1:
                        raise e

                    wait = 20
                    logging.getLogger(__name__).warning(
                        "CrewAI LLM call rate limit hit (attempt %d/%d). Sleeping %ds...",
                        attempt + 1, retries, wait
                    )
                    time.sleep(wait)

        # Override the method on the instance
        base_llm.call = rate_limit_call
        
        return base_llm
