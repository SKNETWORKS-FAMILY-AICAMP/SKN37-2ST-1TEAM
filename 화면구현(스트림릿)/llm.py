"""LLM(Anthropic/Gemini/OpenAI/Groq/ollama) 호출을 담당하는 단일 창구 모듈.

이 모듈에만 실제 API 호출 코드를 둔다. 다른 파일(app.py, rules.py)은
ask_llm() / ask_llm_json() / has_api_key() / is_ollama_available() /
list_ollama_models() 함수만 호출해야 한다.

이 모듈은 열 이름/범주값 등 도메인 지식을 전혀 모른다. 프롬프트 문자열은
호출하는 쪽(app.py)에서 config.py 를 참조해 만들어서 넘겨준다.

제공자(provider)는 "anthropic" / "gemini" / "openai" / "groq" / "ollama" 중
하나이며, 인자로 넘기지 않으면 config.LLM_PROVIDER 를 기본값으로 쓴다. 선택된
provider의 라이브러리만 함수 내부에서 import 하므로, 설치되지 않은 다른
provider의 라이브러리 때문에 이 모듈 자체의 import가 깨지지 않는다. groq는
OpenAI 호환(Chat Completions) API라 openai 패키지를 base_url만 바꿔
재사용한다(_call_openai_compatible).

"ollama"는 API 키가 필요 없는 로컬 제공자다 (로컬에서 `ollama serve`로 띄운
서버를 호출). 화면에 노출되는 문구에서 "ollama"라는 이름을 어떻게 표시할지는
app.py의 책임이며, 이 모듈은 내부 식별자로만 "ollama"를 사용한다.
"""

from __future__ import annotations

import json
import logging
import os
import re

from dotenv import load_dotenv

import config

# .env 파일을 읽어 환경변수로 등록한다. override=True로, 이 프로젝트의 .env 값이 항상
# 우선한다. (기존에는 override=False라 OS/Windows 사용자 환경변수에 이미 같은 이름의
# 키가 설정되어 있으면 .env를 바꿔도 무시되는 문제가 있었다 - 로컬 PC에 예전 키가
# 환경변수로 남아 있던 경우 새로 발급한 키가 적용되지 않는 원인이 됐다.)
load_dotenv(override=True)

logger = logging.getLogger(__name__)

# 응답 길이 상한 (너무 길어지는 것을 방지)
MAX_TOKENS = 1024

# provider 별로 확인해야 할 환경변수 이름.
# "ollama"는 API 키 개념이 없는 로컬 제공자라 여기에 포함하지 않는다
# (대신 is_ollama_available()로 서버 실행 여부를 확인한다). has_api_key()는
# 기존 시그니처/동작을 그대로 유지하므로, ollama에 대해서는 항상 False를 반환한다.
ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
}


def _resolve_provider(provider: str | None) -> str:
    """provider가 None이면 config.LLM_PROVIDER를 기본값으로 쓴다."""
    return provider or config.LLM_PROVIDER


def has_api_key(provider: str | None = None) -> bool:
    """해당 provider의 API 키 환경변수 설정 여부만 간단히 확인한다 (UI 안내 문구용)."""
    provider = _resolve_provider(provider)
    env_key = ENV_KEYS.get(provider)
    if not env_key:
        return False
    return bool(os.environ.get(env_key))


def _call_anthropic(system: str, user: str, json_mode: bool = False, model: str | None = None) -> str | None:
    api_key = os.environ.get(ENV_KEYS["anthropic"])
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model or config.LLM_MODELS["anthropic"],
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        text_parts = []
        for block in getattr(response, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                text_parts.append(text)

        result = "".join(text_parts).strip()
        return result if result else None
    except Exception as e:
        # 키 누락, 네트워크 오류, 응답 형식 오류 등 어떤 경우에도 앱이 죽지 않게 None 반환.
        # 키 값은 절대 로그에 남기지 않고 예외 타입만 남긴다.
        logger.warning(f"LLM 호출 실패 (anthropic): {type(e).__name__}")
        return None


def _call_gemini(system: str, user: str, json_mode: bool = False, model: str | None = None) -> str | None:
    api_key = os.environ.get(ENV_KEYS["gemini"])
    if not api_key:
        return None
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return None

    try:
        client = genai.Client(api_key=api_key)
        gen_config_kwargs = {
            "system_instruction": system,
            "max_output_tokens": MAX_TOKENS,
        }
        if json_mode:
            gen_config_kwargs["response_mime_type"] = "application/json"

        response = client.models.generate_content(
            model=model or config.LLM_MODELS["gemini"],
            contents=user,
            config=types.GenerateContentConfig(**gen_config_kwargs),
        )

        result = (getattr(response, "text", None) or "").strip()
        return result if result else None
    except Exception as e:
        logger.warning(f"LLM 호출 실패 (gemini): {type(e).__name__}")
        return None


def _call_openai_compatible(
    base_url: str,
    env_var: str,
    provider_key: str,
    system: str,
    user: str,
    json_mode: bool = False,
    model: str | None = None,
) -> str | None:
    """OpenAI 호환(Chat Completions) API 공통 호출 로직. 지금은 Groq만 쓰지만, base_url/env_var만
    바꾸면 다른 OpenAI 호환 provider(Cerebras, NVIDIA NIM 등)도 같은 방식으로 추가할 수 있다."""
    api_key = os.environ.get(env_var)
    if not api_key:
        return None
    try:
        import openai
    except ImportError:
        return None

    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        extra_kwargs = {}
        if json_mode:
            extra_kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(
            model=model or config.LLM_MODELS[provider_key],
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **extra_kwargs,
        )
        result = (response.choices[0].message.content or "").strip()
        return result if result else None
    except Exception as e:
        logger.warning(f"LLM 호출 실패 ({provider_key}): {type(e).__name__}")
        return None


def _call_groq(system: str, user: str, json_mode: bool = False, model: str | None = None) -> str | None:
    return _call_openai_compatible(
        "https://api.groq.com/openai/v1", ENV_KEYS["groq"], "groq", system, user, json_mode, model
    )


def _call_openai(system: str, user: str, json_mode: bool = False, model: str | None = None) -> str | None:
    api_key = os.environ.get(ENV_KEYS["openai"])
    if not api_key:
        return None
    try:
        import openai
    except ImportError:
        return None

    try:
        client = openai.OpenAI(api_key=api_key)
        extra_kwargs = {}
        if json_mode:
            extra_kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(
            model=model or config.LLM_MODELS["openai"],
            max_completion_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **extra_kwargs,
        )

        result = (response.choices[0].message.content or "").strip()
        return result if result else None
    except Exception as e:
        logger.warning(f"LLM 호출 실패 (openai): {type(e).__name__}")
        return None


# 서버 연결/모델 목록 조회용 짧은 타임아웃 (초). 서버가 꺼져 있을 때 UI가
# 오래 멈춰 있지 않도록 짧게 잡는다. 실제 채팅 호출은 로컬 추론이라 느릴 수
# 있으므로 OLLAMA_CHAT_TIMEOUT_SECONDS를 더 길게 둔다.
OLLAMA_PROBE_TIMEOUT_SECONDS = 2.0
OLLAMA_CHAT_TIMEOUT_SECONDS = 60.0


def is_ollama_available() -> bool:
    """로컬 AI(ollama) 서버가 떠 있는지 확인한다.

    ollama는 API 키 개념이 없으므로, has_api_key()가 아니라 이 함수로
    "사용 가능 여부"를 판단한다. 서버가 꺼져 있거나 라이브러리가 없으면
    짧은 타임아웃 뒤 예외를 삼키고 False를 반환한다 (앱이 멈추지 않도록).
    """
    try:
        import ollama
    except ImportError:
        return False
    try:
        ollama.Client(timeout=OLLAMA_PROBE_TIMEOUT_SECONDS).list()
        return True
    except Exception as e:
        logger.warning(f"로컬 AI(ollama) 서버 확인 실패: {type(e).__name__}")
        return False


def list_ollama_models() -> list[str]:
    """설치된 로컬 AI(ollama) 모델 이름 목록을 반환한다.

    서버가 꺼져 있거나 조회에 실패하면 예외를 삼키고 빈 리스트를 반환한다.
    호출하는 쪽(app.py)은 빈 리스트일 때 config.LLM_MODELS["ollama"] 기본값
    하나만 보여주는 식으로 대체해야 한다.
    """
    try:
        import ollama
    except ImportError:
        return []
    try:
        response = ollama.Client(timeout=OLLAMA_PROBE_TIMEOUT_SECONDS).list()
        return [m.model for m in response.models if getattr(m, "model", None)]
    except Exception as e:
        logger.warning(f"로컬 AI(ollama) 모델 목록 조회 실패: {type(e).__name__}")
        return []


def _call_ollama(system: str, user: str, json_mode: bool = False, model: str | None = None) -> str | None:
    try:
        import ollama
    except ImportError:
        return None

    try:
        client = ollama.Client(timeout=OLLAMA_CHAT_TIMEOUT_SECONDS)
        chat_kwargs = {}
        if json_mode:
            chat_kwargs["format"] = "json"

        response = client.chat(
            model=model or config.LLM_MODELS["ollama"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"num_predict": MAX_TOKENS},
            **chat_kwargs,
        )

        result = (getattr(response.message, "content", None) or "").strip()
        return result if result else None
    except Exception as e:
        # 서버가 꺼져 있거나, 모델이 없거나, 응답 형식이 다른 경우에도 앱이
        # 죽지 않도록 예외를 삼키고 None을 반환한다.
        logger.warning(f"LLM 호출 실패 (ollama): {type(e).__name__}")
        return None


_PROVIDER_CALLERS = {
    "anthropic": _call_anthropic,
    "gemini": _call_gemini,
    "openai": _call_openai,
    "groq": _call_groq,
    "ollama": _call_ollama,
}


def ask_llm(system: str, user: str, provider: str | None = None, model: str | None = None) -> str | None:
    """system/user 프롬프트로 LLM을 호출해 텍스트 응답을 반환한다.

    - provider가 None이면 config.LLM_PROVIDER를 사용한다.
    - model이 None이면 해당 provider의 config.LLM_MODELS 기본값을 사용한다
      (사이드바에서 사용자가 다른 모델을 선택했다면 그 값을 넘겨준다).
    - anthropic/gemini/openai는 해당 provider의 API 키 환경변수가 없으면 None을
      반환하고, ollama는 로컬 서버가 꺼져 있거나 모델이 없으면 None을 반환한다.
    - 호출 중 어떤 예외가 나도 예외를 던지지 않고 None을 반환한다
      (앱이 죽지 않도록 하기 위함. 호출하는 쪽에서 None이면 규칙 기반
      문구 등으로 대체해야 한다).
    """
    provider = _resolve_provider(provider)
    caller = _PROVIDER_CALLERS.get(provider)
    if caller is None:
        return None
    try:
        return caller(system, user, False, model)
    except Exception:
        return None


def _extract_json(text: str | None) -> dict | None:
    """응답 텍스트에서 JSON 객체를 최대한 관대하게 파싱한다."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def ask_llm_json(system: str, user: str, provider: str | None = None, model: str | None = None) -> dict | None:
    """system/user 프롬프트로 LLM을 호출해 JSON(dict) 응답을 반환한다.

    - 가능하면 provider별 JSON 모드를 사용해 호출한다.
    - JSON 모드 호출이 실패하거나 지원되지 않으면 일반 호출로 재시도한다.
    - 응답 텍스트가 바로 JSON이 아니면 텍스트 안의 `{...}` 부분만 추출해 파싱한다.
    - 그래도 실패하면 예외를 던지지 않고 None을 반환한다.
    - model이 None이면 해당 provider의 config.LLM_MODELS 기본값을 사용한다.
    """
    provider = _resolve_provider(provider)
    caller = _PROVIDER_CALLERS.get(provider)
    if caller is None:
        return None

    try:
        text = caller(system, user, True, model)
    except Exception:
        text = None

    parsed = _extract_json(text)
    if parsed is not None:
        return parsed

    try:
        text = caller(system, user, False, model)
    except Exception:
        text = None

    return _extract_json(text)
