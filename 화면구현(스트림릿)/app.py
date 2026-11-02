"""고객(카드) 이탈 예측 Streamlit 앱.

이 파일은 라우터 역할만 한다. st.set_page_config를 한 번 호출한 뒤, st.navigation으로
페이지 3개(메인/page_home.py, 1. 모델 개발 파이프라인/page_pipeline.py, 2. 고객 이탈
예측 대시보드/아래에서 설명하는 render_dashboard_page)를 사이드바에 "아이콘 + 이름"
목록으로 등록한다 (기본 선택은 "메인", main() 참고). 이 파일의 나머지 설명(화면 구조 등)은
전부 "2. 고객 이탈 예측 대시보드" 페이지 기준이다.

열 이름/범주값/클래스명은 이 파일에 직접 문자열로 쓰지 않는다. 전부 config.py를
import 해서 반복문으로 화면 문구/입력창을 만든다. 데이터가 바뀌어도
config.py / data/ / models/ 세 곳만 바꾸면 이 파일은 그대로 동작해야 한다.

화면 구조(위→아래): 상단 바(고객 번호/모델·조합 선택/프리셋 버튼/AI 조건 입력,
한 줄) -> 핵심 숫자 카드 3개(이탈 확률/위험 등급/데이터 이탈 비율) -> 2열 동일
높이 카드(좌: 이탈 위험 게이지, 우: 이 점수를 만드는 요인) -> 탭 3개(권장 조치 /
조정·비교 시나리오 / 모델 정보) -> 하단 접이식(모델 성능 비교 / 결과서 초안).
AI 챗봇은 하단이 아니라 상단 "AI 챗봇" 버튼을 누르면 @st.dialog 모달 팝업으로
열린다. AI 제공자는 anthropic/gemini/openai/ollama 4개를 사이드바에서 선택한다.
"ollama"는 로컬에서 API 키 없이 띄우는 제공자인데, 화면에 보이는 문구에는
"로컬 AI"로만 표시한다 (코드 내부 식별자는 그대로 "ollama"를 쓴다 - llm.py /
config.py 참고).

모델 입력(104개 변수) 중 config.NUM_COLS(10개)만 슬라이더로 직접 조작하고(이
슬라이더는 "조정·비교 시나리오" 탭에만 있다 - 중복 제거), 나머지 94개는 상단 바의
"기준 고객 선택" UI로 고른 실제 고객의 값을 그대로 사용한다 (build_row_df /
compute_row_shap / recommend_scenarios 등 모델에 값을 넣는 모든 곳이 공통으로
이 104개 열을 채운다).

핵심 숫자 카드/2열 카드는 화면상 탭보다 "위"에 있지만 값은 탭(조정·비교 시나리오)의
슬라이더가 만든 customer_input에 달려 있다. st.tabs는 선택되지 않은 탭도 매
rerun마다 전부 계산하므로, 슬라이더가 있는 탭을 먼저 채워 customer_input을
확정한 뒤 st.empty()로 미리 잡아둔 자리에 숫자 카드/게이지를 그려 넣는다
("요소를 순서 밖에서 넣기" 패턴, main() 참고).

모델은 상단 바의 "모델/조합 선택" 드롭다운(config.COMBOS)으로 고른 1~4개 모델의
predict_proba를 단순 평균해서 쓴다(ComboModel). ExtraTrees는 파일이 커서
필요한 조합이 선택됐을 때만 st.cache_resource로 로드한다.

실행: streamlit run app.py
"""

from __future__ import annotations

import ast
import itertools
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

import config
import llm
import page_home
import page_pipeline
import rules

# --------------------------------------------------------------------------
# AI 제공자/모델 선택 헬퍼 (사이드바에서 선택한 값을 session_state로 공유)
# --------------------------------------------------------------------------

# 화면에 보이는 제공자 이름. "ollama"는 화면 어디에도 "ollama"/"Ollama" 글자가
# 노출되지 않도록 "로컬 AI"로만 표시한다 (내부 식별자는 그대로 "ollama").
PROVIDER_DISPLAY_LABELS = {
    "anthropic": "Anthropic",
    "gemini": "Gemini",
    "openai": "OpenAI",
    "groq": "Groq",
    "ollama": "로컬 AI",
}


def get_selected_provider() -> str:
    """사이드바에서 선택한 LLM 제공자를 session_state에서 읽어온다.

    아직 선택되지 않았으면(사이드바 렌더링 전 등) config.LLM_PROVIDER를 기본값으로 쓴다.
    """
    return st.session_state.get("llm_provider", config.LLM_PROVIDER)


def get_selected_model() -> str | None:
    """사이드바에서 선택한 모델명을 session_state에서 읽어온다.

    아직 선택되지 않았으면 선택된 provider의 config.LLM_MODELS 기본값을 쓴다.
    """
    provider = get_selected_provider()
    return st.session_state.get("llm_model") or config.LLM_MODELS.get(provider)


def ai_is_available(provider: str | None = None) -> bool:
    """해당 provider로 AI 기능을 쓸 수 있는지 확인한다.

    로컬 AI(ollama)는 API 키가 아니라 서버 실행 여부로 판단하므로
    llm.has_api_key() 대신 캐시된 llm.is_ollama_available()을 쓴다.
    """
    provider = provider or get_selected_provider()
    if provider == "ollama":
        return cached_ollama_status()
    return llm.has_api_key(provider)


def ai_unavailable_tag(provider: str | None = None) -> str:
    """버튼/카드에 짧게 붙이는 "AI 사용 불가" 안내 태그.

    provider가 ollama여도 화면 문구에는 "ollama"라는 글자를 쓰지 않는다.
    """
    provider = provider or get_selected_provider()
    if provider == "ollama":
        return "로컬 AI 꺼짐, 기본 문구 사용"
    return "AI 키 없음, 기본 문구 사용"


def ai_unavailable_detail(provider: str | None = None) -> str:
    """사이드바 등에서 조금 더 구체적으로 보여줄 안내 문구."""
    provider = provider or get_selected_provider()
    if provider == "ollama":
        return "로컬 AI 서버가 꺼져 있어 AI 기능을 사용할 수 없습니다."
    env_key_name = llm.ENV_KEYS.get(provider, "")
    return f"{env_key_name} 환경변수가 설정되지 않아 AI 기능을 사용할 수 없습니다."


def ai_call_failed_tag() -> str:
    """키/서버는 있지만 실제 호출이 실패했을 때(네트워크 오류, 모델 오류 등) 붙이는 태그.

    ai_unavailable_tag()("AI 키 없음"/"로컬 AI 꺼짐")는 호출을 시도하기도 전에 키/서버
    상태만으로 판단하는 반면, 이 태그는 llm.ask_llm()/ask_llm_json()을 실제로 호출했지만
    None이 돌아온 경우에만 붙인다 (두 상태를 화면에서 구분하기 위함). 실패 원인(예외 타입)은
    llm.py가 logger.warning으로 이미 남기므로 여기서는 반복하지 않는다.
    """
    return "AI 호출 실패, 기본 문구 사용"


# 로컬 AI 서버 상태/모델 목록 조회는 네트워크 호출이라, 매 rerun마다 부르면
# 서버가 꺼져 있을 때 응답을 기다리느라 화면이 느려진다. 짧은 TTL로 캐시한다.
@st.cache_data(ttl=5, show_spinner=False)
def cached_ollama_status() -> bool:
    return llm.is_ollama_available()


@st.cache_data(ttl=5, show_spinner=False)
def cached_ollama_models() -> list[str]:
    return llm.list_ollama_models()


# --------------------------------------------------------------------------
# 데이터 로드 (캐싱)
# --------------------------------------------------------------------------


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_PATH)
    if config.DROP_COLS:
        df = df.drop(columns=[c for c in config.DROP_COLS if c in df.columns])
    return df


SAMPLE_METRICS = {
    "SampleModel": {
        "accuracy": 0.80,
        "precision": 0.30,
        "recall": 0.55,
        "f1": 0.39,
        "roc_auc": 0.75,
        "pr_auc": 0.35,
        "confusion_matrix": [[700, 150], [50, 100]],
    },
    "best_model": "SampleModel",
}


@st.cache_data
def load_metrics_file():
    if not os.path.exists(config.METRICS_PATH):
        return None
    try:
        with open(config.METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_metrics_data():
    """metrics.json 을 읽어오고, 없으면 샘플 값과 안내 플래그를 함께 반환한다."""
    metrics = load_metrics_file()
    is_sample = metrics is None
    return (metrics if metrics is not None else SAMPLE_METRICS), is_sample


@st.cache_data
def load_combos_file():
    """models/combos.csv (모델 조합 15가지 점수표)를 읽어온다. 없으면 None."""
    if not os.path.exists(config.COMBOS_PATH):
        return None
    try:
        return pd.read_csv(config.COMBOS_PATH)
    except Exception:
        return None


# --------------------------------------------------------------------------
# 모델 로드 / 조합(앙상블) 래퍼
#
# ExtraTrees(ET)는 파일이 약 360MB라, 선택된 조합에 ET가 포함될 때만
# st.cache_resource로 로드한다(다른 키로 다시 호출해도 매번 디스크에서 읽지 않음).
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner="모델 로딩 중...")
def load_pipeline(model_key: str):
    path = os.path.join(config.MODELS_DIR, config.MODEL_FILES[model_key])
    return joblib.load(path)


class ComboModel:
    """선택한 모델 조합(1~4개)의 predict_proba를 단순 평균하는 래퍼.

    predict_proba/recommend_scenarios 등 예측이 필요한 모든 곳에서 이 래퍼
    하나로 동작하면, 조합이 바뀌어도 호출부 코드를 바꿀 필요가 없다.
    """

    def __init__(self, model_keys: list[str]):
        self.model_keys = list(model_keys)
        self._pipelines = [load_pipeline(key) for key in self.model_keys]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = np.mean([p.predict_proba(X)[:, 1] for p in self._pipelines], axis=0)
        return np.column_stack([1 - probs, probs])


def get_selected_combo() -> dict:
    """"모델 정보" 카드에서 선택한 모델 조합(config.COMBOS 항목)을 반환한다."""
    label = st.session_state.get("selected_combo_label")
    for combo in config.COMBOS:
        if combo["label"] == label:
            return combo
    for combo in config.COMBOS:
        if combo["models"] == config.DEFAULT_COMBO_MODELS:
            return combo
    return config.COMBOS[0]


def get_combo_model(model_keys: list[str]) -> ComboModel:
    return ComboModel(model_keys)


def get_shap_model_keys(model_keys: list[str]) -> tuple[list[str], bool]:
    """SHAP 계산에 쓸 모델 키 목록과, ExtraTrees 단독 선택으로 인한 폴백 여부를 반환한다.

    선택한 모델 중 XGB/LGBM/CAT가 하나라도 있으면 그 모델들의 SHAP을 평균해서 쓰고,
    ExtraTrees만 선택된 경우(느려서 직접 SHAP을 계산하지 않음)는
    config.SHAP_FALLBACK_MODEL(XGB) 기준으로 대신 설명한다.
    """
    capable = [k for k in model_keys if k in config.SHAP_FAST_MODELS]
    if capable:
        return capable, False
    return [config.SHAP_FALLBACK_MODEL], True


# --------------------------------------------------------------------------
# 공통 예측 / 위험 등급 헬퍼
# --------------------------------------------------------------------------


def build_row_df(input_dict: dict) -> pd.DataFrame:
    """모델이 요구하는 전체 104개 열(config.FEATURE_COLS) 순서로 1행 DataFrame을 만든다."""
    return pd.DataFrame([{c: input_dict.get(c) for c in config.FEATURE_COLS}])[config.FEATURE_COLS]


def predict_proba(model, input_dict: dict) -> float:
    row = build_row_df(input_dict)
    return float(model.predict_proba(row)[:, 1][0])


def get_base_row_dict(df: pd.DataFrame, idx: int) -> dict:
    """샘플 데이터의 idx번 행에서, 모델 입력에 필요한 104개 열 값을 그대로 꺼낸다."""
    row = df.iloc[int(idx)]
    result = {}
    for col in config.FEATURE_COLS:
        val = row[col]
        result[col] = val.item() if hasattr(val, "item") else val
    return result


def get_risk_level(proba: float) -> str:
    if proba >= config.RISK_HIGH:
        return "고위험"
    if proba >= config.RISK_MID:
        return "중위험"
    return "저위험"


def risk_badge_class(risk_label: str) -> str:
    return {"고위험": "risk-high", "중위험": "risk-mid", "저위험": "risk-low"}.get(risk_label, "risk-low")


def format_value(col: str, value) -> str:
    """config.COL_FORMAT 기준으로 값을 화면 표시용 문자열로 통일한다.

    개월 수·횟수(int)는 정수, 비율·증감율(percent)은 값*100 + 소수 둘째 자리 + "%",
    금액(amount)은 천 단위 콤마로 표시한다. 값이 없거나 변환할 수 없으면 원래 값을
    그대로 문자열로 보여준다.
    """
    if value is None:
        return "-"
    fmt = config.COL_FORMAT.get(col, "int")
    try:
        if fmt == "percent":
            return f"{float(value) * 100:.2f}%"
        if fmt == "amount":
            return f"{float(value):,.0f}"
        return f"{int(round(float(value)))}"
    except (TypeError, ValueError):
        return str(value)


def apply_pending_overrides(df: pd.DataFrame) -> None:
    """"원래대로" / "이 추천을 비교 시나리오에 적용" 버튼이 남겨둔 슬라이더 값을,
    해당 슬라이더 위젯이 만들어지기 전(main()의 맨 앞)에 session_state에 반영한다.

    위젯이 이미 만들어진 뒤에 같은 키의 session_state를 바꾸면 안 되므로(기존
    render_profile_buttons 주석 참고), 버튼 클릭 시에는 이 "pending" 자리에만 값을
    남기고 st.rerun()으로 다음 run에서 이 함수가 먼저 적용하도록 한다.
    """
    pending = st.session_state.pop("pending_slider_overrides", None)
    if not pending:
        return
    for col, val in pending.items():
        if col not in config.NUM_COLS or val is None:
            continue
        is_int = pd.api.types.is_integer_dtype(df[col])
        st.session_state[f"t1_{col}"] = int(round(float(val))) if is_int else float(val)


def make_risk_bar_figure(proba: float) -> go.Figure:
    """이탈 확률을 가로 막대 하나로 보여주고, 40%/70% 기준선을 표시한다."""
    pct = proba * 100
    mid_pct = config.RISK_MID * 100
    high_pct = config.RISK_HIGH * 100

    if proba >= config.RISK_HIGH:
        bar_color = "#d62728"
    elif proba >= config.RISK_MID:
        bar_color = "#e6a800"
    else:
        bar_color = "#2ca02c"

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[pct],
            y=[""],
            orientation="h",
            marker_color=bar_color,
            text=[f"{pct:.1f}%"],
            textposition="outside",
            width=0.55,
            hovertemplate="이탈 확률: %{x:.1f}%<extra></extra>",
        )
    )
    fig.add_vline(x=mid_pct, line_width=2, line_dash="dash", line_color="#c47e0a")
    fig.add_vline(x=high_pct, line_width=2, line_dash="dash", line_color="#d62728")
    fig.add_annotation(x=mid_pct, y=1.35, text=f"중위험 {mid_pct:.0f}%", showarrow=False, font=dict(size=10, color="#c47e0a"))
    fig.add_annotation(x=high_pct, y=1.35, text=f"고위험 {high_pct:.0f}%", showarrow=False, font=dict(size=10, color="#d62728"))
    fig.update_xaxes(range=[0, 100], title_text=None, showgrid=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=140, margin=dict(l=10, r=50, t=40, b=20), showlegend=False)
    return fig


# --------------------------------------------------------------------------
# 입력 위젯 자동 생성 (NUM_COLS -> slider, CAT_COLS -> selectbox)
# --------------------------------------------------------------------------


def get_slider_bounds(col: str, df: pd.DataFrame, original_value=None) -> tuple[float, float]:
    """전체 최솟값~최댓값 대신 1~99 퍼센타일로 슬라이더 범위를 좁힌다.

    다만 선택한 고객의 원래 값이 그 범위 밖(상/하위 1% 극단값)이면 "원래대로" 버튼이
    항상 동작하도록 범위를 그 값까지만 살짝 넓힌다. 데이터 샘플에 변화가 없어
    low == high인 열(예: 전부 0인 열)은 슬라이더가 동작하도록 범위를 넓힌다.
    """
    low = float(df[col].quantile(0.01))
    high = float(df[col].quantile(0.99))
    if original_value is not None:
        low = min(low, float(original_value))
        high = max(high, float(original_value))
    if low == high:
        high = low + max(abs(low), 1.0)
    return low, high


def get_slider_step(col: str, df: pd.DataFrame, low: float, high: float) -> float:
    """값이 좁은 구간(1~75 퍼센타일 폭이 전체 범위의 10% 이하)에 몰린 변수는 step을
    더 세밀하게 줘서 슬라이더로 움직이기 쉽게 한다."""
    span = high - low
    if span <= 0:
        return 0.01
    p25, p75 = float(df[col].quantile(0.25)), float(df[col].quantile(0.75))
    concentrated = (p75 - p25) <= span * 0.1
    step = span / 500 if concentrated else span / 100
    return round(max(step, span / 1000), 4) or 0.0001


def render_num_input(col: str, df: pd.DataFrame, key_prefix: str, original_value=None):
    key = f"{key_prefix}_{col}"
    is_int = pd.api.types.is_integer_dtype(df[col])
    low, high = get_slider_bounds(col, df, original_value)

    if key not in st.session_state:
        default_val = original_value if original_value is not None else df[col].median()
        st.session_state[key] = int(round(default_val)) if is_int else float(default_val)

    label = config.COL_LABEL.get(col, config.COL_DESC.get(col, col))
    help_text = f"{config.COL_DESC.get(col, col)} ({col})"
    if is_int:
        value = st.slider(label, min_value=int(low), max_value=int(high), key=key, help=help_text)
    else:
        step = get_slider_step(col, df, low, high)
        value = st.slider(
            label, min_value=float(low), max_value=float(high), step=step, key=key, help=help_text
        )

    if original_value is not None and value != original_value:
        info_col, btn_col = st.columns([3, 1])
        with info_col:
            st.caption(f"현재: {format_value(col, value)} · 원래 값: {format_value(col, original_value)}")
        with btn_col:
            if st.button("원래대로", key=f"{key}_reset", width="stretch"):
                st.session_state["pending_slider_overrides"] = {col: original_value}
                st.rerun()
    else:
        st.caption(f"현재: {format_value(col, value)}")
    return value


def render_cat_input(col: str, df: pd.DataFrame, key_prefix: str, current_value=None):
    key = f"{key_prefix}_{col}"
    options = sorted(df[col].dropna().unique().tolist())

    if key not in st.session_state:
        st.session_state[key] = current_value if current_value in options else (options[0] if options else None)

    label = f"{config.COL_DESC.get(col, col)} ({col})"
    value = st.selectbox(label, options=options, key=key)
    return value


def _assign_input_value(col: str, raw_value, df: pd.DataFrame, key_prefix: str = "t1") -> bool:
    """입력값 하나를 session_state에 안전하게 반영한다.

    수치형은 데이터 범위로 클램프하고, 범주형은 실제 데이터에 존재하는 값인지
    확인한다 (AI 자연어 입력 적용에서 쓰는 헬퍼).
    """
    if col in config.NUM_COLS:
        try:
            val = float(raw_value)
        except (TypeError, ValueError):
            return False
        col_min, col_max = float(df[col].min()), float(df[col].max())
        val = min(max(val, col_min), col_max)
        if pd.api.types.is_integer_dtype(df[col]):
            val = int(round(val))
        st.session_state[f"{key_prefix}_{col}"] = val
        return True
    if col in config.CAT_COLS:
        options = sorted(df[col].dropna().unique().tolist())
        raw_val = str(raw_value).strip()
        if raw_val in options:
            st.session_state[f"{key_prefix}_{col}"] = raw_val
            return True
        return False
    return False


# --------------------------------------------------------------------------
# SHAP 헬퍼
#
# 선택한 모델 중 SHAP 계산이 빠른 모델(XGB/LGBM/CAT)이 있으면 그 모델들의 SHAP을
# 평균하고, ExtraTrees만 선택된 경우는 느리므로 XGBoost 기준으로 대신 설명한다
# (get_shap_model_keys가 결정). 예측에 쓰는 모델 조합과는 독립적으로 동작한다.
# --------------------------------------------------------------------------

SHAP_SAMPLE_SIZE = 2000


def _extract_positive_class_shap(shap_values):
    """shap 버전/모델 종류에 따라 다른 shap_values 형태를 (n_samples, n_features)로 통일."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[-1] if len(shap_values) > 1 else shap_values[0])
    sv = np.asarray(shap_values)
    if sv.ndim == 3:  # (n_samples, n_features, n_classes)
        return sv[:, :, -1]
    return sv


@st.cache_resource(show_spinner="SHAP 분석 준비 중...")
def build_shap_context(shap_model_keys: tuple, _df: pd.DataFrame):
    sample_df = _df.sample(min(SHAP_SAMPLE_SIZE, len(_df)), random_state=0) if len(_df) > SHAP_SAMPLE_SIZE else _df
    x_full = sample_df[config.FEATURE_COLS]

    explainers = {}
    feature_names = None
    sv_list = []
    for key in shap_model_keys:
        pipe = load_pipeline(key)
        pre = pipe.named_steps["preprocessor"]
        model = pipe.named_steps["classifier"]
        x_t = pre.transform(x_full)
        if feature_names is None:
            feature_names = pre.get_feature_names_out()

        explainer = shap.TreeExplainer(model)
        explainers[key] = explainer
        shap_values = explainer.shap_values(x_t)
        sv_list.append(_extract_positive_class_shap(shap_values))

    sv = np.mean(sv_list, axis=0)
    return {
        "model_keys": tuple(shap_model_keys),
        "explainers": explainers,
        "feature_names": feature_names,
        "shap_values": sv,
    }


def compute_row_shap(shap_ctx: dict, input_dict: dict):
    row_df = build_row_df(input_dict)
    sv_list = []
    for key in shap_ctx["model_keys"]:
        pipe = load_pipeline(key)
        pre = pipe.named_steps["preprocessor"]
        x_row = pre.transform(row_df)
        sv = shap_ctx["explainers"][key].shap_values(x_row)
        sv_list.append(_extract_positive_class_shap(sv))
    sv = np.mean(sv_list, axis=0)
    return sv[0], shap_ctx["feature_names"]


def map_feature_to_original_col(feature_name: str) -> str:
    """ColumnTransformer가 붙인 접두사(예: num__열이름)를 떼고 원래 열 이름으로 매핑.

    config.NUM_COLS(슬라이더 10개)에 없는 열은 전부 "기타 변수"로 묶는다
    (CAT_COLS도 같은 방식으로 지원하되, 이번 데이터는 CAT_COLS가 비어 있음).
    """
    rest = feature_name.split("__", 1)[1] if "__" in feature_name else feature_name
    if rest in config.NUM_COLS:
        return rest
    for col in config.CAT_COLS:
        if rest.startswith(f"{col}_"):
            return col
    return "기타 변수"


def aggregate_shap_by_column(shap_row, feature_names) -> dict:
    agg: dict = {}
    for val, fname in zip(shap_row, feature_names):
        col = map_feature_to_original_col(fname)
        agg[col] = agg.get(col, 0.0) + float(val)
    return agg


def top_shap_factors(shap_row, feature_names, n: int = 3):
    agg = aggregate_shap_by_column(shap_row, feature_names)
    return sorted(agg.items(), key=lambda x: abs(x[1]), reverse=True)[:n]


def top_action_factors(shap_row, feature_names, n: int = 3):
    """"권장 조치" 탭에서 쓰는 순위. "기타 변수"(슬라이더 밖 94개 열 합산)는 rules.py에
    대응하는 조치 문구가 없으므로 제외하고, config.NUM_COLS(슬라이더 10개)만 SHAP
    영향 크기로 정렬한다. 상위 n개(카드용)와 나머지(접이식용)를 나눠서 반환한다.
    """
    agg = aggregate_shap_by_column(shap_row, feature_names)
    agg = {col: val for col, val in agg.items() if col in config.NUM_COLS}
    ranked = sorted(agg.items(), key=lambda x: abs(x[1]), reverse=True)
    return ranked[:n], ranked[n:]


def overall_importance(shap_ctx: dict) -> dict:
    sv = shap_ctx["shap_values"]
    feature_names = shap_ctx["feature_names"]
    mean_abs = np.abs(sv).mean(axis=0)
    agg: dict = {}
    for val, fname in zip(mean_abs, feature_names):
        col = map_feature_to_original_col(fname)
        agg[col] = agg.get(col, 0.0) + float(val)
    return agg


# --------------------------------------------------------------------------
# AI 자연어 입력 / AI 결과 설명 헬퍼
# --------------------------------------------------------------------------


def _parse_json_loose(text: str):
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


def extract_values_from_text(nl_text: str, df: pd.DataFrame):
    col_lines = []
    for col in config.NUM_COLS:
        col_min, col_max = df[col].min(), df[col].max()
        col_lines.append(f'- "{col}": {config.COL_DESC.get(col, col)} (숫자, 대략 {col_min}~{col_max} 범위)')
    for col in config.CAT_COLS:
        options = sorted(df[col].dropna().unique().tolist())
        col_lines.append(f'- "{col}": {config.COL_DESC.get(col, col)} (다음 중 하나: {", ".join(map(str, options))})')

    system = (
        "너는 고객 설명 문장에서 항목 값을 추출하는 도우미야. 설명이나 마크다운 없이 "
        "JSON 객체 하나만 출력해. 문장에 언급되지 않은 항목은 JSON에 포함하지 마."
    )
    user = "다음 항목들의 값을 고객 설명에서 추출해서 JSON으로 답해줘.\n\n항목 목록:\n" + "\n".join(
        col_lines
    ) + f"\n\n고객 설명: {nl_text}"

    provider = get_selected_provider()
    response = llm.ask_llm(system, user, provider=provider, model=get_selected_model())
    if not response:
        return None
    return _parse_json_loose(response)


def apply_parsed_values(parsed: dict, df: pd.DataFrame) -> bool:
    applied_any = False
    for col in config.NUM_COLS + config.CAT_COLS:
        if col in parsed and _assign_input_value(col, parsed[col], df, key_prefix="t1"):
            applied_any = True
    return applied_any


def explain_prediction(top_factors, proba: float, risk_label: str) -> str:
    provider = get_selected_provider()
    factor_names = ", ".join(config.COL_DESC.get(c, c) for c, _ in top_factors)
    if not ai_is_available(provider):
        return f"[{ai_unavailable_tag(provider)}] 주요 영향 요인: {factor_names}"

    factor_text = ", ".join(
        f"{config.COL_DESC.get(c, c)}({'위험 증가' if v > 0 else '위험 감소'} 방향)" for c, v in top_factors
    )
    reply = llm.ask_llm(
        "너는 이탈 예측 결과를 쉽게 설명하는 상담사야. 2~3문장으로 간결하게 한국어로 설명해.",
        f"예측된 이탈 확률은 {proba * 100:.1f}%이고 위험 등급은 {risk_label}입니다. "
        f"주요 영향 요인: {factor_text}. 이 고객의 상황을 쉽게 설명해줘.",
        provider=provider,
        model=get_selected_model(),
    )
    if reply:
        return reply
    return f"[{ai_call_failed_tag()}] 주요 영향 요인: {factor_names}"


# --------------------------------------------------------------------------
# 가정 시나리오 시뮬레이션 헬퍼
# --------------------------------------------------------------------------


@st.cache_data
def numeric_percentiles(df: pd.DataFrame) -> dict:
    result = {}
    for col in config.NUM_COLS:
        raw = {
            "p25": df[col].quantile(0.25),
            "p50": df[col].median(),
            "p75": df[col].quantile(0.75),
        }
        if pd.api.types.is_integer_dtype(df[col]):
            result[col] = {k: int(round(v)) for k, v in raw.items()}
        else:
            result[col] = {k: float(v) for k, v in raw.items()}
    return result


def recommend_scenarios(model, base_input: dict, percentiles: dict, top_n: int = 3):
    """슬라이더 10개 변수를 한 번에 하나씩 p25/p50/p75로 바꿔보고, 확률이 가장 많이
    떨어지는 변경을 추천한다 (변수가 10개라 전체 조합을 다 만들면 수가 너무 커지므로,
    변수 하나씩 바꾸는 방식으로 계산량을 억제한다. 나머지 94개 변수는 base_input의
    실제 고객 값을 그대로 둔다)."""
    base_proba = predict_proba(model, base_input)

    candidate_rows = []
    for col in config.NUM_COLS:
        levels = sorted({percentiles[col]["p25"], percentiles[col]["p50"], percentiles[col]["p75"]})
        for level in levels:
            if level == base_input.get(col):
                continue
            row = dict(base_input)
            row[col] = level
            candidate_rows.append((col, level, row))

    if not candidate_rows:
        return [], base_proba

    batch_df = pd.DataFrame([row for _, _, row in candidate_rows])[config.FEATURE_COLS]
    probs = model.predict_proba(batch_df)[:, 1]

    candidates = []
    for (col, level, _row), proba in zip(candidate_rows, probs):
        changes = {col: level}
        candidates.append((changes, float(proba), base_proba - float(proba)))

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[:top_n], base_proba


def fallback_scenario_text(changes: dict) -> str:
    parts = [f"{config.COL_DESC.get(c, c)}을(를) {format_value(c, v)}" for c, v in changes.items()]
    return ", ".join(parts) + "로 바꾸면 효과적입니다."


def scenario_description(changes: dict, base_proba: float, new_proba: float) -> str:
    provider = get_selected_provider()
    if ai_is_available(provider):
        change_text = ", ".join(f"{config.COL_DESC.get(c, c)}을(를) {format_value(c, v)}로" for c, v in changes.items())
        reply = llm.ask_llm(
            "너는 고객 이탈 방지 컨설턴트야. 한 문장으로 간결하게 조언해.",
            f"{change_text} 바꾸면 이탈 확률이 {base_proba * 100:.1f}%에서 {new_proba * 100:.1f}%로 "
            "낮아집니다. 왜 효과적인지 한 문장으로 설명해줘.",
            provider=provider,
            model=get_selected_model(),
        )
        if reply:
            return reply
    return fallback_scenario_text(changes)


# --------------------------------------------------------------------------
# 해결 방안 헬퍼
# --------------------------------------------------------------------------


def generate_personalized_solution(customer_input: dict, predict_result: dict | None) -> str:
    provider = get_selected_provider()
    if not ai_is_available(provider):
        return f"[{ai_unavailable_tag(provider)}]\n" + rules.build_strategy_text(customer_input)

    risk_label = predict_result.get("risk_label") if predict_result else "알 수 없음"
    top_factors = predict_result.get("top_factors", []) if predict_result else []
    factor_text = ", ".join(config.COL_DESC.get(c, c) for c, _ in top_factors)
    input_text = ", ".join(f"{config.COL_DESC.get(c, c)}={customer_input.get(c)}" for c in config.NUM_COLS)

    reply = llm.ask_llm(
        "너는 고객 이탈 방지 전략을 제안하는 CRM 컨설턴트야. 고객 정보를 참고해 실행 가능한 "
        "맞춤 유지 전략을 3~4문장으로 한국어로 제안해.",
        f"고객 정보: {input_text}\n위험 등급: {risk_label}\n주요 영향 요인: {factor_text}",
        provider=provider,
        model=get_selected_model(),
    )
    if reply:
        return reply
    return f"[{ai_call_failed_tag()}]\n" + rules.build_strategy_text(customer_input)


def build_customer_report(customer_input: dict, predict_result: dict | None) -> str:
    lines = ["=== 고객 이탈 분석 요약 리포트 ===", ""]
    customer_idx = st.session_state.get("selected_customer_idx")
    if customer_idx is not None:
        lines.append(f"기준 고객 번호: {customer_idx} (나머지 94개 변수는 이 고객의 실제 값 사용)")
        lines.append("")

    lines.append("[고객 입력값(슬라이더)]")
    for col in config.NUM_COLS:
        lines.append(f"- {config.COL_DESC.get(col, col)}: {format_value(col, customer_input.get(col))}")

    lines.append("")
    if predict_result:
        lines.append("[예측 결과]")
        lines.append(f"- 이탈 확률: {predict_result.get('proba', 0) * 100:.1f}%")
        lines.append(f"- 위험 등급: {predict_result.get('risk_label', '-')}")
        top_factors = predict_result.get("top_factors", [])
        if top_factors:
            factor_text = ", ".join(config.COL_DESC.get(c, c) for c, _ in top_factors)
            lines.append(f"- 주요 영향 요인: {factor_text}")
        lines.append("")

    lines.append("[유지 전략]")
    solution_text = st.session_state.get("solution_text") or rules.build_strategy_text(customer_input)
    lines.append(solution_text)

    return "\n".join(lines)


# --------------------------------------------------------------------------
# AI 챗봇 헬퍼 (pandas 코드 샌드박스 실행 포함)
#
# 보안 검토를 통과한 로직이므로 아래 샌드박스 관련 함수/상수는 내용을 바꾸지
# 않고 그대로 재사용한다 (위치만 하단 expander로 옮김).
# --------------------------------------------------------------------------

DATA_QUESTION_KEYWORDS = [
    "평균", "최대", "최소", "비율", "몇 명", "몇명", "합계", "개수", "분포",
    "가장 높은", "가장 낮은", "통계", "표로", "그룹별", "상관",
]

# exec() 안에서 쓸 수 있는 내장 함수(안전한 조회/집계용). 파일/네트워크/시스템 접근
# 함수(open, __import__, eval, exec 등)는 포함하지 않는다.
SAFE_BUILTINS = {
    "len": len, "round": round, "min": min, "max": max, "sum": sum,
    "abs": abs, "sorted": sorted, "range": range, "list": list,
    "dict": dict, "float": float, "int": int, "str": str, "bool": bool,
    "enumerate": enumerate, "zip": zip,
}

# 전역 함수 호출(예: len(df))로 허용하는 이름. SAFE_BUILTINS 와 동일한 키 집합이어야 한다.
ALLOWED_CALL_NAMES = set(SAFE_BUILTINS.keys())

# df에서 허용하는 속성/메서드(조회·집계만). read_*/to_*(파일/네트워크 I/O), pickle,
# os/sys/subprocess 등 모듈 접근은 이 목록에 없으므로 전부 거부된다.
ALLOWED_ATTRS = {
    "groupby", "mean", "sum", "count", "value_counts", "sort_values", "head",
    "tail", "describe", "corr", "loc", "iloc", "query", "shape", "columns",
    "nunique", "min", "max", "std", "median", "round", "reset_index",
    "sort_index", "unique", "isna", "notna", "size", "agg", "index", "values",
    "dtypes",
}

# ThreadPoolExecutor 기반 실행 시간 제한 (Windows에서는 signal.alarm 을 쓸 수 없음)
CODE_TIMEOUT_SECONDS = 5


class UnsafeCodeError(Exception):
    """AST 화이트리스트 검증에서 거부된 코드."""


def _validate_query_arg(call_node: ast.Call) -> None:
    """df.query(...) 는 내부적으로 문자열을 다시 평가(eval)하므로 상수 문자열만 허용하고,
    그 문자열 안에도 위험한 패턴(로컬변수 참조 @, 밑줄, import/lambda/eval/exec)이
    없는지 추가로 확인한다."""
    if not call_node.args or not isinstance(call_node.args[0], ast.Constant) or not isinstance(
        call_node.args[0].value, str
    ):
        raise UnsafeCodeError("query()는 문자열 리터럴만 허용됩니다.")
    query_str = call_node.args[0].value
    for bad in ("@", "__", "import", "lambda", "exec", "eval"):
        if bad in query_str:
            raise UnsafeCodeError("query() 문자열에 허용되지 않는 패턴이 포함되어 있습니다.")


def validate_pandas_code_ast(code: str) -> None:
    """ast로 코드를 파싱해 화이트리스트에 없는 구문/이름/속성/호출을 모두 거부한다.

    거부 대상: import, exec/eval/open 등 임의 함수 호출, 함수/람다/클래스 정의,
    del/global/nonlocal, 밑줄로 시작하는 이름·속성(__xxx__ 포함), 속성 대입/삭제,
    ALLOWED_ATTRS에 없는 메서드·속성 접근(= read_*/to_*/pd/os/sys/subprocess 등
    모듈 접근 포함), ALLOWED_CALL_NAMES에 없는 전역 함수 호출.
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise UnsafeCodeError(f"코드를 파싱할 수 없습니다: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCodeError("import는 허용되지 않습니다.")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            raise UnsafeCodeError("함수/클래스/람다 정의는 허용되지 않습니다.")
        if isinstance(node, (ast.With, ast.AsyncWith)):
            raise UnsafeCodeError("with 문은 허용되지 않습니다.")
        if isinstance(node, ast.Delete):
            raise UnsafeCodeError("삭제(del) 문은 허용되지 않습니다.")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise UnsafeCodeError("global/nonlocal은 허용되지 않습니다.")
        if isinstance(node, ast.Name):
            if node.id.startswith("_"):
                raise UnsafeCodeError(f"허용되지 않은 이름입니다: {node.id}")
        if isinstance(node, ast.Attribute):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                raise UnsafeCodeError(f"속성에 값을 대입/삭제할 수 없습니다: {node.attr}")
            if node.attr.startswith("_") or node.attr not in ALLOWED_ATTRS:
                raise UnsafeCodeError(f"허용되지 않은 속성/메서드 접근입니다: {node.attr}")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id not in ALLOWED_CALL_NAMES:
                    raise UnsafeCodeError(f"허용되지 않은 함수 호출입니다: {func.id}")
            elif isinstance(func, ast.Attribute):
                # func.attr 자체는 위 ast.Attribute 분기에서 이미 화이트리스트 검증됨
                if func.attr == "query":
                    _validate_query_arg(node)
            else:
                raise UnsafeCodeError("허용되지 않은 형태의 함수 호출입니다.")


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def run_pandas_code(code: str, df: pd.DataFrame):
    """생성된 pandas 코드를 AST 화이트리스트로 검증한 뒤, 격리된 네임스페이스/별도
    스레드에서 시간 제한을 두고 실행한다.

    - ast.parse + ast.walk 로 import/exec/eval/open, 밑줄로 시작하는 속성(__xxx__), 화이트리스트
      밖 메서드·함수 호출을 모두 거부한다 (단순 부분 문자열 블랙리스트 대신 사용).
    - exec() 전역에는 pd 모듈을 넘기지 않는다. df.copy() 를 'df' 라는 이름으로만 노출해
      pd.io.common.os 같은 속성 체인으로 os 모듈에 접근하는 것을 원천적으로 막는다.
    - exec(code, namespace, namespace) 처럼 전역/지역을 분리하지 않고 네임스페이스 하나만
      사용한다 (globals/locals 분리로 인한 NameError 방지).
    - Windows에는 signal.alarm이 없으므로 ThreadPoolExecutor + future.result(timeout=...)
      로 실행 시간을 제한한다. 시간 초과 시 "코드 실행 시간 초과"를 안내한다. (스레드를
      강제로 죽일 수는 없어, 시간 초과 후에도 해당 스레드 자체는 백그라운드에서 끝까지
      실행될 수 있다는 한계가 있다 - 데모 앱 범위에서는 감내 가능한 트레이드오프.)
    """
    try:
        validate_pandas_code_ast(code)
    except UnsafeCodeError as e:
        return None, f"안전하지 않은 코드가 감지되어 실행을 거부했습니다: {e}"

    namespace: dict = {"__builtins__": SAFE_BUILTINS, "df": df.copy()}

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(exec, code, namespace, namespace)
        try:
            future.result(timeout=CODE_TIMEOUT_SECONDS)
        except FutureTimeoutError:
            return None, "코드 실행 시간 초과"
        except Exception as e:
            return None, f"코드 실행 중 오류가 발생했습니다: {e}"
    finally:
        executor.shutdown(wait=False)

    result = namespace.get("result")
    if result is None:
        return None, "코드가 result 변수를 만들지 않았습니다."
    return result, None


def classify_data_question(user_msg: str) -> bool:
    provider = get_selected_provider()
    result = llm.ask_llm(
        "다음 사용자 질문이 데이터(표, 통계, 집계)에 대한 질문이면 'DATA', "
        "일반적인 대화/상담 질문이면 'CHAT'이라고 한 단어로만 답해.",
        user_msg,
        provider=provider,
        model=get_selected_model(),
    )
    if result:
        return "DATA" in result.upper()
    return any(kw in user_msg for kw in DATA_QUESTION_KEYWORDS)


def generate_pandas_code(user_msg: str):
    col_lines = [f"- {c}: {config.COL_DESC.get(c, c)}" for c in config.NUM_COLS + config.CAT_COLS + [config.TARGET]]
    system = (
        "너는 pandas 코드 생성기야. 이미 df 라는 pandas DataFrame 변수가 있다고 가정해. "
        "질문에 답하는 pandas 코드만 생성해. import, 파일 입출력(open/to_csv 등), 네트워크 호출, "
        "삭제(del) 등은 절대 쓰지 마. 최종 답은 반드시 result 라는 변수에 담아. 코드 외 설명이나 "
        "마크다운 표시 없이 코드만 출력해."
    )
    user = "df 열 설명:\n" + "\n".join(col_lines) + f"\n\n질문: {user_msg}\npandas 코드만 출력해."
    provider = get_selected_provider()
    code = llm.ask_llm(system, user, provider=provider, model=get_selected_model())
    if not code:
        return None
    return _strip_code_fence(code)


def build_prediction_context_text() -> str:
    customer_input = st.session_state.get("customer_input")
    predict_result = st.session_state.get("predict_result")
    if not customer_input or not predict_result:
        return "아직 예측을 실행하지 않았습니다."
    parts = [f"{config.COL_DESC.get(c, c)}={customer_input.get(c)}" for c in config.NUM_COLS]
    return (
        ", ".join(parts)
        + f" / 예측 이탈 확률={predict_result.get('proba', 0) * 100:.1f}%"
        + f" / 위험등급={predict_result.get('risk_label', '-')}"
    )


def handle_chat_message(user_msg: str, df: pd.DataFrame) -> str:
    provider = get_selected_provider()
    if not ai_is_available(provider):
        return f"[{ai_unavailable_tag(provider)}] {ai_unavailable_detail(provider)}"

    try:
        is_data_question = classify_data_question(user_msg)
    except Exception:
        is_data_question = any(kw in user_msg for kw in DATA_QUESTION_KEYWORDS)

    if is_data_question:
        code = generate_pandas_code(user_msg)
        if code is None:
            return "AI 코드 생성에 실패했습니다. 잠시 후 다시 시도해 주세요."
        result, err = run_pandas_code(code, df)
        if err:
            return err
        explanation = llm.ask_llm(
            "너는 데이터 분석 결과를 쉽게 설명하는 도우미야. 한국어로 간단히 설명해.",
            f"질문: {user_msg}\n계산 결과: {str(result)[:1500]}",
            provider=provider,
            model=get_selected_model(),
        )
        return explanation if explanation else f"계산 결과: {str(result)[:1500]}"

    context = build_prediction_context_text()
    reply = llm.ask_llm(
        "너는 고객 이탈 예측 대시보드의 AI 상담 챗봇이야. 친절하고 간결하게 한국어로 답해.",
        f"현재 예측 상황: {context}\n\n사용자 질문: {user_msg}",
        provider=provider,
        model=get_selected_model(),
    )
    if reply:
        return reply
    return f"[{ai_call_failed_tag()}] 잠시 후 다시 시도해 주세요."


# --------------------------------------------------------------------------
# 결과서 초안 헬퍼
# --------------------------------------------------------------------------


def fallback_report_paragraph(best_model, best_metrics: dict, top_cols: list, is_sample: bool) -> str:
    desc_list = ", ".join(config.COL_DESC.get(c, c) for c in top_cols)
    note = "(참고: metrics.json이 없어 샘플 값입니다) " if is_sample else ""
    recall = best_metrics.get("recall", 0)
    f1 = best_metrics.get("f1", 0)
    pr_auc = best_metrics.get("pr_auc", 0)
    return (
        f"{note}선택된 모델은 {best_model}이며, 재현율(recall) {recall * 100:.1f}%, "
        f"F1 점수 {f1:.3f}, PR-AUC {pr_auc:.3f}을 기록했습니다. "
        f"전체 데이터에서 이탈 예측에 가장 큰 영향을 준 변수는 {desc_list}입니다. "
        "이 지표를 바탕으로 고위험 고객을 조기에 식별하고, 주요 영향 변수에 맞는 유지 "
        "전략을 우선적으로 적용하는 것을 권장합니다."
    )


def generate_report_paragraph(best_model, best_metrics: dict, top_cols: list, is_sample: bool) -> str:
    fallback = fallback_report_paragraph(best_model, best_metrics, top_cols, is_sample)
    provider = get_selected_provider()
    if not ai_is_available(provider):
        return f"[{ai_unavailable_tag(provider)}]\n" + fallback

    metrics_text = ", ".join(
        f"{k}={v:.3f}" for k, v in best_metrics.items() if k not in ("confusion_matrix", "file")
    )
    factor_text = ", ".join(config.COL_DESC.get(c, c) for c in top_cols)
    reply = llm.ask_llm(
        "너는 머신러닝 모델 성능 보고서를 작성하는 데이터 분석가야. 아래 정보를 바탕으로 "
        "비전문가도 이해할 수 있는 한국어 해석 문단을 3~5문장으로 작성해.",
        f"선택된 모델: {best_model}\n성능 지표: {metrics_text}\n주요 영향 변수: {factor_text}",
        provider=provider,
        model=get_selected_model(),
    )
    if reply:
        return reply
    return f"[{ai_call_failed_tag()}]\n" + fallback


def model_info_summary(df: pd.DataFrame, metrics: dict, best_model: str | None) -> dict:
    """모델 정보 카드에 쓸 값들을 모은다.

    best 모델의 confusion_matrix 합계를 "평가(테스트) 건수"로 보여준다. 이 값은
    ml-engineer가 모델을 학습/평가할 때 쓴 원본 전체 데이터 기준이고, 지금 앱이
    쓰는 data/sample_churn.csv는 화면 시연용 샘플이라 행 수(total_rows)가 훨씬
    적을 수 있다. 두 수는 같은 모집단이 아니므로 "학습 건수 = 전체 행수 - 평가
    건수" 같은 뺄셈은 하지 않는다 (예전 구조는 두 값이 같은 데이터였어서 뺄셈이
    성립했지만, 지금은 평가 건수가 샘플 행 수보다 더 커서 음수가 나올 수 있음).
    """
    total_rows = len(df)
    best_metrics = metrics.get(best_model, {}) if best_model else {}
    cm = best_metrics.get("confusion_matrix")
    eval_count = sum(sum(row) for row in cm) if cm else None
    churn_rate = float(df[config.TARGET].mean()) if config.TARGET in df.columns else None
    return {
        "best_model": best_model,
        "eval_count": eval_count,
        "total_rows": total_rows,
        "churn_rate": churn_rate,
        "metrics": best_metrics,
    }


# --------------------------------------------------------------------------
# 사이드바
# --------------------------------------------------------------------------


def provider_status_label(provider: str) -> str:
    if provider == "ollama":
        return "✅ 실행 중" if cached_ollama_status() else "❌ 꺼져 있음"
    return "✅ 키 있음" if llm.has_api_key(provider) else "❌ 키 없음"


def render_combo_selector():
    """모델/조합 선택 드롭다운. 사이드바가 아니라 "모델 정보" 카드 안에서 보여준다."""
    labels = [combo["label"] for combo in config.COMBOS]
    if not labels:
        st.caption("config.COMBOS 에 등록된 모델 조합이 없습니다.")
        return

    default_label = next(
        (combo["label"] for combo in config.COMBOS if combo["models"] == config.DEFAULT_COMBO_MODELS),
        labels[0],
    )
    if st.session_state.get("selected_combo_label") not in labels:
        st.session_state["selected_combo_label"] = default_label

    st.selectbox("모델/조합 선택", options=labels, key="selected_combo_label")
    st.caption("ExtraTrees가 포함된 조합은 처음 선택할 때 모델 로딩에 시간이 걸릴 수 있습니다.")


def render_sidebar():
    """AI 설정 사이드바. "AI 제공자 선택"/"모델 선택"은 st.navigation 페이지 전환 시
    위젯에 바인딩된 session_state(key="llm_provider_widget" 등)가 초기화되므로,
    실제 선택값은 위젯과 다른 이름의 일반 session_state 키("llm_provider"/"llm_model",
    get_selected_provider()/get_selected_model()이 읽는 키)에 따로 저장해 페이지를
    옮겨 다녀도 유지되게 한다. 위젯은 매번 그 값으로 다시 초기화(index=)해서 그린다.
    """
    st.sidebar.header("AI 설정")

    provider_keys = list(config.LLM_MODELS.keys())
    default_provider = config.LLM_PROVIDER if config.LLM_PROVIDER in provider_keys else provider_keys[0]
    stored_provider = st.session_state.get("llm_provider", default_provider)
    if stored_provider not in provider_keys:
        stored_provider = default_provider

    selected_provider = st.sidebar.selectbox(
        "AI 제공자 선택",
        options=provider_keys,
        index=provider_keys.index(stored_provider),
        format_func=lambda p: PROVIDER_DISPLAY_LABELS.get(p, p),
        key="llm_provider_widget",
    )
    st.session_state["llm_provider"] = selected_provider

    st.sidebar.caption("제공자별 상태")
    for p in provider_keys:
        st.sidebar.write(f"{PROVIDER_DISPLAY_LABELS.get(p, p)} · {provider_status_label(p)}")

    st.sidebar.divider()
    st.sidebar.caption("모델")
    if selected_provider == "ollama":
        models = cached_ollama_models()
        if models:
            default_model = config.LLM_MODELS.get("ollama")
            stored_model = st.session_state.get("llm_model", default_model)
            if stored_model not in models:
                stored_model = default_model if default_model in models else models[0]
            selected_model = st.sidebar.selectbox(
                "모델 선택", options=models, index=models.index(stored_model), key="llm_model_widget"
            )
            st.session_state["llm_model"] = selected_model
        else:
            st.session_state["llm_model"] = config.LLM_MODELS.get("ollama")
            st.sidebar.caption(f"모델: {config.LLM_MODELS.get('ollama')} (목록을 가져올 수 없어 기본값 사용)")
    else:
        st.session_state["llm_model"] = config.LLM_MODELS.get(selected_provider)
        st.sidebar.caption(f"모델: {config.LLM_MODELS.get(selected_provider)}")

    st.sidebar.divider()
    if not ai_is_available(selected_provider):
        st.sidebar.warning(f"{ai_unavailable_tag(selected_provider)}\n\n{ai_unavailable_detail(selected_provider)}")
    else:
        st.sidebar.success("AI 기능 사용 가능")


# --------------------------------------------------------------------------
# 좌측 입력 패널
# --------------------------------------------------------------------------


def render_customer_selector(df: pd.DataFrame) -> int:
    """샘플 데이터에서 "기준 고객"을 하나 고르는 UI.

    슬라이더 10개(config.NUM_COLS) 외 나머지 94개 변수는 전부 여기서 고른 고객의
    실제 값을 그대로 모델에 넣는다. 고객이 바뀌면 슬라이더 값도 그 고객의 실제
    값으로 다시 초기화한다.
    """
    max_idx = len(df) - 1
    if "selected_customer_idx" not in st.session_state:
        st.session_state["selected_customer_idx"] = 0

    st.caption("기준 고객 번호")
    idx = st.number_input(
        f"고객 번호 (0~{max_idx})",
        min_value=0,
        max_value=max_idx,
        step=1,
        key="selected_customer_idx",
        label_visibility="collapsed",
    )

    prev_idx = st.session_state.get("_prev_customer_idx")
    if prev_idx != idx:
        for col in config.NUM_COLS:
            st.session_state.pop(f"t1_{col}", None)
        st.session_state["_prev_customer_idx"] = idx
    return int(idx)


def render_profile_buttons(df: pd.DataFrame):
    if not config.PROFILES:
        return
    st.caption("프로필 예시")
    cols_ui = st.columns(len(config.PROFILES))
    for col_ui, (name, values) in zip(cols_ui, config.PROFILES.items()):
        with col_ui:
            if st.button(name, key=f"profile_btn_{name}", width="stretch"):
                row_index = values.get("row_index")
                if row_index is not None and 0 <= row_index <= len(df) - 1:
                    # customer selector 위젯이 이미 만들어지기 전에 session_state를
                    # 바꿔야 하므로, render_profile_buttons는 render_customer_selector
                    # 보다 먼저 호출되어야 한다 (render_top_bar 참고).
                    st.session_state["selected_customer_idx"] = int(row_index)


def render_nl_fill(df: pd.DataFrame):
    st.caption("AI로 조건 입력")
    nl_text = st.text_input(
        "고객 설명",
        key="t1_nl_text",
        placeholder="예: 최근 6개월 상담 3번, 혜택수혜율 20% 수준인 고객",
        label_visibility="collapsed",
    )
    if st.button("적용 하기", key="t1_ai_fill_btn"):
        provider = get_selected_provider()
        if not ai_is_available(provider):
            st.info(f"{ai_unavailable_tag(provider)}: 자연어 자동 입력 기능을 사용할 수 없습니다. 직접 입력해 주세요.")
        elif not nl_text.strip():
            st.warning("고객 설명을 입력해 주세요.")
        else:
            parsed = extract_values_from_text(nl_text, df)
            if parsed is None:
                st.warning("AI 응답을 해석할 수 없습니다. 직접 입력해 주세요.")
            elif apply_parsed_values(parsed, df):
                st.success("값을 자동으로 채웠습니다.")
                st.rerun()
            else:
                st.warning("설명에서 유효한 값을 찾지 못했습니다.")


def render_top_bar(df: pd.DataFrame) -> int:
    """상단 바(한 줄): 고객 번호 선택 / 모델·조합 선택 / 프리셋 버튼 / AI 조건 입력."""
    with st.container(border=True, key="card_top_bar"):
        bar_cols = st.columns([1.1, 1.3, 1.6, 1.8])
        # 프리셋 버튼이 session_state["selected_customer_idx"]를 바꿀 수 있으므로,
        # number_input 위젯을 만드는 render_customer_selector보다 먼저 채워야 한다
        # (위젯이 생성된 뒤에는 그 키의 session_state를 코드로 바꿀 수 없음). 다만
        # st.columns는 채우는 순서와 무관하게 선언된 위치(bar_cols[i])에 그려지므로
        # 화면에 보이는 가로 순서는 그대로 유지된다.
        with bar_cols[2]:
            render_profile_buttons(df)
        with bar_cols[0]:
            idx = render_customer_selector(df)
        with bar_cols[1]:
            render_combo_selector()
        with bar_cols[3]:
            render_nl_fill(df)
    return idx


def render_kpi_row(proba: float, risk_label: str, df: pd.DataFrame):
    """핵심 숫자 카드 3개: 이탈 확률(가장 크게) / 위험 등급 / 데이터 이탈 비율."""
    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        with st.container(border=True, key="card_kpi_proba", height=140):
            st.caption("이탈 확률")
            st.markdown(f'<span class="risk-number">{proba * 100:.1f}%</span>', unsafe_allow_html=True)
    with kpi_cols[1]:
        with st.container(border=True, key="card_kpi_badge", height=140):
            st.caption("위험 등급")
            badge_cls = risk_badge_class(risk_label)
            st.markdown(f'<span class="risk-badge {badge_cls}">{risk_label}</span>', unsafe_allow_html=True)
    with kpi_cols[2]:
        with st.container(border=True, key="card_kpi_churn", height=140):
            churn_rate = float(df[config.TARGET].mean()) if config.TARGET in df.columns else None
            st.metric("데이터 이탈 비율", f"{churn_rate * 100:.1f}%" if churn_rate is not None else "-")


def render_risk_gauge_card(proba: float, risk_label: str, factors):
    with st.container(border=True, key="card_gauge", height="stretch"):
        st.markdown("#### 이탈 위험 게이지")
        st.plotly_chart(make_risk_bar_figure(proba), width="stretch", key="risk_bar_chart")

        if factors:
            top_col, top_val = factors[0]
            direction = "높이는" if top_val > 0 else "낮추는"
            summary = (
                f"현재 **{risk_label}** 고객입니다. "
                f"'{config.COL_DESC.get(top_col, top_col)}' 항목이 위험을 가장 {direction} 요인입니다."
            )
        else:
            summary = f"현재 **{risk_label}** 고객입니다."
        st.markdown(f'<p class="risk-summary">{summary}</p>', unsafe_allow_html=True)

        provider = get_selected_provider()
        if not ai_is_available(provider):
            st.caption(ai_unavailable_tag(provider))
        if st.button("예측 결과 설명 보기 (AI)", key="risk_explain_btn"):
            st.info(explain_prediction(factors, proba, risk_label))


def render_factors_card(customer_input: dict, factors):
    with st.container(border=True, key="card_factors", height="stretch"):
        st.markdown("#### 이 점수를 만드는 요인")
        if not factors:
            st.info("표시할 요인이 없습니다.")
            return

        rows = []
        for col, val in factors:
            label = config.COL_DESC.get(col, col)
            if col in customer_input:
                label += f" ({format_value(col, customer_input.get(col))})"
            rows.append(
                {
                    "항목": label,
                    "SHAP 값": val,
                    "방향": "위험 증가" if val > 0 else "위험 감소",
                }
            )
        factor_df = pd.DataFrame(rows)
        fig = px.bar(
            factor_df, x="SHAP 값", y="항목", orientation="h", color="방향",
            color_discrete_map={"위험 증가": "#e6792a", "위험 감소": "#2a6fe6"},
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed", title=None),
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            legend_title_text="",
        )
        st.plotly_chart(fig, width="stretch", key="factors_chart")


def render_action_card(idx: int, col: str, val: float, customer_input: dict):
    with st.container(border=True, key=f"card_action_{idx}", height=config.CARD_HEIGHT):
        st.markdown(f"**{config.COL_SHORT.get(col, config.COL_DESC.get(col, col))}**")
        st.metric("현재 값", format_value(col, customer_input.get(col)))
        direction = "위험 증가" if val > 0 else "위험 감소"
        st.caption(f"{direction} 요인 (SHAP {val:+.3f})")
        st.caption(rules.get_rule(col, customer_input.get(col)))


def render_solution_result(text: str):
    """"맞춤 해결 방안 생성 (AI)" 결과를 고정 높이 카드에 담고, 줄(항목)이 3개를
    넘으면 상위 3개만 보이고 나머지는 expander에 넣는다. 내용이 넘치면 카드 안에서
    스크롤된다."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    with st.container(border=True, height=config.CARD_HEIGHT, key="card_solution_result"):
        if len(lines) <= 3:
            for line in lines:
                st.markdown(line)
        else:
            for line in lines[:3]:
                st.markdown(line)
            with st.expander("나머지 보기"):
                for line in lines[3:]:
                    st.markdown(line)


# --------------------------------------------------------------------------
# 모델 기반 해결 방안 생성 (신규)
#
# AI(LLM) 호출 없이, 회사가 개입 가능한 변수(config.ACTIONABLE_COLS)를 후보 변경량만큼
# 바꿔봤을 때 선택된 모델·조합의 이탈 확률이 얼마나 떨어지는지 직접 재계산한다.
# --------------------------------------------------------------------------


def actionable_ranked_cols(action_top3, action_rest) -> list[tuple[str, float]]:
    """SHAP 영향 크기 순으로 정렬된 NUM_COLS 목록에서, 회사가 개입 가능한
    (config.ACTIONABLE_COLS에 있는) 변수만 그 순서 그대로 추려낸다."""
    ranked = list(action_top3) + list(action_rest)
    return [(c, v) for c, v in ranked if c in config.ACTIONABLE_COLS]


@st.cache_data
def actionable_scale_map(df: pd.DataFrame) -> dict:
    """증감율처럼 0/음수일 수 있는 변수의 현재 값이 0이라 "현재 값의 N%"로는 변경량을
    만들 수 없을 때 대신 쓸 변화 단위(열의 75% 분위수 절댓값)를 미리 계산해 둔다."""
    scales = {}
    for col in config.ACTIONABLE_COLS:
        q75 = abs(float(df[col].quantile(0.75)))
        scales[col] = q75 if q75 > 0 else 1.0
    return scales


def actionable_candidate_value(col: str, current_value, delta: float, scale: float, df: pd.DataFrame) -> float:
    """현재 값을 ACTIONABLE_COLS에 정의된 방향(증가/감소)으로 delta만큼 바꾼 후보 값을
    만든다. 현재 값이 0이면 scale(75% 분위수)을 대신 기준으로 쓴다. 감소 방향은 0 밑으로
    내려가지 않게 한다."""
    direction = config.ACTIONABLE_COLS[col]["direction"]
    current_value = float(current_value) if current_value is not None else 0.0
    base = abs(current_value) if current_value else scale
    if direction == "increase":
        new_val = current_value + base * delta
    else:
        new_val = max(current_value - base * delta, 0.0)
    if pd.api.types.is_integer_dtype(df[col]):
        new_val = float(int(round(new_val)))
    return new_val


@st.cache_data(show_spinner=False)
def compute_model_based_recommendations(
    model_keys: tuple,
    customer_input_items: tuple,
    actionable_cols: tuple,
    limit_for_et: bool,
):
    """선택한 고객 한 명에 대해, 개입 가능한 변수를 하나씩(그리고 SHAP 영향이 가장 큰
    상위 2개는 조합으로도) 후보 변경량만큼 바꿔보고, 현재 선택된 모델·조합으로 이탈
    확률을 다시 계산한다. ExtraTrees가 포함된 조합이면 변수/변경량 수를 줄인다.
    확률 감소폭(%p) 상위 3개를 반환한다."""
    customer_input = dict(customer_input_items)
    df = load_data()
    model = get_combo_model(list(model_keys))
    scales = actionable_scale_map(df)

    cols = list(actionable_cols)
    if limit_for_et:
        cols = cols[: config.MODEL_BASED_ET_LIMIT_VARS]

    def levels_for(col: str):
        deltas = config.ACTIONABLE_COLS[col]["pct_deltas"]
        return deltas[: config.MODEL_BASED_ET_LIMIT_LEVELS] if limit_for_et else deltas

    base_proba = predict_proba(model, customer_input)

    candidate_rows: list[tuple[dict, dict]] = []
    for col in cols:
        for delta in levels_for(col):
            new_val = actionable_candidate_value(col, customer_input.get(col), delta, scales[col], df)
            if new_val == customer_input.get(col):
                continue
            row = dict(customer_input)
            row[col] = new_val
            candidate_rows.append(({col: new_val}, row))

    if len(cols) >= 2:
        c1, c2 = cols[0], cols[1]
        for d1 in levels_for(c1):
            for d2 in levels_for(c2):
                v1 = actionable_candidate_value(c1, customer_input.get(c1), d1, scales[c1], df)
                v2 = actionable_candidate_value(c2, customer_input.get(c2), d2, scales[c2], df)
                if v1 == customer_input.get(c1) and v2 == customer_input.get(c2):
                    continue
                row = dict(customer_input)
                row[c1] = v1
                row[c2] = v2
                candidate_rows.append(({c1: v1, c2: v2}, row))

    if not candidate_rows:
        return [], base_proba

    batch_df = pd.DataFrame([row for _, row in candidate_rows])[config.FEATURE_COLS]
    probs = model.predict_proba(batch_df)[:, 1]

    seen = set()
    results = []
    for (changes, _row), proba in zip(candidate_rows, probs):
        dedup_key = tuple(sorted(changes.items()))
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        results.append((changes, float(proba), base_proba - float(proba)))

    results.sort(key=lambda x: x[2], reverse=True)
    return results[:3], base_proba


def describe_changes(changes: dict, customer_input: dict) -> str:
    parts = []
    for col, new_val in changes.items():
        label = config.COL_LABEL.get(col, config.COL_DESC.get(col, col))
        before = format_value(col, customer_input.get(col))
        after = format_value(col, new_val)
        parts.append(f"{label} {before} → {after}")
    return ", ".join(parts)


def render_model_action_card(idx: int, changes: dict, proba: float, drop: float, base_proba: float, customer_input: dict):
    # 카드 폭이 좁으므로(3열) 안에서 다시 2열로 나누지 않고, 세로로 쌓아 글자가
    # 잘리지 않게 한다.
    with st.container(border=True, key=f"card_model_action_{idx}", height=config.CARD_HEIGHT):
        st.markdown(f"**{describe_changes(changes, customer_input)}**")
        st.metric("변경 후 이탈 확률", f"{proba * 100:.1f}%")
        st.caption(f"변경 전 {base_proba * 100:.1f}% → 변경 후 {proba * 100:.1f}% (확률 감소 {drop * 100:.1f}%p)")
        if st.button("이 추천을 비교 시나리오에 적용", key=f"model_action_apply_{idx}", width="stretch"):
            st.session_state["pending_slider_overrides"] = dict(changes)
            st.rerun()


def render_actions_tab(customer_input: dict, predict_result: dict | None, action_top3, action_rest, combo: dict):
    if not action_top3:
        st.info("표시할 조치가 없습니다.")
    else:
        card_cols = st.columns(len(action_top3))
        for idx, (col_ui, (col, val)) in enumerate(zip(card_cols, action_top3)):
            with col_ui:
                render_action_card(idx, col, val, customer_input)

    if action_rest:
        with st.expander("나머지 조치 보기"):
            for col, val in action_rest:
                text = rules.get_rule(col, customer_input.get(col))
                label = config.COL_SHORT.get(col, config.COL_DESC.get(col, col))
                st.markdown(f"- **{label}** (현재 값: {format_value(col, customer_input.get(col))}): {text}")

    st.divider()
    provider = get_selected_provider()

    if st.button("모델 기반 해결 방안 생성", key="model_based_btn", width="stretch"):
        ranked_actionable = actionable_ranked_cols(action_top3, action_rest)
        limit_for_et = "ET" in combo["models"]
        with st.spinner("여러 조합을 계산 중입니다..."):
            results, base_proba = compute_model_based_recommendations(
                tuple(combo["models"]),
                tuple(sorted(customer_input.items())),
                tuple(c for c, _ in ranked_actionable),
                limit_for_et,
            )
        st.session_state["model_based_results"] = results
        st.session_state["model_based_base_proba"] = base_proba

    results = st.session_state.get("model_based_results")
    if results:
        result_cols = st.columns(len(results))
        for idx, (col_ui, (changes, proba, drop)) in enumerate(zip(result_cols, results)):
            with col_ui:
                render_model_action_card(
                    idx, changes, proba, drop,
                    st.session_state.get("model_based_base_proba", 0.0), customer_input,
                )
        st.caption("모델이 학습한 상관관계에 기반한 추정이며 실제 효과를 보장하지 않습니다.")
    elif results == []:
        st.info("효과적인 개입 방안을 찾지 못했습니다.")

    if not ai_is_available(provider):
        st.caption(ai_unavailable_tag(provider))
    if st.button("맞춤 해결 방안 생성 (AI)", key="actions_ai_btn", width="stretch"):
        st.session_state["solution_text"] = generate_personalized_solution(customer_input, predict_result)
    if st.session_state.get("solution_text"):
        render_solution_result(st.session_state["solution_text"])

    if st.button("고객 요약 리포트 생성", key="actions_report_btn", width="stretch"):
        st.session_state["customer_report_text"] = build_customer_report(customer_input, predict_result)
    if st.session_state.get("customer_report_text"):
        st.download_button(
            "리포트 다운로드 (txt)",
            data=st.session_state["customer_report_text"],
            file_name="customer_report.txt",
            mime="text/plain",
            key="actions_report_download",
        )


def render_scenario_tab(df: pd.DataFrame, base_row: dict, model) -> dict:
    """슬라이더 10개(config.NUM_COLS)는 이 탭에만 둔다(다른 곳과 중복하지 않음).

    "변경 전" = 기준 고객의 원래 값(base_row) 그대로의 확률, "변경 후" = 지금 이
    슬라이더로 조정한 값(customer_input)의 확률. 별도의 시뮬레이션용 슬라이더를
    따로 두지 않고, 이 탭의 슬라이더 하나가 대시보드 전체의 입력값이자 비교 기준이
    된다.
    """
    st.caption("슬라이더를 움직이면 대시보드 전체 예측이 바로 바뀝니다.")
    num_cols = config.NUM_COLS
    for row_start in range(0, len(num_cols), 2):
        row_cols = st.columns(2)
        for offset, col in enumerate(num_cols[row_start : row_start + 2]):
            with row_cols[offset]:
                render_num_input(col, df, "t1", original_value=base_row.get(col))

    customer_input = {**base_row, **{col: st.session_state[f"t1_{col}"] for col in config.NUM_COLS}}

    base_proba = predict_proba(model, base_row)
    after_proba = predict_proba(model, customer_input)

    st.divider()
    metric_cols = st.columns(2)
    with metric_cols[0]:
        with st.container(border=True, key="card_scenario_before"):
            st.metric("변경 전", f"{base_proba * 100:.1f}%")
    with metric_cols[1]:
        with st.container(border=True, key="card_scenario_after"):
            st.metric(
                "변경 후", f"{after_proba * 100:.1f}%",
                delta=f"{(after_proba - base_proba) * 100:.1f}%p",
                delta_color="inverse",
            )

    provider = get_selected_provider()
    st.divider()
    if not ai_is_available(provider):
        st.caption(ai_unavailable_tag(provider))
    if st.button("시나리오 추천 받기 (AI)", key="scenario_recommend_btn"):
        with st.spinner("여러 조합을 계산 중입니다..."):
            percentiles = numeric_percentiles(df)
            candidates, base_proba2 = recommend_scenarios(model, customer_input, percentiles, top_n=3)

        if not candidates:
            st.info("추천할 수 있는 조합을 찾지 못했습니다.")
        else:
            rows = []
            for changes, proba, drop in candidates:
                change_text = ", ".join(f"{config.COL_DESC.get(c, c)} → {format_value(c, v)}" for c, v in changes.items())
                rows.append(
                    {
                        "변경 내용": change_text,
                        "예측 확률": f"{proba * 100:.1f}%",
                        "확률 감소": f"{drop * 100:.1f}%p",
                        "AI 설명": scenario_description(changes, base_proba2, proba),
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch")

    return customer_input


def render_model_tab(df: pd.DataFrame, combo: dict, shap_fallback: bool):
    """모델 정보 탭: 모델·조합 설명, metrics 표, 데이터 이탈 비율.

    모델/조합을 "고르는" 드롭다운은 상단 바(render_top_bar -> render_combo_selector)에
    있고, 여기서는 지금 선택된 조합을 설명만 한다.
    """
    st.markdown(f"**현재 선택된 모델/조합**: {combo['label']}")
    model_names = ", ".join(config.MODEL_METRICS_KEYS.get(k, k) for k in combo["models"])
    st.caption(f"예측에 쓰는 모델: {model_names} (여러 개면 predict_proba를 단순 평균)")

    metrics, is_sample = get_metrics_data()
    metrics_best_model = metrics.get("best_model")
    info = model_info_summary(df, metrics, metrics_best_model)

    if is_sample:
        st.caption(f"{config.METRICS_PATH} 파일이 없어 샘플 값입니다.")

    churn_rate = info["churn_rate"]
    st.metric("데이터 이탈 비율", f"{churn_rate * 100:.1f}%" if churn_rate is not None else "-")

    st.caption(f"현재 샘플 데이터 {info['total_rows']:,}건 (화면 시연용)")
    if info["eval_count"] is not None:
        st.caption(
            f"metrics.json 기준 best 모델({metrics_best_model}) 평가 건수: "
            f"{info['eval_count']:,}건 (원본 전체 데이터 기준, 위 샘플 건수와는 다른 모집단)"
        )

    if shap_fallback:
        fallback_name = config.MODEL_METRICS_KEYS.get(config.SHAP_FALLBACK_MODEL, config.SHAP_FALLBACK_MODEL)
        st.caption(f"ExtraTrees는 SHAP 계산이 느려, 요인 설명은 {fallback_name} 기준으로 표시합니다.")

    perf = {k: v for k, v in info["metrics"].items() if k not in ("confusion_matrix", "file")}
    if perf:
        st.table(pd.DataFrame([perf]).round(3))


# --------------------------------------------------------------------------
# 하단 접이식: 모델 성능 비교 / 결과서 초안 (AI 챗봇은 상단 버튼의 @st.dialog 팝업으로 분리)
# --------------------------------------------------------------------------


def render_metrics_expander():
    metrics, is_sample = get_metrics_data()
    if is_sample:
        st.info(
            f"{config.METRICS_PATH} 파일이 없어 샘플 값으로 표시합니다. "
            "ml-engineer가 학습한 pipeline_*.joblib / metrics.json을 models/ 에 넣어주세요."
        )

    best_model = metrics.get("best_model")
    threshold = metrics.get("threshold")
    model_rows = {k: v for k, v in metrics.items() if k not in ("best_model", "threshold")}
    metrics_df = pd.DataFrame(model_rows).T

    display_cols = [c for c in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"] if c in metrics_df.columns]
    st.dataframe(metrics_df[display_cols].round(3), width="stretch")

    if best_model:
        st.caption(f"선택된(best) 모델: {best_model}" + (f" / 판단 기준(threshold): {threshold}" if threshold is not None else ""))

    chart_metrics = [m for m in ["recall", "f1", "pr_auc"] if m in metrics_df.columns]
    if chart_metrics:
        plot_df = metrics_df[chart_metrics].reset_index().rename(columns={"index": "model"})
        plot_df = plot_df.melt(id_vars="model", var_name="metric", value_name="value")
        fig = px.bar(
            plot_df, x="model", y="value", color="metric", barmode="group",
            title="모델별 recall / f1 / PR-AUC 비교",
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch", key="metrics_compare_chart")

    st.divider()
    st.markdown("**모델 조합 15가지 점수표 (PR-AUC 순)**")
    combos_df = load_combos_file()
    if combos_df is not None:
        st.dataframe(combos_df, width="stretch")
    else:
        st.caption(f"{config.COMBOS_PATH} 파일이 없어 조합 점수표를 표시할 수 없습니다.")


@st.dialog("AI 챗봇", width="large")
def render_chatbot_dialog(df: pd.DataFrame):
    """AI 챗봇을 팝업(모달)으로 띄운다. 상단 버튼으로 연다 (하단 접이식이 아님).

    @st.dialog는 st.fragment를 상속해서, 안의 위젯(st.chat_input) 상호작용은 이
    다이얼로그만 다시 그린다. 그래서 새 메시지를 추가한 뒤에는 st.rerun() 대신
    st.rerun(scope="fragment")를 써야 다이얼로그가 닫히지 않고 대화창만 갱신된다.
    """
    provider = get_selected_provider()
    if not ai_is_available(provider):
        st.info(f"{ai_unavailable_tag(provider)}: AI 챗봇 기능이 제한됩니다.")

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("질문을 입력하세요 (예: 상담 건수가 많을수록 이탈률이 높아지나요?)")
    if user_msg:
        st.session_state["chat_history"].append({"role": "user", "content": user_msg})
        reply = handle_chat_message(user_msg, df)
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        st.rerun(scope="fragment")


def render_report_expander(shap_ctx: dict):
    """결과서 초안 해석 문단. 비용/지연 방지를 위해 버튼을 눌렀을 때만 LLM을 호출한다
    (렌더링만으로는 절대 자동 호출하지 않음)."""
    metrics, is_sample = get_metrics_data()
    best_model = metrics.get("best_model")
    best_metrics = metrics.get(best_model, {}) if best_model else {}

    overall = overall_importance(shap_ctx)
    top_cols = [c for c, _ in sorted(overall.items(), key=lambda x: x[1], reverse=True)[:3]]

    provider = get_selected_provider()
    if not ai_is_available(provider):
        st.caption(ai_unavailable_tag(provider))

    if st.button("결과 해석 생성", key="report_generate_btn"):
        st.session_state["report_paragraph"] = generate_report_paragraph(best_model, best_metrics, top_cols, is_sample)

    if st.session_state.get("report_paragraph"):
        st.text_area("결과 해석 문단 (복사 가능)", st.session_state["report_paragraph"], height=220, key="report_text_area")
    else:
        st.caption("버튼을 누르면 결과 해석 문단을 생성합니다.")


# --------------------------------------------------------------------------
# CSS (흰 카드 / 연한 회색 배경 / 위험색 배지)
# --------------------------------------------------------------------------


def inject_css():
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"] {{
            font-family: "Segoe UI", "Malgun Gothic", -apple-system, sans-serif;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            border-radius: 14px;
            border: 1px solid rgba(128, 128, 128, 0.3);
        }}
        /* 카드 크기 통일: 같은 행의 카드 높이를 맞춘다 (고정 height와 함께, 혹시
           내용이 더 커지는 경우를 대비한 min-height 백업). */
        div[class*="st-key-card_action_"],
        div[class*="st-key-card_model_action_"],
        div[class*="st-key-card_solution_result"] {{
            min-height: {config.CARD_HEIGHT}px;
        }}
        div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-card_action_"]),
        div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-card_model_action_"]) {{
            align-items: stretch;
        }}
        .risk-number {{
            font-size: 2.6rem;
            font-weight: 700;
        }}
        .risk-badge {{
            display: inline-block;
            padding: 0.25rem 0.9rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.85rem;
            margin-left: 0.8rem;
            vertical-align: middle;
        }}
        .risk-high {{ background: rgba(192, 57, 43, 0.18); color: #e74c3c; }}
        .risk-mid  {{ background: rgba(196, 126, 10, 0.18); color: #e2982b; }}
        .risk-low  {{ background: rgba(30, 132, 73, 0.18); color: #2ecc71; }}
        .risk-summary {{ opacity: 0.75; margin-top: 0.5rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# 메인
# --------------------------------------------------------------------------


def render_dashboard_page():
    """화면 구조(위→아래): 상단 바 -> 핵심 숫자 카드 3개 -> 2열(게이지/요인) ->
    탭 3개(권장 조치 / 조정·비교 시나리오 / 모델 정보) -> 하단 접이식(모델 성능
    비교 / 결과서 초안). AI 챗봇은 상단 버튼의 @st.dialog 팝업.

    핵심 숫자 카드와 2열(게이지/요인)은 탭보다 "위"에 그려지지만, 그 값(proba,
    factors)은 탭 중 하나인 "조정·비교 시나리오"의 슬라이더가 만든 customer_input에
    달려 있다. st.tabs는 기본적으로 선택되지 않은 탭도 매 rerun마다 전부 계산하므로
    (레이아웃-layouts 참고), 탭 내용(슬라이더 포함)을 먼저 채워 customer_input을
    확정한 뒤, st.empty()로 미리 잡아둔 자리에 숫자 카드/게이지를 나중에 그려
    넣는다("요소를 순서 밖에서 넣기" 패턴). 사이드바에서 "2. 고객 이탈 예측
    대시보드"를 선택했을 때만 st.navigation이 이 함수를 호출한다(main() 참고).
    """
    df = load_data()
    apply_pending_overrides(df)
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    title_col, chatbot_col = st.columns([5, 1])
    with title_col:
        st.title("고객 이탈 예측 대시보드")
        st.caption("고객 정보를 입력하면 이탈 확률, 주요 원인, 권장 조치를 한 화면에서 확인할 수 있습니다.")
    with chatbot_col:
        st.write("")
        if st.button("AI 챗봇", key="open_chatbot_btn", icon=":material/smart_toy:", width="stretch"):
            render_chatbot_dialog(df)

    render_sidebar()

    # 1) 상단 바: 고객 번호 선택 / 모델·조합 선택 / 프리셋 버튼 / AI 조건 입력
    idx = render_top_bar(df)
    base_row = get_base_row_dict(df, idx)

    combo = get_selected_combo()
    model = get_combo_model(combo["models"])
    shap_model_keys, shap_fallback = get_shap_model_keys(combo["models"])
    shap_ctx = build_shap_context(tuple(shap_model_keys), df)

    # 2)/3) 자리만 먼저 잡아둔다 (값은 탭을 채운 뒤 아래에서 넣음)
    kpi_placeholder = st.empty()
    gauge_placeholder = st.empty()

    tab_actions, tab_scenario, tab_model = st.tabs(["권장 조치", "조정·비교 시나리오", "모델 정보"])

    # 슬라이더(조정·비교 시나리오 탭)를 먼저 채워 customer_input을 확정한다.
    with tab_scenario:
        customer_input = render_scenario_tab(df, base_row, model)

    proba = predict_proba(model, customer_input)
    risk_label = get_risk_level(proba)
    shap_row, feature_names = compute_row_shap(shap_ctx, customer_input)
    overall_factors = top_shap_factors(shap_row, feature_names, n=3)
    action_top3, action_rest = top_action_factors(shap_row, feature_names, n=3)

    predict_result = {"proba": proba, "risk_label": risk_label, "top_factors": overall_factors}
    st.session_state["customer_input"] = customer_input
    st.session_state["predict_result"] = predict_result

    with tab_actions:
        render_actions_tab(customer_input, predict_result, action_top3, action_rest, combo)

    with tab_model:
        render_model_tab(df, combo, shap_fallback)

    # 2) 핵심 숫자 카드 3개
    with kpi_placeholder.container():
        render_kpi_row(proba, risk_label, df)

    # 3) 2열 동일 높이 카드: 좌(게이지) / 우(요인)
    with gauge_placeholder.container():
        gauge_col, factors_col = st.columns(2)
        with gauge_col:
            render_risk_gauge_card(proba, risk_label, overall_factors)
        with factors_col:
            render_factors_card(customer_input, overall_factors)

    st.divider()
    with st.expander("모델 성능 비교"):
        render_metrics_expander()
    with st.expander("결과서 초안"):
        render_report_expander(shap_ctx)


def main():
    """st.navigation으로 페이지 3개를 사이드바에 "아이콘 + 이름" 목록으로 등록하고,
    선택된 페이지 하나만 그린다. 기본 선택은 "메인"(page_home.render_home_page)이다.

    st.set_page_config / inject_css는 앱 전체에서 한 번만 호출해야 하므로 여기서만
    부른다. "AI 설정" 사이드바와 상단 "AI 챗봇" 버튼은 render_dashboard_page() 안에만
    있어서 "2. 고객 이탈 예측 대시보드" 페이지에서만 보인다.

    page_home.render_home_page()의 카드 버튼이 st.switch_page()로 다른 페이지로
    이동할 수 있어야 하는데, st.navigation이 각 페이지 함수를 인자 없이 호출하므로
    pipeline_page/dashboard_page의 st.Page 객체를 session_state에 미리 담아 둔다.
    """
    st.set_page_config(page_title="고객 이탈 예측 대시보드", layout="wide")
    inject_css()

    home_page = st.Page(page_home.render_home_page, title="메인", icon=":material/home:", default=True)
    pipeline_page = st.Page(
        page_pipeline.render_pipeline_page, title="1. 모델 개발 파이프라인", icon=":material/model_training:"
    )
    dashboard_page = st.Page(
        render_dashboard_page, title="2. 고객 이탈 예측 대시보드", icon=":material/dashboard:"
    )
    st.session_state["_nav_pipeline_page"] = pipeline_page
    st.session_state["_nav_dashboard_page"] = dashboard_page

    pg = st.navigation([home_page, pipeline_page, dashboard_page], position="sidebar")
    pg.run()


if __name__ == "__main__":
    main()
