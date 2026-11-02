# 이탈 예측 Streamlit 앱 - 요구사항 (한 페이지 대시보드 단계)

사용자가 상세 스펙을 직접 제공했으므로 planner를 생략하고 총괄이 정리함. 전체 지시사항은 이 문서를 단일 출처로 본다.

이 문서는 3차 변경(카드 이탈 실데이터/다중 모델 조합 + 기준 고객 선택 UI 도입) 이후의 구조를 기준으로
작성되었다. 이전 버전(탭 7개 구조, 단일 모델 구조)의 내용은 git 이력에서 확인할 수 있다.

## 구조 (project root, 백엔드/프론트엔드 분리 없음 - 예외 적용)
- `config.py` (총괄 작성): DATA_PATH, TARGET, NUM_COLS(슬라이더 10개), CAT_COLS, DROP_COLS,
  FEATURE_COLS(DATA_PATH 헤더에서 자동 계산되는 모델 입력 104개 열, 순서 포함), RISK_HIGH/MID, COL_DESC,
  MODELS_DIR/METRICS_PATH/COMBOS_PATH, MODEL_FILES(모델 짧은 키 -> joblib 파일명),
  MODEL_METRICS_KEYS(모델 짧은 키 -> metrics.json 키), SHAP_FAST_MODELS/SHAP_FALLBACK_MODEL,
  COMBOS(사이드바 "모델/조합 선택" 드롭다운 15개, PR-AUC 순), DEFAULT_COMBO_MODELS(기본값: XGB 단일),
  LLM_PROVIDER, LLM_MODELS(anthropic/gemini/openai/ollama), PROFILES(실제 데이터에서 고른 저/중/고위험
  고객 3명의 row_index)
- `data/sample_churn.csv` (ml-engineer 제공): 카드 이탈 실데이터 샘플(약 2만 행, 이탈률 약 2%).
  104개 수치형 변수(CAT_COLS 없음) + `churn`. 열 이름/순서를 바꾸지 않고 그대로 교체하면 다른 파일은
  고칠 필요가 없다 (config.FEATURE_COLS가 이 파일 헤더에서 자동 계산되므로).
- `models/pipeline_XGB.joblib` / `pipeline_LGBM.joblib` / `pipeline_CAT.joblib` / `pipeline_ET.joblib`
  (ml-engineer 제공): 각각 `named_steps["preprocessor"]`(ColumnTransformer, FEATURE_COLS 104개 통과)
  + `named_steps["classifier"]`인 sklearn Pipeline. `models/metrics.json`(4개 모델 성능 + best_model +
  threshold), `models/combos.csv`(단일 4 + 조합 11 = 15가지 PR-AUC 순위표)도 함께 제공.
  ExtraTrees(ET)는 파일이 약 360MB라 `app.py`가 실제 그 조합이 선택됐을 때만 `st.cache_resource`로 로드한다.
- `rules.py` (frontend): 해결 방안 규칙 딕셔너리. NUM_COLS 순서로 매칭하며, 규칙 없는 열은 기본 문구.
- `llm.py` (frontend): `ask_llm(system, user, provider=None, model=None)` / `ask_llm_json(...)` /
  `has_api_key(provider=None)` / `is_ollama_available()` / `list_ollama_models()` 단일 창구.
  anthropic/gemini/openai는 `ENV_KEYS`의 환경변수(.env)로 키를 읽고, ollama는 API 키 없이 로컬
  `ollama serve` 서버를 호출한다. 어떤 provider든 키 없음/서버 꺼짐/호출 실패/라이브러리 미설치
  시 예외 대신 None을 반환해 앱이 죽지 않는다. 선택된 provider의 라이브러리만 함수 내부에서
  import하므로, 설치되지 않은 다른 provider의 라이브러리 때문에 이 모듈 자체의 import가 깨지지 않는다.
- `app.py` (frontend): 상단 바에서 고른 모델 조합(`config.COMBOS`)의 joblib 파이프라인들을
  `ComboModel`로 평균해서 `predict_proba`. 모델/데이터 파일만 바꾸면 동작. 화면은 탭+카드형
  대시보드(아래 "app.py 화면 구조" 참고) + 하단 접이식(expander) 2개 + 상단 AI 챗봇 팝업 구조.

## app.py 화면 구조 (4차 변경: 상단 바 + KPI 카드 + 탭 구조)

레이아웃/표시 방식만 바꾼 변경이며, 모델 로딩·SHAP·LLM 호출·pandas AST 샌드박스·config 기반
구조 등 기능/로직은 그대로다. 화면 순서(위→아래):

### 공통
- `st.set_page_config(layout="wide")`, 상단 제목 + 한 줄 설명 + 그 옆에 "AI 챗봇" 버튼.
- CSS(`st.markdown(..., unsafe_allow_html=True)`)로 흰 카드(둥근 모서리, 옅은 테두리), 연한 회색
  배경, 깔끔한 sans-serif, 위험색(높음=빨강/중간=주황/낮음=초록) 배지를 적용. 카드는
  `st.container(border=True, key="card_xxx")`로 만들고, 생성되는 `.st-key-card_xxx` 클래스를
  CSS로 꾸민다(공식 문서에서 권장하는 커스텀 CSS 타겟팅 방식). 같은 높이로 맞춰야 하는 카드들은
  `st.container(..., height="stretch")`(핵심 숫자 카드는 고정 픽셀 `height=140`)를 쓴다.
- **실행 순서가 화면 순서와 다르다**: 핵심 숫자 카드/2열 카드는 화면상 탭보다 위에 있지만, 그 값은
  "조정·비교 시나리오" 탭의 슬라이더가 만든 `customer_input`에 달려 있다. `st.tabs`는 선택되지
  않은 탭도 매 rerun마다 전부 계산하므로, `main()`은 ①`st.empty()`로 숫자 카드/2열 카드 자리를
  먼저 잡아두고 → ②탭 내용(슬라이더가 있는 "조정·비교 시나리오" 탭부터)을 채워 `customer_input`을
  확정한 뒤 → ③그 값으로 먼저 잡아둔 자리를 채운다("요소를 순서 밖에서 넣기" 패턴).

### 사이드바 (AI 설정만 유지)
- **AI 제공자 선택**: anthropic/gemini/openai/ollama 4개를 **항상 전부** 표시 (키 유무와 무관).
  화면에는 "Ollama"/"ollama" 글자가 절대 노출되지 않고 "로컬 AI"로만 표시된다 (내부 식별자는
  `config.LLM_PROVIDER == "ollama"`, `import ollama`처럼 그대로 "ollama"를 쓴다 - 코드 이름과
  화면 문구는 별개).
- 제공자별 상태: anthropic/gemini/openai는 `llm.has_api_key(p)`로 "✅ 키 있음"/"❌ 키 없음",
  로컬 AI는 `llm.is_ollama_available()`로 "✅ 실행 중"/"❌ 꺼져 있음" (서버 호출은 짧은 TTL
  캐시로 감싸 매 rerun마다 네트워크를 때리지 않게 함).
- 모델 선택: anthropic/gemini/openai는 `config.LLM_MODELS[provider]`를 캡션으로 표시(현재 provider당
  모델 1개). 로컬 AI는 `llm.list_ollama_models()`로 설치된 모델 목록을 가져와 selectbox로 보여주고
  (실패 시 `config.LLM_MODELS["ollama"]` 기본값 캡션으로 대체), 기본 선택값은 config의 기본 모델.
- 선택값(`llm_provider`, `llm_model`)은 `st.session_state`에 저장되고, 모든 AI 호출(`llm.ask_llm`,
  `llm.ask_llm_json`)이 `provider=`/`model=` 인자로 이 값을 참조한다.
- 키/실행 없는 상태로 AI 기능을 누르면 "AI 키 없음, 기본 문구 사용"(로컬 AI는 "로컬 AI 꺼짐, 기본
  문구 사용") 안내가 뜨고 앱이 죽지 않는다.
- **모델/조합 선택은 사이드바가 아니라 상단 바에 있다** (기존 LLM 멀티 프로바이더 테스트가 "사이드바
  첫 번째 selectbox = provider"를 가정하므로, 사이드바에는 AI 제공자 selectbox 하나만 남긴다).

### 1. 상단 바 (한 줄, `st.columns([1.1, 1.3, 1.6, 1.8])`, `render_top_bar`)
- **고객 번호 선택**: `st.number_input`으로 샘플 데이터의 행 번호(0 ~ 행수-1)를 고른다
  (`render_customer_selector`). 고객이 바뀌면 슬라이더(2번 탭) session_state를 지워서 새 고객의
  실제 값으로 다시 초기화한다.
- **모델/조합 선택**: `config.COMBOS`(combos.csv의 15가지를 PR-AUC 순위 순서로 나열한 리스트, 단일
  4개 + 조합 11개)를 `"1. LGBM + ET (PR-AUC 0.4592)"` 형식 라벨로 보여준다(`render_combo_selector`).
  선택한 조합의 모델 키들로 `ComboModel`을 만들어 `predict_proba`/`recommend_scenarios` 등 예측이
  필요한 모든 곳에서 공통으로 쓴다. 기본 선택은 `config.DEFAULT_COMBO_MODELS`(XGB 단일).
  ExtraTrees가 포함된 조합을 고르면 그 순간 처음으로 360MB 파일을 `st.cache_resource`로 로드한다.
- **프리셋 버튼(저/중/고위험)**: `config.PROFILES`(row_index만 저장)를 순회해 버튼을 생성
  (`render_profile_buttons`). 클릭 시 "고객 번호 선택" 값을 그 row_index로 바꾼다. `st.columns`는
  채우는 코드 순서와 무관하게 선언된 칸에 그려지므로, 프리셋 버튼은 코드상 고객 번호 선택보다 먼저
  채워도(= session_state를 먼저 바꿔도) 화면에는 여전히 세 번째 칸에 나온다 (number_input 위젯이
  만들어진 뒤에는 그 키의 session_state를 코드로 바꿀 수 없어서, 먼저 바꾸는 순서가 필요함).
- **AI 조건 입력**: 한 줄 설명을 LLM으로 파싱해 슬라이더 10개 값에 반영 (`render_nl_fill`, 기존
  기능 재사용).

### 2. 핵심 숫자 카드 3개 (`st.columns(3)`, `render_kpi_row`)
이탈 확률(가장 크게, `.risk-number` CSS), 위험 등급 배지(`.risk-badge`), 데이터 이탈 비율
(`df[config.TARGET].mean()`). 셋 다 `height=140`으로 높이를 맞춘다.

### 3. 2열 동일 높이 카드 (`st.columns(2)`, 둘 다 `height="stretch"`)
- **좌: 이탈 위험 게이지** (`render_risk_gauge_card`): 가로 막대(plotly, 40%·70% 기준선 표시), 한
  줄 요약, `[AI] 예측 결과 설명 보기` 버튼.
- **우: 이 점수를 만드는 요인** (`render_factors_card`, 기존 로직 재사용): SHAP 기반 가로 막대.
  선택한 모델 중 XGB/LGBM/CAT가 있으면 그 모델들의 SHAP을 평균하고, ExtraTrees만 선택된 경우는
  느려서 XGBoost 기준으로 대신 계산한다(`get_shap_model_keys`). `config.NUM_COLS`에 없는 열(나머지
  94개)은 모두 **"기타 변수"** 하나로 묶어서 표시한다(`map_feature_to_original_col`). 위험을 올리는
  요인은 주황(`#e6792a`), 낮추는 요인은 파랑(`#2a6fe6`).

### 4. 탭 3개 (`st.tabs(["권장 조치", "조정·비교 시나리오", "모델 정보"])`)
1. **권장 조치** (`render_actions_tab`): SHAP 영향이 큰 상위 3개(`top_action_factors`, "기타 변수"
   제외하고 슬라이더 10개 중에서만 순위를 매김)만 `st.columns(3)` 카드로 표시. 각 카드
   (`render_action_card`)는 짧은 한글 이름(`config.COL_SHORT`), 현재 값, 위험 증가/감소 한 줄
   요약, `rules.get_rule()` 조치 문구. 나머지 7개는 `st.expander("나머지 조치 보기")`에 불릿으로.
   `[AI] 맞춤 해결 방안 생성`/`고객 요약 리포트 생성` 버튼은 탭 하단에 `st.columns(2)`로 가로 배치.
2. **조정·비교 시나리오** (`render_scenario_tab`): **슬라이더 10개(config.NUM_COLS)는 이 탭에만
   있다**(다른 곳과 중복 없음). "변경 전" = 기준 고객의 원래 값(base_row) 그대로의 확률, "변경 후"
   = 지금 이 슬라이더로 조정한 값(customer_input)의 확률 — 별도 시뮬레이션용 슬라이더를 두지 않고
   이 슬라이더 하나가 대시보드 전체의 입력값이자 비교 기준이 된다. 아래에 `[AI] 시나리오 추천 받기`
   버튼(변수가 10개라 전체 조합을 다 만들면 수가 너무 커지므로, `recommend_scenarios`는 슬라이더
   변수를 한 번에 하나씩 p25/p50/p75로 바꿔보고 확률이 가장 많이 떨어지는 변경 top3을 추천).
3. **모델 정보** (`render_model_tab`): 지금 선택된 모델/조합 설명(선택 드롭다운 자체는 상단 바에
   있음), 데이터 이탈 비율, 현재 샘플 데이터 건수(`len(df)`)와 metrics.json의 best_model
   confusion_matrix 합계(평가 건수)를 **따로** 표시(둘은 같은 모집단이 아니므로 "전체 행수 - 평가
   건수 = 학습건수" 같은 뺄셈은 하지 않음), `metrics.json`의 성능 지표 표.

### 5. 하단 접이식 (`st.expander`, 2개만)
- **모델 성능 비교**: `metrics.json` 4개 모델 비교표 + recall/f1/PR-AUC 막대 그래프, 그 아래
  `models/combos.csv`(조합 15가지 PR-AUC 순위표)를 표로 표시 (metrics.json 없으면 샘플 값).
- **결과서 초안**: `metrics.json` + SHAP(모델 조합 평균, "기타 변수" 포함) → LLM이 해석 문단 생성.
  **렌더링 시 자동으로 LLM을 호출하지 않고, "결과 해석 생성" 버튼을 눌렀을 때만 호출**한다
  (비용/지연 방지).
- AI 챗봇은 여기 없음 — 상단 "AI 챗봇" 버튼을 누르면 `@st.dialog(width="large")` 모달 팝업
  (`render_chatbot_dialog`)으로 열린다. `st.chat_message`, 데이터 질문 → LLM이 pandas 코드 생성/
  실행. 샌드박스(`validate_pandas_code_ast`, `run_pandas_code`, AST 화이트리스트 방식)는 보안
  검토를 통과한 로직이라 **내용을 바꾸지 않고 위치만 이동**했다. 다이얼로그는 `st.fragment`를
  상속하므로, 메시지 전송 후에는 `st.rerun()` 대신 `st.rerun(scope="fragment")`를 써서 팝업이
  닫히지 않고 대화창만 갱신되게 한다.

## 1페이지(파이프라인): "1. 모델 개발 파이프라인"

팀원이 다른 프로젝트(통신사 이탈, 가짜 데이터) 기준으로 만든 파이프라인 대시보드 페이지의 화면
양식/틀(상단 `st.radio(horizontal)` 7단계 메뉴, 카드/표/차트 배치, 색상·높이)은 그대로 두고, 안의
숫자·문구만 이 프로젝트(카드 가입 고객 이탈 예측)의 실제 결과로 교체했다.

- `page_pipeline.py`: `render_pipeline_page()`가 진입점. `render_step_*` 7개 함수(1.데이터 전처리 ~
  7.최종 보고서)가 각 단계를 그린다. 숫자/문구를 직접 쓰지 않고 전부 `config.PIPELINE_*` 상수에서
  읽는다. 무거운 연산(모델 학습/로딩) 없이 `data/sample_churn.csv`와 `data/pipeline/*.csv`(아래)만
  읽는 가벼운 화면이다. 위젯 key는 `pipeline_step`(7단계 라디오), `pipeline_overfit_model`(6단계
  과적합 진단 모델 선택)로 2페이지와 겹치지 않게 지정했다.
- `data/pipeline/importance_train_top5.csv` / `importance_eval_top5.csv`: 6단계(과적합 진단) 차트용
  학습용/평가용 특성중요도 Top5(열: 모델, 순위, 변수, 중요도). ml-engineer가 제공한 원본에서 그대로
  복사.
- `config.py`의 "1페이지(파이프라인) 표시용 프로젝트 결과값" 섹션(`PIPELINE_STEP1~3`,
  `PIPELINE_MISSING_COLS`, `PIPELINE_PREPROCESS_STEPS`, `PIPELINE_BASELINE_*`,
  `PIPELINE_THRESHOLD_BEFORE/AFTER`, `PIPELINE_TUNING_*`, `PIPELINE_HYPERPARAMS`,
  `PIPELINE_OVERFIT_*`, `PIPELINE_IMPORTANCE_*_CSV`, `PIPELINE_FINAL`)에 모든 수치/문구를 모아 둔다.
  최종 모델이 팀에서 확정되면 `PIPELINE_FINAL`(title/bullets/note)만 고치면 7단계 카드가 바뀐다.
- 과적합 진단(6단계)의 "PR-AUC 성능 격차"는 `PIPELINE_OVERFIT_PRAUC`의 train/test 원값에서 코드로
  계산하고(`PIPELINE_OVERFIT_GAP_ERROR`=10%p 초과 시 `st.error`, `PIPELINE_OVERFIT_GAP_WARNING`=3%p
  초과 시 `st.warning`, 이하 `st.success`), 모델별 요약 문구(`PIPELINE_OVERFIT_SUMMARY`)는 정성적
  설명만 담고 격차 숫자를 다시 쓰지 않는다 (train/test를 각각 반올림해서 뺀 값과 코드가 원값으로
  계산한 gap이 반올림 경계에서 서로 다르게 반올림될 수 있어, 화면에 두 숫자가 어긋나 보이는 걸
  피하기 위함 - XGBoost가 실제로 그런 경계 케이스였음).
- 카드 높이는 원본 값(120/380/220 등)을 기준으로 두되, 교체한 문구가 길어져 넘치는 카드만 늘렸다
  (예: 1단계 KPI 190, 2단계 EDA 430, 7단계 최종 보고서 320).

## 5차 변경: 메인 페이지 + st.navigation 라우터 / "1. 모델 개발 파이프라인" 페이지 추가

사용자가 상세 스펙을 직접 제공함(최초 사이드바 드롭다운 지시를 st.navigation 구조로 대체 지시).

### 페이지 구조
- `app.py`는 이제 라우터 역할만 한다: `st.set_page_config`를 한 번 호출한 뒤 `st.navigation([...],
  position="sidebar")`으로 페이지 3개를 등록하고 `pg.run()`으로 선택된 페이지만 그린다. 사이드바는
  selectbox가 아니라 st.navigation이 만드는 "아이콘 + 이름" 세로 목록이다.
  1. **메인**(`page_home.py`의 `render_home_page`, icon `:material/home:`, 기본 선택)
  2. **1. 모델 개발 파이프라인**(`page_pipeline.py`의 `render_pipeline_page`, icon `:material/model_training:`)
  3. **2. 고객 이탈 예측 대시보드**(`app.py`의 `render_dashboard_page` - 예전 `main()` 본문을 그대로
     옮김, icon `:material/dashboard:`)
- `page_home.py`: 제목 "고객 이탈 예측 프로젝트" + 카드 2개(`st.columns(2)`, 둘 다
  `config.CARD_HEIGHT` 높이). 각 카드 문구/핵심 숫자는 `config.PIPELINE_HOME` 딕셔너리에서 읽는다.
  "열기" 버튼은 `st.switch_page()`로 다른 페이지로 이동한다. `st.Page` 객체(함수 기반 페이지라 파일
  경로가 없음)는 `app.py`의 `main()`이 `st.session_state["_nav_pipeline_page"]` /
  `["_nav_dashboard_page"]`에 미리 담아 두고, `page_home.py`는 이 키로 읽어서 `switch_page`에 넘긴다
  (circular import 방지를 위해 `page_home.py`는 `app.py`를 import하지 않음).
- `page_pipeline.py`: 기존 1페이지(팀원 원본 7단계 파이프라인 구조를 실제 프로젝트 결과로 교체한 것,
  아래 "1페이지(파이프라인)" 절 참고)를 그대로 유지. `st.set_page_config`는 호출하지 않는다.
- "AI 설정" 사이드바와 상단 "AI 챗봇" 버튼은 `render_dashboard_page()` 안에만 있어서 "2. 고객 이탈
  예측 대시보드" 페이지에서만 보인다.

### 페이지 전환 시 AI 설정 유지 (중요)
st.navigation은 페이지를 전환할 때 **위젯에 바인딩된 session_state**(예: `key="llm_provider"`로 직접
widget을 만들면 그 키)를 초기화한다. 이걸 그대로 두면 "2페이지 → 메인 → 2페이지"로 돌아왔을 때 AI
제공자 선택이 기본값(OpenAI)으로 되돌아간다. 그래서 `render_sidebar()`는:
- 위젯 자체의 key를 `llm_provider_widget`/`llm_model_widget`(ollama 모델 목록일 때만)으로 분리하고,
- 위젯이 고른 값을 매 run마다 **일반 session_state 키** `llm_provider`/`llm_model`(위젯이 아닌, 코드가
  직접 대입하는 키라 페이지 전환으로 지워지지 않음)에 복사해 둔다.
- `get_selected_provider()`/`get_selected_model()`은 그대로 `llm_provider`/`llm_model`을 읽으므로
  호출부 변경은 없다. 위젯은 매 run마다 이 저장된 값을 `index=`로 다시 넣어 그린다.

### Gemini 모델 버전
- `config.LLM_MODELS["gemini"]`를 `gemini-3.8-flash`로 변경(기존 `gemini-2.5-flash`). 실제
  `client.models.list()` 조회 결과 "3.8"이 들어간 모델 중 `generateContent`를 지원하는 일반 대화용
  모델은 `gemini-3.8-flash` 하나뿐이었음(나머지는 tts/live 전용). `llm.ask_llm`/`ask_llm_json`
  (JSON 모드 포함) 양쪽 다 짧은 프롬프트로 실제 호출해 정상 동작 확인함(사고형 모델이 아니라
  `max_output_tokens` 조정은 불필요).

### AI 호출 실패 vs 키 없음 구분
`explain_prediction`/`generate_personalized_solution`/`handle_chat_message`/`generate_report_paragraph`
4곳은 기존에 "키가 없어서 못 부름"과 "키는 있는데 호출이 실패해서 규칙 기반 문구로 대체함"을 같은
문구로 보여줬다. `ai_call_failed_tag()`("AI 호출 실패, 기본 문구 사용")를 추가해, `ai_is_available()`가
True인데(=호출을 시도했는데) `llm.ask_llm()`이 None을 반환한 경우에만 이 태그를 붙인다. 실패 원인
(예외 타입)은 기존처럼 `llm.py`의 `logger.warning`에만 남긴다.

### tests 변경
`st.navigation`을 쓰는 `app.py`를 `AppTest.from_file()`에 바로 넘기면 기본 선택 페이지인 "메인"이
렌더링되는데, AppTest는 실제 브라우저 URL 라우팅이 없어 st.navigation의 페이지 전환을 흉내 내기
어렵다. 그래서 새 파일 `tests/_dashboard_entry.py`(= `app.py`를 import한 뒤
`app.render_dashboard_page()`를 직접 호출하는 테스트 전용 진입 스크립트, `app.py`의 실제 로직은
그대로 재사용)를 만들고, `test_app_apptest.py`/`test_ai_key_absent_messages.py`/
`test_llm_multiprovider.py` 3개의 `APP_PATH`를 이 파일로 바꿨다. `test_llm_multiprovider.py`는 provider
selectbox를 더 이상 `sb_list[0]`이 아니라 `key == "llm_provider_widget"`로 찾도록도 수정했다(사이드바
구성이 바뀌어도 안전하게). `test_sandbox_manual.py`/`test_sandbox_rebuttal_verify.py`는 `app.py`를
직접 import해 `run_pandas_code()`만 호출하므로 수정 불필요.

## 데이터/모델 교체 대응 (중요 제약)
바뀌어도 아래 2곳만 수정하면 나머지는 동작해야 함:
1. `config.py` (NUM_COLS/COL_DESC/PROFILES/MODEL_FILES/MODEL_METRICS_KEYS/COMBOS 등)
2. `data/`, `models/` 파일 (FEATURE_COLS는 data 헤더에서 자동 계산되므로 열 순서만 모델 학습 때와
   일치하면 됨)
- `app.py`, `llm.py`, `rules.py`에는 열 이름/값/범위/클래스명을 직접 쓰지 않는다. 화면 문구도 config의
  열 목록을 순회해서 생성.
- `make_sample.py`/`train.py`는 이전(단일 합성 데이터 + 단일 sklearn 모델) 구조의 유산으로, 현재
  앱은 ml-engineer가 미리 학습해 둔 `pipeline_*.joblib`/`metrics.json`/`combos.csv`를 그대로 쓰고
  이 두 스크립트를 호출하지 않는다.

## 공통 규칙
- 그래프: plotly, 문구: 짧고 간결한 한국어
- AI 키/서버 없어도 앱이 죽지 않아야 함 (버튼/카드에 "AI 키 없음, 기본 문구 사용" 또는 "로컬 AI
  꺼짐, 기본 문구 사용" 표시)
- 화면 텍스트에는 "Ollama"/"ollama" 글자를 쓰지 않는다 ("로컬 AI"로 통일). 코드 내부 식별자
  (`config.LLM_PROVIDER == "ollama"`, `import ollama` 등)는 그대로 둔다.
- 완료 후 pip 설치 목록, 실행 방법(`streamlit run app.py`), 데이터 교체 체크리스트, 로컬 AI 설치/모델
  받는 방법(`ollama serve`, `ollama pull qwen2.5:7b`) 안내

## 팀 배정 (총괄 판단: 중형 - 화면 다수 + LLM 코드실행 보안 민감 → reviewer 필수)
data-analyst → ml-engineer → frontend → tester → reviewer
(이 프로젝트는 단일 Streamlit 앱 구조라 frontend/data 폴더 분리 대신, 각 팀원이 위에 명시된 파일만 담당하는 것으로 예외 적용함)
