"""프로젝트 전역 설정. 열 이름/경로/임계값/설명/모델 조합은 이 파일에만 둔다.
데이터나 모델이 바뀌면 이 파일만 고치면 된다. 다른 파일(.py)에는 열 이름을 직접 쓰지 않는다.
"""

import pandas as pd

DATA_PATH = "data/sample_churn.csv"
TARGET = "churn"

# 슬라이더로 직접 조작하는 10개 변수 (중요도 상위 변수, 슬라이더_추천변수.txt 기준)
NUM_COLS = [
    "연속유실적개월수_기본_24M_카드",
    "증감율_이용금액_신용_전월",
    "연속무실적개월수_기본_24M_카드",
    "혜택수혜금액_R3M",
    "연회비발생카드수_B0M",
    "혜택수혜율_R3M",
    "최종탈회후경과월",
    "강제한도감액횟수_R12M",
    "이용후경과월_신용",
    "상담건수_R6M",
]

# NUM_COLS를 "권장 조치" 탭의 카드 제목처럼 좁은 공간에 쓸 짧은 한글 이름.
# 없는 열은 COL_DESC(긴 설명) 또는 열 이름 그대로를 대신 쓴다.
COL_SHORT = {
    "연속유실적개월수_기본_24M_카드": "카드 연속사용",
    "증감율_이용금액_신용_전월": "이용금액 증감률",
    "연속무실적개월수_기본_24M_카드": "카드 미사용",
    "혜택수혜금액_R3M": "혜택 수혜금액",
    "연회비발생카드수_B0M": "연회비 카드수",
    "혜택수혜율_R3M": "혜택 수혜율",
    "최종탈회후경과월": "탈회 경과월",
    "강제한도감액횟수_R12M": "한도감액 횟수",
    "이용후경과월_신용": "미이용 경과월",
    "상담건수_R6M": "상담 건수",
}

# 슬라이더 라벨에 쓰는 짧은 한글 이름(한 줄). COL_SHORT와 같은 내용을 재사용한다.
# 이 딕셔너리에 없는 열은 화면에서 COL_DESC(긴 설명)로 대신한다.
COL_LABEL = COL_SHORT

# 슬라이더/카드에 값을 표시할 때 쓰는 형식. "int"=정수, "percent"=비율(값*100 + 소수
# 둘째 자리 + %), "amount"=금액(천 단위 콤마). 여기 없는 열은 "int"로 표시한다.
COL_FORMAT = {
    "연속유실적개월수_기본_24M_카드": "int",
    "증감율_이용금액_신용_전월": "percent",
    "연속무실적개월수_기본_24M_카드": "int",
    "혜택수혜금액_R3M": "amount",
    "연회비발생카드수_B0M": "int",
    "혜택수혜율_R3M": "percent",
    "최종탈회후경과월": "int",
    "강제한도감액횟수_R12M": "int",
    "이용후경과월_신용": "int",
    "상담건수_R6M": "int",
}

# 범주형 열 (이번 데이터는 전부 수치형이라 비어 있음)
CAT_COLS = []

# 학습/예측에서 제외할 열 (ID 등)
DROP_COLS = []

# 모델이 실제로 요구하는 전체 입력 열(순서 포함). DATA_PATH 헤더에서 TARGET을 뺀
# 나머지를 그대로 쓴다 - 데이터가 바뀌어도 이 파일을 고칠 필요가 없다.
_HEADER_COLS = list(pd.read_csv(DATA_PATH, nrows=0).columns)
FEATURE_COLS = [c for c in _HEADER_COLS if c != TARGET]

# 위험 등급 기준 (이탈 확률 기준)
RISK_HIGH = 0.7
RISK_MID = 0.4

# 카드형 컨테이너(st.container(height=...))의 공통 높이. 같은 행에 놓인 카드들의
# 높이를 이 값 하나로 맞춘다. 내용이 넘치면 카드 안에서 스크롤된다.
CARD_HEIGHT = 280

# 슬라이더 10개 변수 + TARGET 에 대한 한국어 설명 (LLM 프롬프트, 화면 문구에 사용).
# "기타 변수"는 슬라이더에 없는 나머지 94개 열의 SHAP 영향을 하나로 묶어 보여줄 때 쓰는 이름.
COL_DESC = {
    "연속유실적개월수_기본_24M_카드": "최근 24개월 중 카드를 연속으로 사용(실적 발생)한 개월 수",
    "증감율_이용금액_신용_전월": "전월 대비 신용카드 이용금액 증감률",
    "연속무실적개월수_기본_24M_카드": "최근 24개월 중 카드를 연속으로 사용하지 않은(무실적) 개월 수",
    "혜택수혜금액_R3M": "최근 3개월간 카드 혜택(포인트/할인 등)으로 받은 금액",
    "연회비발생카드수_B0M": "이번 달 기준 연회비가 발생한 보유 카드 수",
    "혜택수혜율_R3M": "최근 3개월간 제공 가능한 혜택 중 실제로 받은 비율",
    "최종탈회후경과월": "마지막으로 카드(서비스)를 탈회(해지)한 뒤 지난 개월 수",
    "강제한도감액횟수_R12M": "최근 12개월간 카드사가 강제로 이용한도를 줄인 횟수",
    "이용후경과월_신용": "마지막 신용카드 이용 이후 지난 개월 수",
    "상담건수_R6M": "최근 6개월간 고객센터 상담/문의 건수",
    "churn": "이탈 여부 (1=이탈, 0=유지)",
    "기타 변수": "슬라이더로 조절하는 10개 변수 외, 나머지 94개 변수의 영향을 모두 합친 값",
}

# 모델/데이터 파일 경로
MODELS_DIR = "models"
METRICS_PATH = "models/metrics.json"
COMBOS_PATH = "models/combos.csv"

# 모델 짧은 키 -> joblib 파일명. ExtraTrees(ET)는 파일이 약 360MB라 필요할 때만 로드한다.
MODEL_FILES = {
    "XGB": "pipeline_XGB.joblib",
    "LGBM": "pipeline_LGBM.joblib",
    "CAT": "pipeline_CAT.joblib",
    "ET": "pipeline_ET.joblib",
}

# 모델 짧은 키 -> metrics.json 안의 키 이름
MODEL_METRICS_KEYS = {
    "XGB": "XGBoost",
    "LGBM": "LightGBM",
    "CAT": "CatBoost",
    "ET": "ExtraTrees",
}

# SHAP 계산이 빠른(TreeExplainer로 바로 처리되는) 모델. ExtraTrees는 트리 수가 많아
# 느리므로 제외하고, ExtraTrees만 선택된 경우에는 아래 폴백 모델로 요인 설명을 대신한다.
SHAP_FAST_MODELS = {"XGB", "LGBM", "CAT"}
SHAP_FALLBACK_MODEL = "XGB"

# 사이드바 "모델/조합 선택" 드롭다운 목록. combos.csv 의 15가지(단일 4 + 조합 11)를
# PR-AUC 순위 순서로 모두 나열한 것이다. 이 리스트에서 줄을 지우면 화면에서도 빠진다.
# 각 항목의 "models"는 MODEL_FILES/MODEL_METRICS_KEYS 의 키를 조합해 평균 예측에 쓴다.
COMBOS = [
    {"models": ["LGBM", "ET"], "label": "1. LGBM + ET (PR-AUC 0.4592)"},
    {"models": ["LGBM", "CAT", "ET"], "label": "2. LGBM + CAT + ET (PR-AUC 0.4591)"},
    {"models": ["XGB", "LGBM", "ET"], "label": "3. XGB + LGBM + ET (PR-AUC 0.4590)"},
    {"models": ["XGB", "LGBM", "CAT", "ET"], "label": "4. XGB + LGBM + CAT + ET (PR-AUC 0.4589)"},
    {"models": ["XGB", "ET"], "label": "5. XGB + ET (PR-AUC 0.4589)"},
    {"models": ["XGB", "CAT", "ET"], "label": "6. XGB + CAT + ET (PR-AUC 0.4588)"},
    {"models": ["CAT", "ET"], "label": "7. CAT + ET (PR-AUC 0.4587)"},
    {"models": ["XGB", "LGBM", "CAT"], "label": "8. XGB + LGBM + CAT (PR-AUC 0.4570)"},
    {"models": ["XGB", "CAT"], "label": "9. XGB + CAT (PR-AUC 0.4563)"},
    {"models": ["ET"], "label": "10. ET (PR-AUC 0.4563)"},
    {"models": ["XGB", "LGBM"], "label": "11. XGB + LGBM (PR-AUC 0.4561)"},
    {"models": ["LGBM", "CAT"], "label": "12. LGBM + CAT (PR-AUC 0.4561)"},
    {"models": ["XGB"], "label": "13. XGB (PR-AUC 0.4555)"},
    {"models": ["CAT"], "label": "14. CAT (PR-AUC 0.4528)"},
    {"models": ["LGBM"], "label": "15. LGBM (PR-AUC 0.4526)"},
]

# 기본 선택 조합: XGBoost 단일
DEFAULT_COMBO_MODELS = ["XGB"]

# LLM 제공자: "anthropic" / "gemini" / "openai" / "groq" / "ollama" 중 선택
# anthropic/openai는 현재 계정 크레딧이 소진되어 호출이 거부된다(키 형식/코드 문제가
# 아니라 계정 크레딧 문제). gemini는 무료 티어라 호출 한도(429)에 자주 걸린다. groq는
# 무료 티어 한도가 넉넉하고 추론 속도도 빨라 기본값으로 둔다. 사이드바에서 사용자가
# 다른 제공자를 직접 선택할 수도 있다.
LLM_PROVIDER = "groq"

# 제공자별 기본 모델명 (화면에는 "ollama"를 "로컬 AI"로 표시하되, 이 내부 키 이름은 그대로 둔다)
LLM_MODELS = {
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-3.8-flash",
    "openai": "gpt-5.1",
    "groq": "openai/gpt-oss-120b",
    "ollama": "qwen2.5:7b",
}

# 입력 패널 "프로필 예시" 버튼. 실제 데이터에서 고른 고객 3명의 행 번호(row_index)만
# 저장한다. 버튼을 누르면 해당 고객이 "기준 고객"으로 선택되고, 슬라이더 10개 값과
# 나머지 94개 변수가 모두 그 고객의 실제 값으로 채워진다.
PROFILES = {
    "저위험 고객": {"row_index": 9950},
    "중위험 고객": {"row_index": 9755},
    "고위험 고객": {"row_index": 10234},
}

# "권장 조치" 탭의 "모델 기반 해결 방안 생성"에서 쓰는, 회사가 실제로 개입할 수 있는
# 변수 목록이다. 연속유실적개월수처럼 "고객 행동의 결과"인 변수는 회사가 직접 바꿀 수
# 없으므로 넣지 않는다.
# - direction: "increase"(값을 늘리는 방향으로 개입) / "decrease"(값을 줄이는 방향으로 개입)
# - pct_deltas: 후보 변경량. increase는 현재 값 대비 증가율(예: 0.10=10% 증가),
#   decrease는 현재 값 대비 감소율(0.5=절반으로 감소, 1.0=0으로 감소)로 쓴다.
ACTIONABLE_COLS = {
    "혜택수혜금액_R3M": {"direction": "increase", "pct_deltas": [0.10, 0.20, 0.50]},
    "혜택수혜율_R3M": {"direction": "increase", "pct_deltas": [0.10, 0.20, 0.50]},
    "증감율_이용금액_신용_전월": {"direction": "increase", "pct_deltas": [0.10, 0.20, 0.50]},
    "상담건수_R6M": {"direction": "decrease", "pct_deltas": [0.5, 1.0]},
    "강제한도감액횟수_R12M": {"direction": "decrease", "pct_deltas": [0.5, 1.0]},
    "연회비발생카드수_B0M": {"direction": "decrease", "pct_deltas": [0.5, 1.0]},
}

# ExtraTrees(ET)가 포함된 조합이 선택됐을 때, "모델 기반 해결 방안 생성" 계산량을
# 줄이기 위한 상한(ACTIONABLE_COLS 중 앞에서부터 이만큼, 변경량도 앞에서부터 이만큼만 씀).
MODEL_BASED_ET_LIMIT_VARS = 5
MODEL_BASED_ET_LIMIT_LEVELS = 3

# --- 1페이지(파이프라인) 표시용 프로젝트 결과값 ---
# page_pipeline.py 는 이 섹션의 상수만 읽어서 화면을 그린다 (숫자/문구를 직접 쓰지 않음).
# 최종 모델이 팀에서 확정되면 PIPELINE_FINAL 과 관련 수치만 갱신하면 된다.

# 1단계: 데이터 확인 및 전처리
PIPELINE_STEP1 = {
    "total_customers": 2345394,
    "total_customers_caption": "원본 3,000,000명 x 6개월(2018-07~12) 중 9월 활성 고객",
    "feature_count": 104,
    "feature_count_caption": "9월 기준 36개 + 7·8월 추가 68개",
    "missing_status": "중앙값 대치 완료 (5개 컬럼)",
    "sample_caption": "앱용 샘플 20,000행 기준(전체 데이터 아님)",
}

# KPI3("결측치 처리 상태")의 대상 컬럼 5개
PIPELINE_MISSING_COLS = [
    "혜택수혜율_R3M",
    "혜택수혜율_R3M_7월",
    "혜택수혜율_R3M_8월",
    "혜택수혜율_R3M_증감_9-8월",
    "혜택수혜율_R3M_증감_9-7월",
]

# "전처리 내역" 표 (단계, 내용)
PIPELINE_PREPROCESS_STEPS = [
    {"단계": "대상 선정", "내용": "9월 활성 고객만 사용 (3,000,000 -> 2,345,394명)"},
    {"단계": "결측 처리", "내용": "혜택수혜율 계열 5개 컬럼, train 중앙값으로 대치"},
    {"단계": "범주형", "내용": "연령, VIP등급코드 더미 변환"},
    {"단계": "이상치", "내용": "변수별 IQR 기준 clip (IQR이 0인 변수는 제외)"},
    {"단계": "스케일링", "내용": "StandardScaler (train에만 fit), 트리 모델은 원본 사용"},
    {"단계": "변수 추가", "내용": "월마다 바뀌는 17개 컬럼에 7·8월 값과 증감(9-7월, 9-8월) 68개 추가"},
]

# 2단계: EDA
PIPELINE_STEP2 = {
    "target_counts": {"정상": 2298254, "이탈": 47140},
    "imbalance_detail_caption": "탈회_3개월내 0.29%, 무실적_3개월 1.72%",
    "num_dist_mean": {"정상": 16.01, "이탈": 2.76},
}

# 3단계: 특성 선정 및 Data Split
PIPELINE_STEP3 = {
    "feature_count": 104,
    "train_count": 1876315,
    "test_count": 469079,
    "split_caption": "stratify, random_state=42, 이탈 비율 train/test 모두 2.01%",
}

# 4단계: 머신러닝 모델 기본 평가 (튜닝 전 기본 설정, 9월 변수 36개 기준, 판단 기준 0.5, test 기준)
PIPELINE_BASELINE_CAPTION = (
    "튜닝 전 기본 설정, 9월 변수 기준, 판단 기준 0.5. 이탈 2%라 전부 정상이라고 해도 "
    "정확도가 98%이므로 PR-AUC(기준선 0.02)와 Recall을 중심으로 봅니다."
)
PIPELINE_BASELINE_ROWS = [
    {"Model": "전부 정상(기준선)", "Accuracy": 0.980, "Recall (재현율)": 0.000, "Precision (정밀도)": 0.000, "F1-Score": 0.000, "PR-AUC": 0.020, "ROC-AUC": 0.500},
    {"Model": "의사결정나무", "Accuracy": 0.888, "Recall (재현율)": 0.904, "Precision (정밀도)": 0.142, "F1-Score": 0.245, "PR-AUC": 0.325, "ROC-AUC": 0.948},
    {"Model": "랜덤포레스트", "Accuracy": 0.908, "Recall (재현율)": 0.887, "Precision (정밀도)": 0.166, "F1-Score": 0.279, "PR-AUC": 0.386, "ROC-AUC": 0.952},
    {"Model": "LightGBM", "Accuracy": 0.905, "Recall (재현율)": 0.901, "Precision (정밀도)": 0.163, "F1-Score": 0.277, "PR-AUC": 0.404, "ROC-AUC": 0.958},
    {"Model": "XGBoost", "Accuracy": 0.912, "Recall (재현율)": 0.898, "Precision (정밀도)": 0.173, "F1-Score": 0.290, "PR-AUC": 0.413, "ROC-AUC": 0.958},
]
PIPELINE_BASELINE_FOOTNOTE = "참고: 로지스틱 회귀 PR-AUC 0.215, 그래디언트 부스팅 0.403 (삭제 전 데이터 기준)."

# 5단계: 모델 최적화 (튜닝 + 7·8월 변수 추가 + 임계값 조정)
PIPELINE_THRESHOLD_BEFORE = 0.5
PIPELINE_THRESHOLD_AFTER = 0.7
PIPELINE_TUNING_CAPTION = (
    "전체의 약 10% 샘플로 RandomizedSearchCV(3-fold, PR-AUC 기준) 튜닝, 불균형 보정 "
    "scale_pos_weight=25, 임계값 0.5 -> 0.7, 7·8월 변수 추가(104개) 후 성과 변화"
)
PIPELINE_TUNING_ROWS = [
    {"Model": "XGBoost", "Before Recall": 0.898, "After Recall": 0.8149, "Before Precision": 0.173, "After Precision": 0.2816, "Before PR-AUC": 0.413, "After PR-AUC": 0.4555},
    {"Model": "LightGBM", "Before Recall": 0.901, "After Recall": 0.8198, "Before Precision": 0.163, "After Precision": 0.2815, "Before PR-AUC": 0.404, "After PR-AUC": 0.4526},
]
PIPELINE_TUNING_SUCCESS_MSG = (
    "임계값을 0.5에서 0.7로 올려 Precision이 약 17%에서 28%로, F1이 0.29에서 0.42로 "
    "개선됐습니다. 대신 Recall은 약 90%에서 81%로 낮아지는 교환(trade-off)이 있으며, "
    "이탈로 지목한 고객 중 실제 이탈 비율이 높아져 마케팅 낭비를 줄입니다."
)
PIPELINE_HYPERPARAMS = {
    "XGBoost": {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 200, "scale_pos_weight": 25, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 20},
    "LightGBM": {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 300, "scale_pos_weight": 25, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_samples": 20},
    "CatBoost": {"depth": 8, "learning_rate": 0.05, "iterations": 400, "scale_pos_weight": 25, "subsample": "0.7(Bernoulli)", "rsm": 0.9, "l2_leaf_reg": 10},
    "ExtraTrees": {"n_estimators": 300, "max_depth": 20, "min_samples_leaf": 20, "max_features": 0.5, "class_weight": "balanced"},
}

# 6단계: 과적합 진단 (판단 기준 0.7 test 기준 PR-AUC/Recall)
PIPELINE_OVERFIT_MODELS = ["XGBoost", "LightGBM", "CatBoost", "ExtraTrees"]
PIPELINE_OVERFIT_PRAUC = {
    "XGBoost": {"train": 0.4650, "test": 0.4555},
    "LightGBM": {"train": 0.4607, "test": 0.4526},
    "CatBoost": {"train": 0.4688, "test": 0.4528},
    "ExtraTrees": {"train": 0.5957, "test": 0.4563},
}
PIPELINE_OVERFIT_TEST_RECALL = {"XGBoost": 0.8149, "LightGBM": 0.8198, "CatBoost": 0.8089, "ExtraTrees": 0.8114}
#   금고 참고: 아래 문구는 PIPELINE_OVERFIT_PRAUC의 train/test에서 코드로 계산하는 gap(%p,
#   위 metric 카드)과 별개의 고정 문구다. gap 숫자를 문구에 다시 쓰지 않는 이유는, train/test를
#   각각 반올림해서 뺄셈한 값과 코드가 원값으로 계산한 gap이 반올림 경계(예: 0.95)에서 서로
#   다르게 반올림될 수 있어 화면에 두 숫자가 어긋나 보일 수 있기 때문이다.
PIPELINE_OVERFIT_SUMMARY = {
    "XGBoost": "Train과 Test 간 격차가 매우 작아 과적합 없이 일반화가 잘 됩니다.",
    "LightGBM": "Train과 Test 간 격차가 작아 안정적입니다.",
    "CatBoost": "Train과 Test 간 격차가 작아 안정적인 수준입니다.",
    "ExtraTrees": (
        "Train과 Test 간 격차가 크게 벌어져 뚜렷한 과적합 패턴입니다(Test 성능은 다른 "
        "모델과 비슷하지만 학습 데이터를 훨씬 깊게 외웁니다). ExtraTrees의 Train PR-AUC는 "
        "20만 행 샘플 기준입니다."
    ),
}
PIPELINE_OVERFIT_GAP_ERROR = 10.0  # %p 초과 -> st.error
PIPELINE_OVERFIT_GAP_WARNING = 3.0  # %p 초과 -> st.warning (이하는 st.success)

# 특성 중요도(6단계 차트)용 CSV 경로. 열: 모델, 순위, 변수, 중요도.
PIPELINE_IMPORTANCE_TRAIN_CSV = "data/pipeline/importance_train_top5.csv"
PIPELINE_IMPORTANCE_EVAL_CSV = "data/pipeline/importance_eval_top5.csv"

# 메인 페이지(page_home.py) 카드 2개의 문구와 핵심 숫자.
PIPELINE_HOME = {
    "title": "고객 이탈 예측 프로젝트",
    "subtitle": "카드 가입 고객의 이탈을 예측하는 모델 개발 과정과, 그 모델을 실제로 활용하는 대시보드입니다.",
    "card_pipeline": {
        "title": "1. 모델 개발 및 성능 검증 파이프라인",
        "desc": "데이터 전처리, EDA, 모델 비교·튜닝, 과적합 진단, 최종 모델 선정 과정을 단계별로 확인합니다.",
        "stats": [("분석 고객", "2,345,394명"), ("이탈률", "2.01%"), ("입력 변수", "104개")],
        "button": "열기",
    },
    "card_dashboard": {
        "title": "2. 고객 이탈 예측 대시보드",
        "desc": "고객별 이탈 확률, 주요 원인(SHAP), 권장 조치, 모델 기반 해결 방안을 확인합니다.",
        "stats": [("사용 가능한 모델", "4개"), ("모델 조합", "15가지"), ("판단 기준", "0.7")],
        "button": "열기",
    },
}

# 7단계: 최종 모델 및 핵심 평가 지표 선정 보고서
# 최종 모델이 팀에서 확정되면 이 딕셔너리(title, bullets, note)만 수정하면 된다.
PIPELINE_FINAL = {
    "title": "🏆 현재 기본 모델: XGBoost (팀 최종 확정 전)",
    "bullets": [
        "XGBoost 단일 test: PR-AUC 0.4555, Recall 0.815, Precision 0.282, F1 0.419, Train/Test 격차 0.9%p",
        "모델 조합(평균 앙상블) 15가지 중 1위는 LightGBM + ExtraTrees (PR-AUC 0.4592)이지만 단일 모델과의 차이는 약 0.004로 노이즈 수준",
    ],
    "note": "※ 최종 모델은 팀 확정 후 config 의 PIPELINE_FINAL 문구를 수정하세요",
}
