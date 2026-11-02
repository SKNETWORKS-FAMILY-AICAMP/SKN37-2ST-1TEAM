"""프로젝트 전역 설정.

경로, 임계값, 컬럼 이름, 모델 하이퍼파라미터는 이 파일에서만 정의한다.
다른 모듈은 값을 직접 쓰지 않고 이 파일을 읽는다. 바꿀 일이 생기면 여기만 고치면 된다.
"""

from pathlib import Path

# ── 재현성 / 평가 기준 ────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2          # train 80% : test 20% (stratify)
THRESHOLD = 0.7          # 이탈 판단 기준(확률 0.7 이상이면 이탈로 판정). 팀 결정값

# ── 데이터 컬럼 ──────────────────────────────────────────────
KEY = "발급회원번호"
TARGET = "이탈_통합"      # "탈회_3개월내" 또는 "무실적_3개월" 로 바꾸면 라벨이 바뀜
LABEL_COLS = ["활성_9월", "탈회_3개월내", "무실적_3개월", "이탈_통합"]
FILTER_ACTIVE = True     # True: 9월에 이용 실적이 있던 고객만 사용
MONTH_COL = "기준년월"

# 7·8월 변수 추가(features.py)
CHANGE_RATIO = 0.01      # 7월과 9월 값이 다른 고객이 1% 넘는 컬럼만 '월마다 바뀌는 컬럼'으로 봄
MAKE_DIFF = True         # 증감 변수(9월-8월, 9월-7월)도 만들지 여부
# 변수 이름 끝의 _7월 / _8월 / _증감_9-7월 / _증감_9-8월 을 떼어 같은 변수끼리 묶는 규칙
MONTH_SUFFIX_PATTERN = r"(_증감_9-[78]월|_[78]월)$"

# ── 출력 폴더 ────────────────────────────────────────────────
OUTPUT_ROOT = Path("모델 학습")
DIRS = {
    "model": "최종모델",
    "result": "학습결과",
    "importance": "변수중요도",
    "pred": "예측",
    "app": "앱용",
    "viz": "시각화결과",
}

# ── 모델 ────────────────────────────────────────────────────
# 키(명령줄에서 쓰는 이름) -> 화면/파일에 쓰는 짧은 이름
MODEL_CODES = {"xgb": "XGB", "lgbm": "LGBM", "cat": "CAT", "et": "ET"}
MODEL_DISPLAY = {"XGB": "XGBoost", "LGBM": "LightGBM", "CAT": "CatBoost", "ET": "ExtraTrees"}
MODEL_COLORS = {"XGB": "#4a7fc1", "LGBM": "#5aa469", "CAT": "#e0a030", "ET": "#9b6bb5"}

# 10% 샘플에서 RandomizedSearchCV로 찾은 최적 조건 (전체 데이터 학습에 그대로 사용)
MODEL_PARAMS = {
    "xgb": dict(max_depth=6, learning_rate=0.05, n_estimators=200, scale_pos_weight=25,
                subsample=0.9, colsample_bytree=0.9, min_child_weight=20,
                tree_method="hist", n_jobs=-1, eval_metric="aucpr", random_state=RANDOM_STATE),
    "lgbm": dict(num_leaves=31, learning_rate=0.05, n_estimators=300, scale_pos_weight=25,
                 subsample=0.9, subsample_freq=1, colsample_bytree=0.9, min_child_samples=20,
                 n_jobs=-1, random_state=RANDOM_STATE, verbose=-1),
    "cat": dict(depth=8, learning_rate=0.05, iterations=400, scale_pos_weight=25,
                subsample=0.7, bootstrap_type="Bernoulli", rsm=0.9, l2_leaf_reg=10,
                random_seed=RANDOM_STATE, verbose=0, thread_count=-1, allow_writing_files=False),
    "et": dict(n_estimators=300, max_depth=20, min_samples_leaf=20, max_features=0.5,
               criterion="gini", class_weight="balanced", n_jobs=-1,
               random_state=RANDOM_STATE, verbose=1),
}

# ── 평가/시각화 ──────────────────────────────────────────────
TRAIN_EVAL_SAMPLE = 200_000       # train 점수는 이 크기의 표본으로 계산(시간 절약)
PERMUTATION_SAMPLE = 100_000      # 평가용 중요도 계산에 쓰는 test 행 수
PERMUTATION_REPEAT = 3            # 변수를 섞는 횟수(평균)
THRESHOLD_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
APP_SAMPLE_SIZE = 20_000          # 스트림릿 앱용 샘플 크기
APP_SLIDER_VARS = 10              # 앱 슬라이더에 쓸 상위 변수 개수
FONT_CANDIDATES = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]


def out_dir(kind: str) -> Path:
    """출력 하위 폴더 경로를 돌려주고, 없으면 만든다. kind: DIRS 의 키."""
    path = OUTPUT_ROOT / DIRS[kind]
    path.mkdir(parents=True, exist_ok=True)
    return path
