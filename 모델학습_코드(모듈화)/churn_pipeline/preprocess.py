"""본 전처리: 확인 -> 범주형 -> 분리 -> 결측 -> 이상치 -> 스케일링 (수업 순서).

단계마다 함수 하나로 나누어 각각 따로 테스트할 수 있다. 결측 대치값과 이상치 경계는
누수를 막기 위해 train 에서만 계산하고 test 에는 같은 값을 적용한다.
"""

from typing import Callable, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config
from .data import Dataset, save_dataset


def load_customers(csv_path, say: Callable[[str], None] = print) -> pd.DataFrame:
    """합본 CSV 를 읽고(필요하면) 9월 활성 고객만 남긴다."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    say(f"[1. 확인] 전체 shape: {df.shape}")
    if config.FILTER_ACTIVE:
        df = df[df["활성_9월"] == 1].copy()
        say(f"  9월 활성 고객만 사용: {df.shape}")
    ratio = df[config.TARGET].value_counts(normalize=True).round(4)
    say(f"  {config.TARGET} 비율:\n{ratio.to_string()}")
    return df


def encode_categoricals(df: pd.DataFrame, say=print) -> pd.DataFrame:
    """문자열 컬럼(연령, VIP등급코드)을 더미 변수로 바꾼다."""
    cat_cols = [c for c in df.select_dtypes(include="object").columns if c != config.KEY]
    say(f"[3. 범주형] 더미 변환 컬럼: {cat_cols}")
    return pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)


def split_xy(df: pd.DataFrame, say=print) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """입력(x)과 정답(y)을 나누고 train/test 로 분리한다(이탈 비율 유지)."""
    x = df.drop(columns=[config.KEY] + [c for c in config.LABEL_COLS if c in df.columns])
    y = df[config.TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y)
    say(f"[4. 분리] x_train {x_train.shape}, x_test {x_test.shape}")
    return x_train, x_test, y_train, y_test


def impute_median(x_train, x_test, columns: List[str], say=print):
    """결측을 train 중앙값으로 채운다(train/test 같은 값 사용 -> 누수 방지)."""
    for c in columns:
        if c in x_train.columns:
            med = x_train[c].median()
            x_train[c] = x_train[c].fillna(med)
            x_test[c] = x_test[c].fillna(med)
            say(f"[5. 결측처리] {c}: train 중앙값 {med:.4f}로 채움 (원인 확인 후 이유를 보완할 것)")
    return x_train, x_test


def clip_outliers_iqr(x_train, x_test, say=print):
    """IQR 기준으로 이상치를 경계값에 맞춘다(clip). 값 종류가 20개 이하인 컬럼과 IQR=0 컬럼은 건너뛴다."""
    for c in x_train.columns:
        if x_train[c].nunique() <= 20:
            continue
        q1, q3 = x_train[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            say(f"[6. 이상치] {c}: IQR=0 이라 clip 안 함")
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        x_train[c] = x_train[c].clip(lo, hi)
        x_test[c] = x_test[c].clip(lo, hi)
        say(f"[6. 이상치] {c}: [{lo:.3f}, {hi:.3f}] 로 clip")
    return x_train, x_test


def shrink_floats(frame: pd.DataFrame) -> pd.DataFrame:
    """용량을 줄이기 위해 float64 를 float32 로 바꾼다."""
    return frame.astype({c: "float32" for c in frame.columns if str(frame[c].dtype) == "float64"})


def run(csv_path, pkl_path, log_path) -> Dataset:
    """전처리 전체를 순서대로 실행하고 결과(pkl)와 기록(txt)을 저장한다."""
    log: List[str] = []

    def say(text: str) -> None:
        print(text)
        log.append(text)

    df = load_customers(csv_path, say)
    na = df.isna().sum()
    na = na[na > 0]
    say(f"[2. 결측] 결측이 있는 컬럼:\n{na.to_string() if len(na) else '없음'}")

    df = encode_categoricals(df, say)
    x_train, x_test, y_train, y_test = split_xy(df, say)
    x_train, x_test = impute_median(x_train, x_test, list(na.index), say)
    x_train, x_test = clip_outliers_iqr(x_train, x_test, say)

    scaler = StandardScaler().fit(x_train)       # 선형/신경망 모델용. 트리 모델은 원본을 쓴다
    say("[7. 스케일링] StandardScaler (train만 fit)")

    ds = Dataset(shrink_floats(x_train), shrink_floats(x_test),
                 y_train.to_numpy(), y_test.to_numpy(), list(x_train.columns), scaler)
    save_dataset(pkl_path, ds)
    say("[8. 저장] 실수는 float32")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    return ds
