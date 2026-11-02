"""변수 중요도 계산.

- 학습용 중요도: 모델이 학습하면서 계산한 값(변수중요도_*.csv)을 7·8·9월 변수끼리 합쳐서 Top5 로 만든다.
- 평가용 중요도(Permutation): test 데이터에서 변수 묶음을 무작위로 섞었을 때 PR-AUC 가 얼마나 떨어지는지 계산한다.
"""

import re
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import config
from .data import load_dataset
from .train import load_model


def group_name(column: str) -> str:
    """연속유실적개월수_..._7월 / _8월 / _증감_9-7월 / _증감_9-8월 -> 같은 이름으로 묶는다."""
    return re.sub(config.MONTH_SUFFIX_PATTERN, "", column)


def merged_top_features(importance_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """변수중요도 표(열: 변수, 중요도)를 월별 변수 합산 후 상위 top_n 으로 줄인다."""
    d = importance_df.copy()
    d["변수"] = d["변수"].map(group_name)
    g = d.groupby("변수")["중요도"].sum().sort_values(ascending=False).head(top_n)
    return pd.DataFrame({"Feature": g.index, "Importance": g.values})


def train_importance_tables(top_n: int = 5) -> pd.DataFrame:
    """저장된 모든 모델의 학습용 중요도 Top5 를 한 표(모델, 순위, 변수, 중요도)로 만든다."""
    rows = []
    for code, display in config.MODEL_DISPLAY.items():
        path = config.out_dir("importance") / f"변수중요도_{code}.csv"
        if not path.exists():
            continue
        top = merged_top_features(pd.read_csv(path, encoding="utf-8-sig"), top_n)
        rows += [[display, i + 1, f, v] for i, (f, v) in enumerate(zip(top["Feature"], top["Importance"]))]
    return pd.DataFrame(rows, columns=["모델", "순위", "변수", "중요도"]).round(4)


def permutation_importance_by_group(model, x: pd.DataFrame, y, n_repeat: int, seed: int) -> Dict[str, float]:
    """변수 묶음별로 값을 섞어서 PR-AUC 하락폭 평균을 구한다."""
    rs = np.random.RandomState(seed)
    base = average_precision_score(y, model.predict_proba(x)[:, 1])
    groups: Dict[str, List[str]] = {}
    for c in x.columns:
        groups.setdefault(group_name(c), []).append(c)
    shuffled = x.copy()
    result = {}
    for g, cols in groups.items():
        original = x[cols].values
        drops = []
        for _ in range(n_repeat):
            shuffled[cols] = original[rs.permutation(len(y))]
            drops.append(base - average_precision_score(y, model.predict_proba(shuffled)[:, 1]))
        shuffled[cols] = original
        result[g] = float(np.mean(drops))
    return result


def eval_importance_tables(pkl_path, top_n: int = 5) -> pd.DataFrame:
    """저장된 모든 모델의 평가용(Permutation) 중요도 Top5 표를 만든다."""
    ds = load_dataset(pkl_path)
    rs = np.random.RandomState(config.RANDOM_STATE)
    idx = rs.choice(len(ds.y_test), min(config.PERMUTATION_SAMPLE, len(ds.y_test)), replace=False)
    x, y = ds.x_test.iloc[idx].reset_index(drop=True), ds.y_test[idx]
    rows = []
    for code, display in config.MODEL_DISPLAY.items():
        if not (config.out_dir("model") / f"최종모델_{code}.pkl").exists():
            continue
        imp = permutation_importance_by_group(load_model(code)["model"], x, y,
                                              config.PERMUTATION_REPEAT, config.RANDOM_STATE)
        top = pd.Series(imp).sort_values(ascending=False).head(top_n)
        rows += [[display, i + 1, f, v] for i, (f, v) in enumerate(top.items())]
        print(f"[{display}] 평가용 중요도 1등: {top.index[0]} {top.iloc[0]:.4f}")
    return pd.DataFrame(rows, columns=["모델", "순위", "변수", "중요도"]).round(4)
