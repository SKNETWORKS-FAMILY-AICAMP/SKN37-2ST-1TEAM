"""저장된 예측 확률을 평균(Soft Voting)해서 모델 조합을 비교한다. 추가 학습 없이 몇 초면 끝난다."""

import itertools
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from . import config, metrics


def load_predictions(codes: Iterable[str] = None):
    """예측_*.npz 에서 모델별 확률을 읽는다. 파일이 없는 모델은 건너뛴다.
    돌려주는 값: ({모델: 확률 배열}, 정답 배열)"""
    codes = list(codes or config.MODEL_CODES.values())
    probs, y = {}, None
    for code in codes:
        path = config.out_dir("pred") / f"예측_{code}.npz"
        if not path.exists():
            print(f"{code} 예측 파일 없음 -> 제외")
            continue
        z = np.load(path)
        probs[code], y = z["pt"], z["y_te"]
    if not probs:
        raise FileNotFoundError("예측 파일이 하나도 없습니다. 먼저 모델을 학습하세요.")
    return probs, y


def average_proba(probs: Dict[str, np.ndarray], combo: Iterable[str]) -> np.ndarray:
    """조합에 속한 모델들의 확률을 같은 비중으로 평균한다."""
    return np.mean([probs[c] for c in combo], axis=0)


def compare_combinations(threshold: float = config.THRESHOLD) -> pd.DataFrame:
    """가능한 모든 조합을 PR-AUC 순으로 비교하고 CSV 로 저장한다."""
    probs, y = load_predictions()
    rows: List[list] = []
    for k in range(1, len(probs) + 1):
        for combo in itertools.combinations(probs, k):
            m = metrics.evaluate(y, average_proba(probs, combo), threshold)
            rows.append([" + ".join(combo), k, threshold, m["pr_auc"], m["roc_auc"],
                         m["recall"], m["precision"], m["f1"]])
    df = pd.DataFrame(rows, columns=["조합", "개수", "판단기준", "PR-AUC", "ROC-AUC", "Recall", "Precision", "F1"])
    df = df.sort_values("PR-AUC", ascending=False).reset_index(drop=True)
    df.index += 1
    df.to_csv(config.OUTPUT_ROOT / "조합 비교.csv", encoding="utf-8-sig", index_label="순위")
    print(df.round(4).to_string())
    return df
