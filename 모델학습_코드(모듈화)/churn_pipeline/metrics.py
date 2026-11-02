"""평가 지표 계산 모듈.

이탈은 약 2%라 정확도(Accuracy)는 의미가 없다(전부 정상이라고 해도 98%).
따라서 임계값과 무관한 PR-AUC / ROC-AUC 와, 임계값 기준 Recall / Precision / F1 을 함께 본다.
"""

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from . import config


def evaluate(y_true, proba, threshold: float = config.THRESHOLD) -> dict:
    """확률(proba)과 정답으로 모든 지표를 한 번에 계산한다."""
    y_true = np.asarray(y_true).ravel()
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "threshold": threshold,
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


def threshold_table(y_true, proba, grid: Iterable[float] = config.THRESHOLD_GRID) -> pd.DataFrame:
    """임계값을 바꿔 가며 Recall / Precision / F1 / 이탈로 지목한 고객 수를 비교한다."""
    y_true = np.asarray(y_true).ravel()
    rows = []
    for th in grid:
        pred = (np.asarray(proba) >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append({"임계값": th,
                     "Recall": recall_score(y_true, pred, zero_division=0),
                     "Precision": precision_score(y_true, pred, zero_division=0),
                     "F1": f1_score(y_true, pred, zero_division=0),
                     "이탈로 지목": int(pred.sum()), "맞춘 이탈자": int(tp), "놓친 이탈자": int(fn)})
    return pd.DataFrame(rows)


def result_text(name, src, n_train, n_test, y_train_mean, y_test_mean, train_pr_auc, test_metrics, minutes) -> str:
    """학습결과_*.txt 에 저장할 문장을 만든다."""
    total = n_train + n_test
    th = test_metrics["threshold"]
    return "\n".join([
        f"모델: {name}", f"입력 파일: {src}",
        f"train {n_train}행 / test {n_test}행 (비율 {n_train / total:.2f} : {n_test / total:.2f})",
        f"이탈률: train {y_train_mean:.4f} / test {y_test_mean:.4f}", "",
        f"[PR-AUC]  train {train_pr_auc:.4f} / test {test_metrics['pr_auc']:.4f}",
        f"[ROC-AUC] test {test_metrics['roc_auc']:.4f}", "",
        f"[test 판단 기준 {th}] Recall {test_metrics['recall']:.4f} / "
        f"Precision {test_metrics['precision']:.4f} / F1 {test_metrics['f1']:.4f}",
        "", f"소요 시간 {minutes:.1f}분"])
