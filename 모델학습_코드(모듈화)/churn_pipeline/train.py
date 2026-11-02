"""모델 1개를 전체 train 으로 학습하고, 저장하고, test 로 평가한다.

저장 순서: 학습 직후 모델부터 저장 -> 다시 불러와 같은 예측이 나오는지 확인 -> 평가/중요도/예측 저장.
(학습이 오래 걸리는 모델이 평가 단계 오류로 날아가지 않게 하기 위함)
"""

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import config, metrics
from .data import load_dataset
from .models import build_model


def _rows(frame, idx):
    return frame.iloc[idx]


def save_model(path, model, columns, threshold, scaler) -> None:
    joblib.dump({"model": model, "columns": columns, "threshold": threshold, "scaler": scaler}, path)


def load_model(code: str) -> dict:
    """저장된 모델 묶음(dict)을 불러온다. code 는 XGB / LGBM / CAT / ET."""
    return joblib.load(config.out_dir("model") / f"최종모델_{code}.pkl")


def train_one(key: str, pkl_path, threshold: float = config.THRESHOLD) -> dict:
    """모델 1개를 학습·저장·평가하고 test 지표(dict)를 돌려준다."""
    t0 = time.time()
    code, model = build_model(key)
    ds = load_dataset(pkl_path)
    print(f"[{code}] train {len(ds.x_train)}행, test {len(ds.x_test)}행, 변수 {len(ds.columns)}개, "
          f"이탈률 train {ds.y_train.mean():.4f} / test {ds.y_test.mean():.4f}")

    model.fit(ds.x_train, ds.y_train)
    print(f"[{code}] 학습 완료 ({(time.time() - t0) / 60:.1f}분)")

    # 1) 학습 직후 저장 + 재로드 확인
    model_path = config.out_dir("model") / f"최종모델_{code}.pkl"
    save_model(model_path, model, ds.columns, threshold, ds.scaler)
    proba_test = model.predict_proba(ds.x_test)[:, 1]
    reloaded = joblib.load(model_path)["model"].predict_proba(ds.x_test.iloc[:200])[:, 1]
    assert np.allclose(reloaded, proba_test[:200], atol=1e-6), "저장한 모델이 원본과 다릅니다"
    print(f"[{code}] 모델 저장 + 재로드 확인 완료: {model_path}")

    # 2) 평가 (train 점수는 표본으로 계산)
    rs = np.random.RandomState(config.RANDOM_STATE)
    idx = rs.choice(len(ds.y_train), min(config.TRAIN_EVAL_SAMPLE, len(ds.y_train)), replace=False)
    train_proba = model.predict_proba(_rows(ds.x_train, idx))[:, 1]
    train_pr_auc = average_precision_score(ds.y_train[idx], train_proba)
    test_metrics = metrics.evaluate(ds.y_test, proba_test, threshold)

    text = metrics.result_text(code, pkl_path, len(ds.x_train), len(ds.x_test), ds.y_train.mean(),
                               ds.y_test.mean(), train_pr_auc, test_metrics, (time.time() - t0) / 60)
    (config.out_dir("result") / f"학습결과_{code}.txt").write_text(text, encoding="utf-8")
    print(text)

    # 3) 변수 중요도와 예측 확률(조합 비교용) 저장
    pd.DataFrame({"변수": ds.columns, "중요도": model.feature_importances_}) \
        .sort_values("중요도", ascending=False) \
        .to_csv(config.out_dir("importance") / f"변수중요도_{code}.csv", index=False, encoding="utf-8-sig")
    np.savez(config.out_dir("pred") / f"예측_{code}.npz", pt=proba_test, y_te=ds.y_test)
    return {"code": code, "train_pr_auc": train_pr_auc, **test_metrics}
