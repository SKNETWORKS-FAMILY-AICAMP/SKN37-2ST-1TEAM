"""스트림릿 앱에 붙일 파일을 만든다 (앱 폴더는 건드리지 않고 '모델 학습/앱용' 에만 저장).

만드는 파일: pipeline_{모델}.joblib, pipeline.joblib(XGBoost 복사본), metrics.json, combos.csv,
sample_churn.csv(앱 화면용 표본, 정답 열 이름 churn), 슬라이더_추천변수.txt
"""

import json
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from . import config, importance, metrics
from .data import load_dataset
from .train import load_model

APP_TARGET = "churn"


def build_pipeline(model, columns, sample_for_fit) -> Pipeline:
    """앱이 named_steps["preprocessor"] / ["classifier"] 로 꺼내 쓰는 Pipeline 을 만든다.
    전처리는 이미 끝난 데이터를 그대로 통과(passthrough)시킨다."""
    pre = ColumnTransformer([("num", "passthrough", columns)])
    pre.fit(sample_for_fit)
    return Pipeline([("preprocessor", pre), ("classifier", model)])


def export_pipelines(ds, out) -> dict:
    """모델별 Pipeline 과 성능 요약(metrics.json)을 저장한다."""
    result = {}
    for code, display in config.MODEL_DISPLAY.items():
        if not (config.out_dir("model") / f"최종모델_{code}.pkl").exists():
            print(f"{display} 모델 파일 없음 -> 제외")
            continue
        model = load_model(code)["model"]
        pipe = build_pipeline(model, ds.columns, ds.x_test.iloc[:5])
        proba = pipe.predict_proba(ds.x_test)[:, 1]
        assert np.allclose(proba[:300], model.predict_proba(ds.x_test.iloc[:300])[:, 1], atol=1e-6), \
            f"{display}: 파이프라인 예측이 원본 모델과 다릅니다"
        joblib.dump(pipe, out / f"pipeline_{code}.joblib")
        m = metrics.evaluate(ds.y_test, proba)
        m.pop("threshold")                      # 앱의 표에는 모델 공통 threshold 만 따로 둔다
        m["accuracy"] = float(((proba >= config.THRESHOLD).astype(int) == ds.y_test).mean())
        m["file"] = f"pipeline_{code}.joblib"
        result[display] = m
        print(f"{display}: pipeline_{code}.joblib 저장, PR-AUC {m['pr_auc']:.4f}")
    result["best_model"] = max((k for k in result), key=lambda k: result[k]["pr_auc"])
    result["threshold"] = config.THRESHOLD
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def export_sample(ds, out) -> None:
    """평가용 데이터에서 무작위로 뽑은 표본을 앱 화면용 CSV 로 저장한다."""
    rs = np.random.RandomState(config.RANDOM_STATE)
    idx = rs.choice(len(ds.y_test), min(config.APP_SAMPLE_SIZE, len(ds.y_test)), replace=False)
    s = ds.x_test.iloc[idx].copy()
    s[APP_TARGET] = ds.y_test[idx]
    s.to_csv(out / "sample_churn.csv", index=False, encoding="utf-8-sig")
    print(f"sample_churn.csv 저장 ({len(s)}행, 이탈 {int(s[APP_TARGET].sum())}명)")


def export_slider_variables(out) -> None:
    """XGBoost 중요도(7·8·9월 합산) 상위 변수를 슬라이더 추천 목록으로 저장한다."""
    path = config.out_dir("importance") / "변수중요도_XGB.csv"
    if not path.exists():
        print("변수중요도_XGB.csv 없음 -> 슬라이더 추천 변수 파일은 만들지 않음")
        return
    imp = pd.read_csv(path, encoding="utf-8-sig")
    imp["그룹"] = imp["변수"].map(importance.group_name)
    g = imp.groupby("그룹")["중요도"].sum().sort_values(ascending=False).head(config.APP_SLIDER_VARS)
    lines = [f"중요도 상위 {config.APP_SLIDER_VARS}개 변수 (7·8·9월 합산, 9월 열 이름 기준)"]
    lines += [f"{i + 1}. {k}  ({v:.4f})" for i, (k, v) in enumerate(g.items())]
    (out / "슬라이더_추천변수.txt").write_text("\n".join(lines), encoding="utf-8")


def run(pkl_path) -> None:
    ds = load_dataset(pkl_path)
    out = config.out_dir("app")
    export_pipelines(ds, out)
    if (out / "pipeline_XGB.joblib").exists():
        shutil.copy(out / "pipeline_XGB.joblib", out / "pipeline.joblib")      # 구버전 호환용
    combos = config.OUTPUT_ROOT / "조합 비교.csv"
    if combos.exists():
        shutil.copy(combos, out / "combos.csv")
    export_sample(ds, out)
    export_slider_variables(out)
    print(f"완료: {out}")
