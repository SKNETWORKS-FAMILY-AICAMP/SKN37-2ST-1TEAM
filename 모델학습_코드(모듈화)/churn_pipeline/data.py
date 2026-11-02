"""전처리 결과(pkl)를 읽고 쓰는 입출력 모듈. 다른 모듈은 pkl 구조를 직접 다루지 않는다."""

from dataclasses import dataclass
from typing import Any, List

import joblib
import numpy as np
import pandas as pd


@dataclass
class Dataset:
    """학습/평가에 필요한 데이터를 한 묶음으로 전달하는 객체."""
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    columns: List[str]
    scaler: Any = None


def save_dataset(path, ds: Dataset) -> None:
    joblib.dump({
        "x_train": ds.x_train, "x_test": ds.x_test,
        "y_train": ds.y_train, "y_test": ds.y_test,
        "scaler": ds.scaler, "columns": ds.columns,
    }, path)


def load_dataset(path) -> Dataset:
    d = joblib.load(path)
    x_train, x_test = d["x_train"], d["x_test"]
    columns = list(d["columns"]) if "columns" in d else list(x_train.columns)
    if not hasattr(x_train, "iloc"):          # numpy 로 저장된 경우도 DataFrame 으로 통일
        x_train = pd.DataFrame(x_train, columns=columns)
        x_test = pd.DataFrame(x_test, columns=columns)
    return Dataset(x_train, x_test,
                   np.asarray(d["y_train"]).ravel(), np.asarray(d["y_test"]).ravel(),
                   columns, d.get("scaler"))
