"""모델 생성을 한 곳에 모은 모듈.

새 모델을 추가하려면 register_model() 로 등록하기만 하면 되고, 학습/평가/조합 코드는 고치지 않는다.
무거운 라이브러리(xgboost, lightgbm, catboost)는 해당 모델을 쓸 때만 불러온다.
"""

from typing import Callable, Dict, Tuple

from . import config

# 키 -> (짧은 이름, 모델을 만드는 함수)
_REGISTRY: Dict[str, Tuple[str, Callable]] = {}


def register_model(key: str, code: str, factory: Callable, display: str = None, color: str = "#888888"):
    """모델을 등록한다. factory 는 인자 없이 호출하면 새 모델 객체를 돌려주는 함수."""
    _REGISTRY[key] = (code, factory)
    config.MODEL_CODES[key] = code
    config.MODEL_DISPLAY.setdefault(code, display or code)
    config.MODEL_COLORS.setdefault(code, color)


def _xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(**config.MODEL_PARAMS["xgb"])


def _lgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**config.MODEL_PARAMS["lgbm"])


def _cat():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(**config.MODEL_PARAMS["cat"])


def _et():
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(**config.MODEL_PARAMS["et"])


register_model("xgb", "XGB", _xgb)
register_model("lgbm", "LGBM", _lgbm)
register_model("cat", "CAT", _cat)
register_model("et", "ET", _et)


def available_keys():
    return list(_REGISTRY)


def build_model(key: str):
    """키로 모델을 만든다. 돌려주는 값: (짧은 이름, 모델 객체)"""
    if key not in _REGISTRY:
        raise ValueError(f"모델은 {' / '.join(_REGISTRY)} 중 하나여야 합니다: {key}")
    code, factory = _REGISTRY[key]
    return code, factory()
