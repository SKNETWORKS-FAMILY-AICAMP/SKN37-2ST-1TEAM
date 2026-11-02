"""파이프라인 통합 테스트 (가짜 데이터, 외부 부스팅 라이브러리 없이 sklearn 모델로 검증).

실행: python -m unittest discover -s tests -v   (프로젝트 루트에서)
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from churn_pipeline import config, data, ensemble, export_app, features, importance, metrics, models, preprocess, train, visualize  # noqa: E402


def make_customers(n=3000, seed=0) -> pd.DataFrame:
    """이탈 약 6%인 가짜 고객 표. 7·8·9월 변수와 정답 컬럼, 문자열 컬럼, 결측을 포함한다."""
    rs = np.random.RandomState(seed)
    use = rs.poisson(8, n).astype(float)
    df = pd.DataFrame({config.KEY: np.arange(n), "연령": rs.choice(["20대", "30대", "40대"], n),
                       "연속유실적개월수": use, "혜택수혜율": rs.rand(n)})
    churn = (rs.rand(n) < 1 / (1 + np.exp(0.6 * use - 1.0))).astype(int)
    df["활성_9월"] = 1
    df.loc[:49, "활성_9월"] = 0
    df["탈회_3개월내"] = 0
    df["무실적_3개월"] = churn
    df[config.TARGET] = churn
    df.loc[:30, "혜택수혜율"] = np.nan
    return df


def make_month_tables(nine: pd.DataFrame, seed=1):
    """9월 합본과 같은 키의 6개월 합본(7·8월 행 포함)을 만든다."""
    rs = np.random.RandomState(seed)
    rows = []
    for month, shift in [(201807, 2), (201808, 1), (201809, 0)]:
        m = nine[[config.KEY, "연속유실적개월수", "혜택수혜율"]].copy()
        m["연속유실적개월수"] = (m["연속유실적개월수"] - shift + rs.randint(-1, 2, len(m))).clip(lower=0)
        m[config.MONTH_COL] = month
        rows.append(m)
    return pd.concat(rows)


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        config.OUTPUT_ROOT = cls.root / "모델 학습"
        # 외부 라이브러리 없이 확인하기 위해 sklearn 모델 2개를 등록한다 (새 모델 추가가 쉬운지도 확인)
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        config.MODEL_PARAMS["et"] = dict(n_estimators=30, max_depth=6, min_samples_leaf=5,
                                         class_weight="balanced", random_state=42, n_jobs=1)
        models.register_model("rf", "RF", lambda: RandomForestClassifier(
            n_estimators=30, max_depth=6, class_weight="balanced", random_state=42, n_jobs=1), "RandomForest")
        config.MODEL_CODES.pop("xgb"), config.MODEL_CODES.pop("lgbm"), config.MODEL_CODES.pop("cat")
        for c in ("XGB", "LGBM", "CAT"):
            config.MODEL_DISPLAY.pop(c, None)

        cls.nine = make_customers()
        cls.csv = cls.root / "합본.csv"
        cls.nine.to_csv(cls.csv, index=False, encoding="utf-8-sig")
        cls.pkl = cls.root / "전처리완료.pkl"
        preprocess.run(cls.csv, cls.pkl, cls.root / "기록.txt")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_01_preprocess_split_and_no_leak(self):
        ds = data.load_dataset(self.pkl)
        self.assertEqual(len(ds.x_train) + len(ds.x_test), int((self.nine["활성_9월"] == 1).sum()))
        self.assertAlmostEqual(len(ds.x_test) / (len(ds.x_train) + len(ds.x_test)), config.TEST_SIZE, places=2)
        self.assertAlmostEqual(ds.y_train.mean(), ds.y_test.mean(), places=2)          # stratify 확인
        self.assertEqual(int(ds.x_train.isna().sum().sum()), 0)                       # 결측 처리 확인
        for forbidden in config.LABEL_COLS + [config.KEY]:
            self.assertNotIn(forbidden, ds.columns)                                  # 정답/식별자가 입력에 없음
        self.assertTrue(any(c.startswith("연령_") for c in ds.columns))                # 더미 변환 확인

    def test_02_features_adds_month_columns(self):
        nine_csv, six_csv = self.root / "nine.csv", self.root / "six.csv"
        self.nine.to_csv(nine_csv, index=False, encoding="utf-8-sig")
        make_month_tables(self.nine).to_csv(six_csv, index=False, encoding="utf-8-sig")
        out = features.run(six_csv, nine_csv, self.root / "out.csv", self.root / "log.txt")
        self.assertIn("연속유실적개월수_7월", out.columns)
        self.assertIn("연속유실적개월수_증감_9-8월", out.columns)
        self.assertEqual(list(out.columns[-4:]), config.LABEL_COLS)                  # 정답 컬럼은 맨 뒤

    def test_03_train_combos_thresholds_importance_export(self):
        for key in ("et", "rf"):
            r = train.train_one(key, self.pkl)
            self.assertGreater(r["pr_auc"], 0.1)                                     # 무작위(약 0.06)보다 높음
        self.assertTrue((config.out_dir("model") / "최종모델_ET.pkl").exists())

        df = ensemble.compare_combinations()
        self.assertEqual(len(df), 3)                                                 # 2개 모델 -> 조합 3가지
        probs, y = ensemble.load_predictions()
        table = metrics.threshold_table(y, ensemble.average_proba(probs, ["ET"]))
        self.assertTrue(table["Recall"].is_monotonic_decreasing)                     # 임계값↑ -> Recall↓

        imp = importance.merged_top_features(
            pd.DataFrame({"변수": ["a_7월", "a_8월", "a", "b"], "중요도": [0.1, 0.2, 0.3, 0.25]}), 2)
        self.assertEqual(imp["Feature"].tolist(), ["a", "b"])                        # 7·8월 합산
        self.assertAlmostEqual(imp["Importance"].iloc[0], 0.6)

        visualize.plot_train_importance()
        visualize.plot_train_vs_test()
        self.assertTrue(any(config.out_dir("viz").glob("03_학습용_평가용_PR-AUC_*.png")))

        export_app.run(self.pkl)
        app = config.out_dir("app")
        metrics_json = json.loads((app / "metrics.json").read_text(encoding="utf-8"))
        self.assertIn("best_model", metrics_json)
        pipe = __import__("joblib").load(app / "pipeline_ET.joblib")
        self.assertEqual(list(pipe.named_steps), ["preprocessor", "classifier"])     # 앱이 기대하는 구조
        self.assertTrue((app / "sample_churn.csv").exists())


if __name__ == "__main__":
    unittest.main()
