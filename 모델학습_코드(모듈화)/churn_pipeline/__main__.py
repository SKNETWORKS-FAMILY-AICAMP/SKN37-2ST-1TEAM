"""명령줄 진입점. 사용법은 README.md 참고.

    python -m churn_pipeline <명령> [옵션]

명령: features / preprocess / train / combos / thresholds / charts / importance / export
각 명령은 해당 모듈의 함수를 호출만 한다(실제 로직은 모듈에 있음).
"""

import argparse
from pathlib import Path

from . import config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="churn_pipeline", description="이탈 예측 모델 개발 파이프라인")
    p.add_argument("--out", default=str(config.OUTPUT_ROOT), help="결과를 저장할 폴더 (기본: 모델 학습)")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("features", help="9월 합본에 7·8월 변수 추가")
    f.add_argument("six_month_csv")
    f.add_argument("nine_month_csv")
    f.add_argument("--out-csv", default="전처리_합본_7,8,9월변수_12월라벨.csv")
    f.add_argument("--log", default="7,8월 변수 추가_기록.txt")

    pp = sub.add_parser("preprocess", help="본 전처리 (결측, 이상치, 분리, 스케일링)")
    pp.add_argument("csv")
    pp.add_argument("--pkl", default="전처리완료.pkl")
    pp.add_argument("--log", default="전처리_기록.txt")

    t = sub.add_parser("train", help="모델 1개 학습·저장·평가")
    t.add_argument("model", help="xgb / lgbm / cat / et")
    t.add_argument("pkl")

    sub.add_parser("combos", help="저장된 예측으로 모델 조합 비교")

    th = sub.add_parser("thresholds", help="임계값별 Recall/Precision/F1 표")
    th.add_argument("--combo", default="XGB", help="예: XGB 또는 XGB+LGBM")

    sub.add_parser("charts", help="학습용 중요도 Top5, 학습용 vs 평가용 PR-AUC 그림")

    i = sub.add_parser("importance", help="평가용(Permutation) 중요도 Top5 그림")
    i.add_argument("pkl")

    e = sub.add_parser("export", help="스트림릿 앱용 파일 만들기")
    e.add_argument("pkl")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    config.OUTPUT_ROOT = Path(args.out)
    cmd = args.command

    if cmd == "features":
        from . import features
        features.run(args.six_month_csv, args.nine_month_csv, args.out_csv, args.log)
    elif cmd == "preprocess":
        from . import preprocess
        preprocess.run(args.csv, args.pkl, args.log)
    elif cmd == "train":
        from . import train
        train.train_one(args.model, args.pkl)
    elif cmd == "combos":
        from . import ensemble
        ensemble.compare_combinations()
    elif cmd == "thresholds":
        from . import ensemble, metrics
        probs, y = ensemble.load_predictions()
        combo = args.combo.split("+")
        table = metrics.threshold_table(y, ensemble.average_proba(probs, combo))
        table.to_csv(config.OUTPUT_ROOT / f"임계값별_지표_{args.combo}.csv", index=False, encoding="utf-8-sig")
        print(table.round(4).to_string(index=False))
    elif cmd == "charts":
        from . import visualize
        visualize.plot_train_importance()
        print(visualize.plot_train_vs_test().round(4).to_string(index=False))
    elif cmd == "importance":
        from . import visualize
        visualize.plot_eval_importance(args.pkl)
    elif cmd == "export":
        from . import export_app
        export_app.run(args.pkl)


if __name__ == "__main__":
    main()
