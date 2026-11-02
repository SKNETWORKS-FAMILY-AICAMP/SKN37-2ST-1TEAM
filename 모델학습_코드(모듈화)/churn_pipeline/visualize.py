"""시각화 모듈. 그림 그리는 코드는 여기에만 둔다(계산은 importance.py, metrics.py)."""

import re

import matplotlib

matplotlib.use("Agg")                       # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt
import pandas as pd

from . import config, importance

plt.rcParams["font.family"] = config.FONT_CANDIDATES
plt.rcParams["axes.unicode_minus"] = False

def _bar_with_labels(ax, labels, values, color, fmt="{:.4f}"):
    bars = ax.bar(labels, values, color=color, width=0.65)
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height(), fmt.format(r.get_height()),
                ha="center", va="bottom", fontsize=11)
    return bars


def plot_top5(top: pd.DataFrame, title: str, color: str, path) -> None:
    """변수 중요도 Top5 막대그래프 1장 (열: Feature, Importance)."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    _bar_with_labels(ax, top["Feature"], top["Importance"], color)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance")
    ax.set_ylim(0, max(top["Importance"].max(), 1e-6) * 1.15)
    ax.set_title(title, fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    _save_html_if_possible(top, title, path)


def _save_html_if_possible(top: pd.DataFrame, title: str, png_path) -> None:
    """plotly 가 설치돼 있으면 마우스로 값을 확인할 수 있는 html 도 같이 저장한다(없으면 건너뜀)."""
    try:
        import plotly.express as px
    except ImportError:
        return
    px.bar(data_frame=top, x="Feature", y="Importance", text_auto=".4f", title=title) \
        .write_html(str(png_path).replace(".png", ".html"))


def plot_importance_tables(table: pd.DataFrame, prefix: str, title_fmt: str) -> None:
    """(모델, 순위, 변수, 중요도) 표를 모델별 그림으로 저장하고 CSV 도 함께 저장한다."""
    out = config.out_dir("viz")
    code_by_display = {v: k for k, v in config.MODEL_DISPLAY.items()}
    for display, g in table.groupby("모델", sort=False):
        code = code_by_display[display]
        top = pd.DataFrame({"Feature": g["변수"].values, "Importance": g["중요도"].values})
        plot_top5(top, title_fmt.format(model=display), config.MODEL_COLORS[code],
                  out / f"{prefix}_Top5_{display}.png")
    table.to_csv(out / f"{prefix}_Top5.csv", index=False, encoding="utf-8-sig")


def plot_train_importance() -> None:
    table = importance.train_importance_tables()
    plot_importance_tables(table, "01_학습용_특성중요도", "[{model}] 학습용 데이터 특성 중요도 Top5 (7·8·9월 합산)")


def plot_eval_importance(pkl_path) -> None:
    table = importance.eval_importance_tables(pkl_path)
    plot_importance_tables(table, "02_평가용_특성중요도", "[{model}] 평가용 데이터 특성 중요도 Top5 (섞었을 때 PR-AUC 하락폭)")


def read_train_test_pr_auc(code: str):
    """학습결과_*.txt 에서 train/test PR-AUC 를 읽는다. 파일이 없으면 None."""
    path = config.out_dir("result") / f"학습결과_{code}.txt"
    if not path.exists():
        return None
    m = re.search(r"\[PR-AUC\]\s+train\s+([\d.]+).*?test\s+([\d.]+)", path.read_text(encoding="utf-8"))
    return (float(m.group(1)), float(m.group(2))) if m else None


def plot_train_vs_test() -> pd.DataFrame:
    """모델별 학습용 vs 평가용 PR-AUC 그림 + 4개 모델 비교 그림/표를 저장한다."""
    out = config.out_dir("viz")
    rows = []
    for code, display in config.MODEL_DISPLAY.items():
        pair = read_train_test_pr_auc(code)
        if pair is None:
            continue
        tr, te = pair
        rows.append([display, tr, te, tr - te])
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        _bar_with_labels(ax, ["학습용(train)", "평가용(test)"], [tr, te], ["#4a7fc1", "#e0773b"])
        ax.set_ylim(0, max(tr, te) * 1.2)
        ax.set_ylabel("PR-AUC (높을수록 좋음, 무작위 기준 약 0.02)")
        ax.set_title(f"[{display}] 학습용 vs 평가용 PR-AUC\n차이 {tr - te:.4f} (작을수록 과적합이 적음)", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / f"03_학습용_평가용_PR-AUC_{display}.png", dpi=130)
        plt.close(fig)
    df = pd.DataFrame(rows, columns=["모델", "학습용(train)", "평가용(test)", "차이"])
    df.round(4).to_csv(out / "04_학습용_평가용_PR-AUC_전체비교.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    w = 0.38
    xs = range(len(df))
    ax.bar([i - w / 2 for i in xs], df["학습용(train)"], w, label="학습용(train)", color="#4a7fc1")
    ax.bar([i + w / 2 for i in xs], df["평가용(test)"], w, label="평가용(test)", color="#e0773b")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(df["모델"])
    ax.set_ylabel("PR-AUC")
    ax.set_title("학습용 vs 평가용 PR-AUC (4개 모델 비교)", fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "04_학습용_평가용_PR-AUC_전체비교.png", dpi=130)
    plt.close(fig)
    return df
