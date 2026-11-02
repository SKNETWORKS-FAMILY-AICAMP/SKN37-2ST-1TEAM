"""7월·8월 값을 변수로 추가한다 (9월 합본에 붙임).

10~12월 값은 쓰지 않는다(정답이 12월을 보고 만들어져 누수가 생기기 때문).
월마다 값이 바뀌는 컬럼에 대해 컬럼_7월, 컬럼_8월, 컬럼_증감_9-8월, 컬럼_증감_9-7월 을 만든다.
"""

from typing import List

import pandas as pd

from . import config

CHUNK_ROWS = 500_000


def read_months(six_month_csv, wanted_columns, months=(201807, 201808)):
    """6개월 합본에서 필요한 월의 행과 컬럼만 읽는다(파일이 커서 조각으로 읽음)."""
    keep = set(wanted_columns) | {config.KEY, config.MONTH_COL}
    parts = {m: [] for m in months}
    for chunk in pd.read_csv(six_month_csv, encoding="utf-8-sig", chunksize=CHUNK_ROWS):
        chunk = chunk[[c for c in chunk.columns if c in keep]]
        for m in months:
            sub = chunk[chunk[config.MONTH_COL] == m]
            if len(sub):
                parts[m].append(sub)
    return {m: pd.concat(p).drop(columns=[config.MONTH_COL]).drop_duplicates(config.KEY).set_index(config.KEY)
            for m, p in parts.items()}


def find_changing_columns(base: pd.DataFrame, month7: pd.DataFrame, month8: pd.DataFrame) -> List[str]:
    """9월과 7월 값이 다른 고객이 CHANGE_RATIO 를 넘는 숫자 컬럼만 '월마다 바뀌는 컬럼'으로 본다."""
    skip = set(config.LABEL_COLS) | {config.MONTH_COL}
    numeric = [c for c in base.columns if c in month7.columns and c in month8.columns
               and c not in skip and pd.api.types.is_numeric_dtype(base[c])]
    v7 = month7.reindex(base.index)
    return [c for c in numeric if float((v7[c] != base[c]).mean()) > config.CHANGE_RATIO]


def build_month_features(base: pd.DataFrame, month7: pd.DataFrame, month8: pd.DataFrame,
                         columns: List[str]) -> pd.DataFrame:
    """7월·8월 값과 증감 변수를 만든다."""
    v7, v8 = month7.reindex(base.index), month8.reindex(base.index)
    new = {}
    for c in columns:
        new[f"{c}_7월"] = v7[c].astype("float32")
        new[f"{c}_8월"] = v8[c].astype("float32")
        if config.MAKE_DIFF:
            new[f"{c}_증감_9-8월"] = (base[c] - v8[c]).astype("float32")
            new[f"{c}_증감_9-7월"] = (base[c] - v7[c]).astype("float32")
    return pd.DataFrame(new, index=base.index)


def run(six_month_csv, nine_month_csv, out_csv, log_path) -> pd.DataFrame:
    """9월 합본에 7·8월 변수를 붙여 저장한다. 정답 컬럼은 맨 뒤에 둔다."""
    base = pd.read_csv(nine_month_csv, encoding="utf-8-sig").set_index(config.KEY)
    months = read_months(six_month_csv, base.columns)
    cols = find_changing_columns(base, months[201807], months[201808])
    new = build_month_features(base, months[201807], months[201808], cols)
    out = pd.concat([base, new], axis=1).reset_index()
    labels = [c for c in config.LABEL_COLS if c in out.columns]
    out = out[[c for c in out.columns if c not in labels] + labels]
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    lines = [f"월마다 바뀌는 컬럼 {len(cols)}개: {', '.join(cols)}",
             f"추가한 변수 {new.shape[1]}개 (증감 포함 {config.MAKE_DIFF})",
             f"7월/8월 값이 없는 고객 비율(평균): {new.isna().mean().mean():.2%}",
             f"최종 {out.shape[0]}행 x {out.shape[1]}컬럼 -> {out_csv}"]
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    return out
