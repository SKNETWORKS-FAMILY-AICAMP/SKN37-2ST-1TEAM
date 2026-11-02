"""이탈 예측 샘플 데이터 생성 스크립트.

열 이름/저장 경로는 config.py를 단일 출처로 사용한다 (하드코딩 금지).
`python make_sample.py` 로 실행하면 DATA_PATH 위치에 5,000행 샘플 CSV가 생성된다.

생성 로직 (현실적인 상관관계를 위한 설계):
- 최근접속경과일, 문의횟수가 클수록 이탈 확률이 높아진다.
- 가입기간, 월이용시간이 길/많을수록 이탈 확률이 낮아진다.
- 요금제가 '베이직'이고 가입경로가 '온라인'일수록 이탈 확률이 소폭 높아진다.
- 위 점수를 로지스틱 함수에 넣어 이탈 확률을 만들고, 절편을 이분탐색으로 보정해
  전체 이탈 비율이 목표치(약 10%)에 맞춰지도록 한다.
"""

import os

import numpy as np
import pandas as pd

from config import CAT_COLS, DATA_PATH, NUM_COLS, TARGET

RANDOM_SEED = 42
N_ROWS = 5000
TARGET_CHURN_RATE = 0.10

rng = np.random.default_rng(RANDOM_SEED)

# NUM_COLS 순서: ["가입기간", "월이용시간", "월결제금액", "문의횟수", "최근접속경과일"]
tenure_months = rng.integers(1, 73, size=N_ROWS)  # 가입기간(월): 1~72
monthly_usage_hours = np.clip(rng.normal(40, 15, size=N_ROWS), 0, None).round(1)  # 월이용시간
inquiry_count = rng.poisson(1.0, size=N_ROWS)  # 문의횟수
days_since_last_access = np.clip(rng.exponential(scale=15, size=N_ROWS), 0, None).round().astype(int)  # 최근접속경과일

# CAT_COLS 순서: ["요금제", "가입경로"]
plan_type = rng.choice(["베이직", "스탠다드", "프리미엄"], size=N_ROWS, p=[0.5, 0.35, 0.15])  # 요금제
signup_channel = rng.choice(["온라인", "매장", "제휴"], size=N_ROWS, p=[0.5, 0.3, 0.2])  # 가입경로

# 월결제금액: 요금제별 기본료 + 잡음
plan_base_fee = {"베이직": 9900, "스탠다드": 19900, "프리미엄": 35900}
base_fee = np.array([plan_base_fee[p] for p in plan_type])
monthly_fee = np.clip(base_fee + rng.normal(0, 3000, size=N_ROWS), 0, None).round(-2).astype(int)

# 이탈 위험 점수 (z) 계산: 접속경과일/문의횟수는 +, 가입기간/이용시간은 -
z = (
    0.05 * days_since_last_access
    + 0.5 * inquiry_count
    - 0.03 * tenure_months
    - 0.02 * monthly_usage_hours
)

plan_effect = np.array([{"베이직": 0.3, "스탠다드": 0.0, "프리미엄": -0.3}[p] for p in plan_type])
channel_effect = np.array([{"온라인": 0.2, "매장": -0.1, "제휴": -0.1}[p] for p in signup_channel])
z = z + plan_effect + channel_effect
z = z + rng.normal(0, 0.5, size=N_ROWS)  # 잡음(개인차)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def find_intercept(score, target_rate, lo=-20.0, hi=20.0, iters=100):
    """mean(sigmoid(score + b)) == target_rate 가 되는 절편 b를 이분탐색으로 찾는다."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        rate = sigmoid(score + mid).mean()
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


intercept = find_intercept(z, TARGET_CHURN_RATE)
churn_prob = sigmoid(z + intercept)
churn = rng.binomial(1, churn_prob)

num_values = [tenure_months, monthly_usage_hours, monthly_fee, inquiry_count, days_since_last_access]
cat_values = [plan_type, signup_channel]

data = {}
for col_name, values in zip(NUM_COLS, num_values):
    data[col_name] = values
for col_name, values in zip(CAT_COLS, cat_values):
    data[col_name] = values
data[TARGET] = churn

df = pd.DataFrame(data)

os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")

print(f"저장 완료: {DATA_PATH}")
print(f"행/열 수: {df.shape}")
print(f"이탈 비율: {df[TARGET].mean():.4f}")
