# SKN37_2ND_1Team
---
## 팀원

<!-- prettier-ignore-start -->
| 권승현 | 김민혁 | 박세윤 | 한세현 |
| :---: | :---: | :---: | :---: 
| [![github](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/edu-ai-jiwon) | [![github](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/nngpfls) | [![github](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/brainkat) | [![github](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/EllyNight) |
<!-- prettier-ignore-end -->

---
## 1. 프로젝트 개요

프로젝트 이름: **💳 카드 고객 이탈 예측 및 대시보드 구축 💸**

### 1-1. 개요
본 프로젝트는 국내 카드사의 VIP 회원 금융 데이터를 기반으로 고객 이탈(Churn) 여부를 예측하는 머신러닝 모델을 구축하고, 이를 활용한 인터랙티브 대시보드를 제공하는 것을 목표로 한다. 
2018년 9월 활성 고객 데이터를 기준으로 3개월 내 탈회·무실적 여부를 예측하기 위한 마스터셋을 구성하였으며, 철저한 탐색적 데이터 분석(EDA)과 전처리를 거쳐 총 6개의 분류 모델(XGBoost, LightGBM, CatBoost, ExtraTrees, Random Forest, MLP)을 적용 및 비교·분석하였다.

### 1-2. 프로젝트 배경
<!-- 내용 공백 처리 -->

### 1-3. 프로젝트 목표
- AI Hub 금융 합성 데이터를 활용한 234만여 명 규모의 데이터 통합 및 고객 이탈 예측 모델 설계
- 클래스 불균형(`이탈 통합 비율 2.01%`)을 고려한 정밀한 전처리 및 평가지표(`Recall`, `Precision`, `F1-Score`, `PR-AUC`) 중심 모델 검증
- Streamlit을 활용한 실시간 이탈 예측 및 시각화 대시보드 구현

---
## 2. 🛠 Tech Stack
| Category | Stack |
|----------|-------|
| 💻 Language | ![Python](https://img.shields.io/badge/PYTHON-3776AB?style=flat-square&logo=python&logoColor=white) |
| 📊 Data Processing | ![Pandas](https://img.shields.io/badge/PANDAS-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NUMPY-013243?style=flat-square&logo=numpy&logoColor=white) |
| 🤖 Machine Learning | ![Scikit-Learn](https://img.shields.io/badge/SCIKIT--LEARN-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBOOST-228B22?style=flat-square&logo=xgboost&logoColor=white) ![LightGBM](https://img.shields.io/badge/LIGHTGBM-9ACD32?style=flat-square) ![CatBoost](https://img.shields.io/badge/CATBOOST-CE412B?style=flat-square&logo=catboost&logoColor=white) ![RandomForest](https://img.shields.io/badge/RANDOM_FOREST-228B22?style=flat-square) ![ExtraTrees](https://img.shields.io/badge/EXTRATREES-2E8B57?style=flat-square) ![DecisionTree](https://img.shields.io/badge/DECISION_TREE-8B4513?style=flat-square) ![MLP](https://img.shields.io/badge/MLP-708090?style=flat-square) ![Optuna](https://img.shields.io/badge/OPTUNA-3C3C3C?style=flat-square) |
| 🖥️ Web Framework | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=Streamlit&logoColor=white) |
| 📈 Visualization | ![Matplotlib](https://img.shields.io/badge/MATPLOTLIB-11557C?style=flat-square&logo=python&logoColor=white) ![Seaborn](https://img.shields.io/badge/SEABORN-4C72B0?style=flat-square) |

---

## 3. 업무 분할 구조
* **데이터 전처리 및 EDA**: 원천 데이터 8개 영역 결합, 결측치 및 이상치 처리, 마스터셋 구성
* **모델링 및 하이퍼파라미터 튜닝**: 10% 샘플 데이터 기반 120개 앙상블 조합 탐색 및 전체 데이터 스케일 업(Scale-up) 재튜닝
* **대시보드 구현**: Streamlit 기반 UI 구성, 최적 앙상블 모델 추론 연동, LLM 피드백 연동 기능 구현

---
## 4. 데이터 수집 및 정의
### 4-1. 금융 합성 데이터 개요
- **데이터 출처**: AI Hub 금융 합성 데이터 (하나카드 원천 기반)
- **원천 범위**: 2018년 7월 ~ 12월 (6개월), 8개 업무 영역, 월별 300만 행
- **분석 기준 시점**: 2018년 9월 활성 고객 대상 (2,345,394행 × 41열)

### 4-2. 목표 변수(Target) 정의 및 비율
- **탈회 3개월 내 (`탈회_3개월내`)**: 비율 **0.29%**
- **무실적 3개월 (`무실적_3개월`)**: 비율 **1.72%**
- **이탈 통합 (`이탈_통합`)**: 탈회 또는 무실적 조건을 충족하는 경우, 비율 **2.01%** (정상 고객 97.99%)

<p align="center">
  <img width="850" alt="타겟 비율 EDA" src="./image/graph6.png">
</p>

> ⚠️ **클래스 불균형**: 정상 고객이 97.99%로 심한 불균형을 보여 정확도(Accuracy) 대신 **Recall, Precision, F1-Score, PR-AUC**를 핵심 평가지표로 활용함.

---
## 5. 데이터 통합 & 전처리 및 EDA

### 5-1. EDA (탐색적 데이터 분석) 주요 결과
- **연속유실적개월수**: 연속 유실적 기간이 짧을수록 이탈률이 급격히 상승함 (정상 평균 ~16개월 vs 이탈 평균 ~2.7개월).
- **이용후경과월_신용**: 신용카드 이용 공백 기간이 길어질수록 이탈 위험이 크게 증가함.
- **강제한도감액횟수_R12M**: 횟수가 많을수록 이탈률이 가파르게 상승함.
- **혜택수혜금액_R3M**: 혜택을 적게 받거나 이용 금액이 낮은 구간에서 이탈 경향이 높게 관찰됨.
- **연령대 및 VIP 등급**: 50대 이상 고연령층에서 이탈률이 점진적으로 증가하는 경향 확인.

#### 📊 주요 EDA 시각화 자료
1. **탈회 집단 표준화 평균 차이**
<p align="center">
  <img width="750" alt="탈회 표준화 차이" src="./image/graph5.png">
</p>

2. **무실적 집단 표준화 평균 차이**
<p align="center">
  <img width="750" alt="무실적 표준화 차이" src="./image/graph4.png">
</p>

3. **통합 이탈 집단 표준화 평균 차이**
<p align="center">
  <img width="750" alt="통합 이탈 표준화 차이" src="./image/graph3.png">
</p>

4. **주요 변수 구간별 이탈률**
<p align="center">
  <img width="750" alt="구간별 이탈률" src="./image/graph2.png">
</p>

5. **연령대별 및 VIP 등급별 이탈률**
<p align="center">
  <img width="750" alt="연령 및 VIP 이탈률" src="./image/graph1.png">
</p>

6. **상위 컬럼 간 상관관계 히트맵**
<p align="center">
  <img width="700" alt="상관관계 히트맵" src="./image/heatmap.png">
</p>

### 5-2. 전처리 과정
1. **결측치 처리**: 
   - `혜택수혜율_R3M` 등 주요 결측치는 데이터 누수를 방지하기 위해 **Train 데이터의 중앙값(`0.7647`)**으로 대체함.
2. **이상치 처리 (Outlier Clipping)**: 
   - IQR(Interquartile Range) 기반 상·하한 경계값 설정을 통해 `이용금액_간편결제_R6M`, `혜택수혜금액_R3M` 등 주요 변수의 극단값 제한.
3. **범주형 변수 인코딩 및 스케일링**: 
   - 연령 및 VIP 등급 코드 대상 더미(Dummy) 인코딩 적용.
   - 데이터 누수 방지를 위해 **Train 데이터 기준만으로 `StandardScaler` 학습 후 적용**.
4. **학습/평가 데이터 분할**:
   - `X_train`: 1,876,315행 / `X_test`: 469,079행 (최종 43개 설명변수).

---
## 6. 파생 변수
- 회원·신용·승인매출·청구·잔액·채널·마케팅·성과정보에서 선정한 변수를 고객-월 단위로 결합 및 가공하여 모델 입력 변수로 구축.

---

# 7. 모델링 및 평가 (전체 데이터 학습 기준)

*(※ 데이터 스케일 확장에 따른 과적합 방지를 위해, 10% 샘플 데이터로 탐색한 Search Space를 바탕으로 전체 데이터에 맞게 하이퍼파라미터를 재조정함)*

### 7-1. 단일 모델 최종 결과 (Top N)
- **XGBoost**: `[최종 파라미터 내용]` (Test PR-AUC `[수치]`, Recall `[수치]`)
- **LightGBM**: `[최종 파라미터 내용]` (Test PR-AUC `[수치]`, Recall `[수치]`)
- **ExtraTrees**: `[최종 파라미터 내용]` (Test PR-AUC `[수치]`, Recall `[수치]`)
*(여기에 전체 학습에 사용한 모델들만 작성)*

### 7-2. 다중 모델 앙상블 시너지 분석
10% 샘플 데이터로 120개의 결합 조합을 사전 테스트한 후, 최상위 시너지를 낸 모델들을 전체 데이터로 앙상블한 결과입니다.
- **최고 성능 조합 (평균 앙상블)**: `[결합한 모델명 ex. XGBoost + LightGBM + ExtraTrees]`
  - 평가 결과: Test PR-AUC `[수치]`, ROC-AUC `[수치]`, Recall `[수치]`, Precision `[수치]`
---

## 8. Streamlit 구현

### 8-1



---
## 9. 인사이트 및 팀원 회고

### 9-1. 주요 인사이트
1. **데이터 누수(Data Leakage) 통제**: 은행 기준인 '12개월 무실적'을 기반으로 Y 데이터를 생성하는 과정에서 타겟 연관성이 지나치게 높은 변수들을 식별하고 통제하는 작업의 중요성을 확인.
2. **클래스 불균형 대응**: 이탈률 2.01%의 불균형 데이터셋에서는 단순 Accuracy가 아닌 PR-AUC, Recall, Precision 중심의 평가 체계가 필수적이며, 모델별로 Precision(CatBoost, XGBoost)과 Recall(RandomForest, LightGBM)의 트레이드오프를 확인.
3. **데이터 스케일업(Scale-up)에 따른 튜닝 전략**: 234만 행의 전체 데이터를 처음부터 앙상블 튜닝하는 대신, 10% 샘플 데이터로 120개의 앙상블 조합을 사전 테스트하여 최적의 조합(XGBoost + ExtraTrees 등)을 먼저 찾아냈습니다. 이후 전체 데이터 학습 시에는 트리의 깊이나 리프 노드 최소 샘플 수를 보수적으로 재조정하여 과적합을 막는 방식으로 튜닝 시간을 획기적으로 단축할 수 있었습니다.

### 9-2. 팀원 회고
- **권승현**: 
- **김민혁**:  
- **박세윤**: 
- **한세현**: 
