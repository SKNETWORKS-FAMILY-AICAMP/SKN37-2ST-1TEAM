# SKN37_2nd_1Team
---
## 팀원

<!-- prettier-ignore-start -->
| 권승현 | 김민혁 | 박세윤 | 한세현 |
| :---: | :---: | :---: | :---: 
| [![github](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/edu-ai-jiwon) | [![github](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/nngpfls) | [![github](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/brainkat) | [![github](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/gugu-eightyone) | [![github](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/hwang-in-kyu) |
<!-- prettier-ignore-end -->

---
## 1. 프로젝트 개요

프로젝트 이름: 

### 1-1. 개요
본 프로젝트는 국내 카드사의 VIP 회원 금융 데이터를 기반으로 고객 이탈(Churn) 여부를 예측하는 머신러닝 모델을 구축하는 것을 목표로 한다. 
2018년 9월 활성 고객 데이터를 기준으로 3개월 내 탈회·무실적 여부를 예측하기 위한 마스터셋을 구성하였으며, 철저한 탐색적 데이터 분석(EDA)과 전처리를 거쳐 다양한 분류 모델을 적용 및 비교·분석하였다.

### 1-2. 프로젝트 배경
<img width="1921" height="955" alt="image" src="https://github.com/user-attachments/assets/7eb4630e-7e80-4b89-b20d-a95a7ff6908e" />
<p align="right">
    <sub> https://www.sisajournal-e.com/news/articleView.html?idxno=410680</sub>  

최근 카드업계는 가맹점 수수료 수익 비중이 하락하며 전통적인 수익 구조의 한계에 직면했다. 이러한 환경 변화에 대응하기 위해 카드사들은 높은 연회비가 책정된 프리미엄 카드 비중을 확대하는 수익 구조 다변화 전략을 추진하고 있다. 
특히 프리미엄 카드를 이용하는 VIP 고객은 높은 소비 여력을 바탕으로 카드사 수익성에 기여하는 바가 크며, 자산 건전성 확보에도 긍정적인 영향을 미친다. 본 프로젝트는 이와 같은 업계의 구조적 변화를 바탕으로, 카드사의 핵심 수익원인 고객의 이탈 예측을 목표로 한다.  

### 1-3. 프로젝트 목표
- AI Hub 금융 합성 데이터를 활용한 234만여 명 규모의 데이터 통합 및 고객 이탈 예측 모델 설계
- 클래스 불균형(`이탈 통합 비율 2.01%`)을 고려한 정밀한 전처리 및 평가지표(`Recall`, `Precision`, `F1-Score`, `PR-AUC`) 중심 모델 검증

---
## 2. 🛠 Tech Stack
| Category | Stack |
|----------|-------|
| 💻 Language | ![Python](https://img.shields.io/badge/PYTHON-3776AB?style=for-the-badge&logo=python&logoColor=white) |
| 📊 Data Processing | ![Pandas](https://img.shields.io/badge/PANDAS-150458?style=for-the-badge&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NUMPY-013243?style=for-the-badge&logo=numpy&logoColor=white) |
| 🤖 Machine Learning | ![Scikit-Learn](https://img.shields.io/badge/SCIKIT--LEARN-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBOOST-228B22?style=for-the-badge&logo=xgboost&logoColor=white) ![LightGBM](https://img.shields.io/badge/LIGHTGBM-9ACD32?style=for-the-badge) ![RandomForest](https://img.shields.io/badge/RANDOM_FOREST-228B22?style=for-the-badge) ![DecisionTree](https://img.shields.io/badge/DECISION_TREE-8B4513?style=for-the-badge) ![Optuna](https://img.shields.io/badge/OPTUNA-3C3C3C?style=for-the-badge) |
| 🖥️ Web Framework | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white) |
| 📈 Visualization | ![Matplotlib](https://img.shields.io/badge/MATPLOTLIB-11557C?style=for-the-badge&logo=python&logoColor=white) ![Seaborn](https://img.shields.io/badge/SEABORN-4C72B0?style=for-the-badge) |

---

## 3. 업무 분할 구조

---
## 4. 데이터 수집 및 정의
### 4-1. 금융 합성 데이터 개요
- **데이터 출처**: AI Hub 금융 합성 데이터 (하나카드 원천 기반)
- **원천 범위**: 2018년 7월 ~ 12월 (6개월), 8개 업무 영역, 월별 300만 행
- **분석 기준 시점**: 2018년 9월 활성 고객 대상 (2,345,394행 × 41열)

### 4-2. 목표 변수(Target) 정의
- **탈회 3개월 내 (`탈회_3개월내`)**: 비율 **0.29%**
- **무실적 3개월 (`무실적_3개월`)**: 비율 **1.72%**
- **이탈 통합 (`이탈_통합`)**: 탈회 또는 무실적 조건을 충족하는 경우, 비율 **2.01%** (정상 고객 97.99%)
> ⚠️ **클래스 불균형**: 정상 고객이 97.99%로 심한 불균형을 보여 정확도(Accuracy) 대신 **Recall, Precision, F1-Score, PR-AUC**를 핵심 평가지표로 활용함.

---
## 5. 데이터 통합 & 전처리 및 EDA

### 5-1. EDA (탐색적 데이터 분석) 주요 결과
- **연속유실적개월수**: 연속 유실적 기간이 짧을수록 이탈률이 급격히 상승함 (정상 평균 ~16개월 vs 이탈 평균 ~2.7개월).
- **이용후경과월_신용**: 신용카드 이용 공백 기간이 길어질수록 이탈 위험이 크게 증가함.
- **강제한도감액횟수_R12M**: 횟수가 많을수록 이탈률이 가파르게 상승함.
- **혜택수혜금액_R3M**: 혜택을 적게 받거나 이용 금액이 낮은 구간에서 이탈 경향이 높게 관찰됨.
- **연령대 및 VIP 등급**: 50대 이상 고연령층에서 이탈률이 점진적으로 증가하는 경향 확인.

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

## 7. 모델링 및 평가 (XGBoost, Decision Tree, RandomForest, LightGBM)

### 7-1. XGBoost
- 이탈 여부를 직접적으로 유추하게 만드는 일부 사후/상태 컬럼을 정제하여 모델이 일반화된 패턴을 학습하도록 유도.
- 정밀도(Precision) 0.66, 재현율(Recall) 0.42, ROC-AUC 0.9944 기록.

### 7-2. Decision Tree
- `max_depth=4`, `class_weight={0:1, 1:5}` 설정을 통해 모델을 단순화하고 이탈자 가중치 부여.
- Precision 0.35, Recall 0.74, F1-Score 0.48 달성.

### 7-3. RandomForest
- `GroupShuffleSplit` 및 `max_depth=5` 적용.
- Precision 0.58, Recall 1.00, F1-Score 0.74 기록.

### 7-4. LightGBM & Optuna 튜닝
- 시계열(Time-based split) 및 사용자 기반 그룹 분할 검증 진행.
- Optuna를 통한 하이퍼파라미터 최적화 결과, **F1-Score 0.95 ~ 0.96**의 우수한 예측 성능 도출.

---

## 8. 인사이트 및 팀원 회고

### 8-1. 주요 인사이트
1. **데이터 누수(Data Leakage) 통제**: 은행 기준인 '12개월 무실적'을 기반으로 Y 데이터를 생성하는 과정에서 타겟 연관성이 지나치게 높은 변수들을 식별하고 통제하는 작업의 중요성을 확인.
2. **클래스 불균형 대응**: 이탈률 2.01%의 불균형 데이터셋에서는 단순 Accuracy가 아닌 PR-AUC, Recall, Precision 중심의 평가 체계가 필수적임을 검증함.

### 8-2. 팀원 회고
- **권승현**: 
- **김민혁**:  
- **박세윤**: 
- **한세현**: 
