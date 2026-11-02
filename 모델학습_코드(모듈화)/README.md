# 가입 고객 이탈 예측 - 모델 개발 파이프라인

전처리 -> 학습 -> 평가 -> 모델 조합 비교 -> 시각화 -> 스트림릿 앱용 변환을 하나의 패키지(`churn_pipeline`)로 정리한 코드입니다.
기존에는 단계마다 독립된 스크립트(약 35개)가 같은 코드(데이터 읽기, 모델 설정, 지표 계산, 저장 규칙)를 각자 복사해서 가지고 있었습니다. 이를 기능별 모듈로 나누어 같은 코드를 한 곳에서만 관리하도록 바꿨습니다.

## 폴더 구조

```
모델학습_코드(모듈화)/
├─ README.md
├─ requirements.txt
├─ churn_pipeline/            # 패키지 (실제 로직)
│   ├─ config.py              # 경로, 임계값, 컬럼 이름, 모델 하이퍼파라미터 (설정은 여기만)
│   ├─ data.py                # 전처리 결과(pkl) 읽기/쓰기, Dataset 객체
│   ├─ features.py            # 7·8월 변수 추가
│   ├─ preprocess.py          # 본 전처리 (결측, 이상치, 분리, 스케일링)
│   ├─ models.py              # 모델 생성/등록 (새 모델은 register_model 한 줄)
│   ├─ metrics.py             # PR-AUC, Recall, Precision, F1, 임계값별 표
│   ├─ train.py               # 모델 1개 학습, 저장, 평가
│   ├─ ensemble.py            # 예측 확률 평균으로 모델 조합 비교
│   ├─ importance.py          # 학습용/평가용(Permutation) 변수 중요도
│   ├─ visualize.py           # 그래프 그리기
│   ├─ export_app.py          # 스트림릿 앱용 파일 만들기
│   └─ __main__.py            # 명령줄 진입점
└─ tests/
    └─ test_pipeline.py       # 가짜 데이터로 전체 흐름 확인
```

## 사용 순서

프로젝트 루트(이 폴더)에서 실행합니다. 결과는 `--out` 폴더(기본 `모델 학습`)에 저장됩니다.

```
python -m churn_pipeline features "전처리_합본_6개월(이탈라벨X).csv" "전처리_합본_9월변수_12월라벨(삭제 후).csv"
python -m churn_pipeline preprocess "전처리_합본_7,8,9월변수_12월라벨.csv" --pkl "전처리완료(7_8월추가).pkl"
python -m churn_pipeline train xgb  "전처리완료(7_8월추가).pkl"
python -m churn_pipeline train lgbm "전처리완료(7_8월추가).pkl"
python -m churn_pipeline train cat  "전처리완료(7_8월추가).pkl"
python -m churn_pipeline train et   "전처리완료(7_8월추가).pkl"
python -m churn_pipeline combos
python -m churn_pipeline thresholds --combo XGB
python -m churn_pipeline charts
python -m churn_pipeline importance "전처리완료(7_8월추가).pkl"
python -m churn_pipeline export "전처리완료(7_8월추가).pkl"
```

## 설계 원칙

- **설정과 로직 분리**: 임계값 0.7, 8:2 분리, 모델 파라미터는 `config.py` 에만 있습니다. 값을 바꿀 때 다른 파일을 열지 않습니다.
- **기능별 모듈, 한 가지 일만**: 계산(`metrics`, `importance`)과 그림(`visualize`), 입출력(`data`), 실행 순서(`__main__`)를 분리했습니다.
- **같은 코드는 한 번만**: 변수 묶기 규칙(`_7월`, `_8월`, `_증감_9-7월` 제거), 지표 계산, 저장 경로 규칙이 각각 한 곳에만 있습니다.
- **확장성**: 새 모델은 `models.register_model()` 로 등록하면 학습, 조합 비교, 앱용 변환에 자동으로 포함됩니다. 라벨 변경은 `config.TARGET` 한 줄입니다.
- **안전한 저장 순서**: 학습 직후 모델을 먼저 저장하고 다시 불러와 예측이 같은지 확인한 뒤 평가를 진행합니다.
- **누수 방지**: 결측 대치값과 이상치 경계는 train 에서만 계산해 test 에 적용합니다.
- **무거운 라이브러리는 필요할 때만 import**: xgboost, lightgbm, catboost 는 해당 모델을 쓸 때만 불러옵니다.

## 기존 스크립트와의 대응

| 기존 스크립트 | 새 위치 |
|---|---|
| 전처리_5_본전처리.py | `preprocess.py` |
| 변수추가_7_8월.py | `features.py` |
| 본학습.py, 본학습_ET.py | `train.py` + `models.py` + `config.MODEL_PARAMS` |
| 본_조합비교.py | `ensemble.py` |
| 시각화.py | `importance.py` + `visualize.py` |
| 평가용_중요도.py | `importance.py` + `visualize.py` |
| 앱용_변환.py | `export_app.py` |
| (신규) 임계값별 지표 표 | `metrics.threshold_table`, `thresholds` 명령 |

원천 데이터 추출·합치기(전처리_1~4)는 원본이 수 GB라 데이터가 바뀔 때 한 번만 쓰는 스크립트여서 이번 정리 범위에 넣지 않았습니다.

## 테스트

```
python -m unittest discover -s tests -v
```

가짜 데이터로 전처리, 7·8월 변수 추가, 학습, 조합 비교, 임계값 표, 그래프, 앱용 변환까지 실행해 결과 형식을 검사합니다.
기존 스크립트와 같은 결과인지는 저장된 XGBoost 예측으로 확인했습니다(PR-AUC 0.4555, Recall 0.8149, Precision 0.2816, F1 0.4186).
