# 논문 기반 고정 설정·지리 군별 학습 실험

조사 기준일: 2026-10-31. 실행 진입점: `scripts/run_literature.py`.
이 실험은 기존 `run_pipeline.py`의 자동 탐색·G1/G2 재편 사이클과 **별개**다.
LightGBM 자체가 자동 파라미터 탐색기는 아니다. 기존 Optuna/후보 탐색 루프가 그 역할을 한다.
이번에는 사람이 근거를 정해 고정한 모델/입력/군 구성을 비교한다. 외부 평가 점수를 보고 자동으로 설정을 바꾸지 않는다.

## 최신 방법론 조사와 채택 판단

| 방법 | 확인한 원문·공식 구현 | 이 데이터에서의 판단 |
|---|---|---|
| **TabM, ICLR 2025** | [논문](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html), [공식 구현](https://github.com/yandex-research/tabm) | 표 형식 입력을 받는 신경망 앙상블. 현재 예보 입력으로 학습·추론 가능하므로 이번 실험에 구현. 일반 표 데이터 성과가 한국 태양광 성능을 보장하지는 않는다. |
| **TabICLv2, ICML 2026** | [논문, 2026-02 최초·09 개정](https://arxiv.org/abs/2602.11139), [공식 구현](https://github.com/soda-inria/tabicl) | 회귀 지원, 사전학습 표 모델. 공식 비교는 300~100,000개 표본 범위 중심이고 더 큰 문맥도 지원하지만, 현재 수백만 site-hour를 그대로 넣는 메모리·시간·정확도 검증이 필요. 이번 전량 학습 기본군에서는 보류. 과거 자료만 뽑는 별도 문맥 표본 추출 실험이 필요하다. |
| **TabPFN 3.5** | [공식 저장소](https://github.com/PriorLabs/TabPFN), [3.5 체크포인트](https://huggingface.co/Prior-Labs/tabpfn_3_5/commit/378b176c62ce7e5d85f7b394514f2b05883b2d67) | 2.5만을 최신으로 취급하지 않았다. 공식 안내의 2.5~3.5 가중치는 비상업 조건이므로, 대회의 외부 자원 조건에 맞는 권한을 확인하기 전 기본 채택하지 않는다. 대규모 문맥 비용도 별도 검증 대상. |
| **Cross-Unet, Nature Communications 2026** | [논문](https://www.nature.com/articles/s41467-026-73817-3), [저자 코드·실험 안내](https://github.com/ZjuMachine/PV-power) | PV 특화 다중 시간척도 encoder-decoder. 과거 PV/기상과 미래 기상·복사 입력을 활용한다. 현재 제출 기간 전체의 관측 PV 이력과 직접 일사량이 없으므로 원 논문 그대로 적용할 수 없다. 예보 전용으로 바꾸면 별도 변형 연구이며 원 논문 재현이라고 부르지 않는다. 출판사 본문 접근 제한으로 구현 판단은 저자 공개 코드/안내에 근거했다. |
| **Chronos-2, 2025** | [논문](https://arxiv.org/abs/2510.15821), [공식 코드](https://github.com/amazon-science/chronos-forecasting) | 다변량·미래 공변량을 지원하는 시계열 기반 모델. 현재 프로토콜에서 매일의 관측 발전량 이력이 계속 들어온다고 가정할 수 없다. 평가 실측을 context에 넣는 누출을 금지한다. 장기 재귀 추론 또는 예보 전용 변형을 별도 설계하기 전 기본 비교에서는 보류. |
| **CatBoost** | [저자 논문](https://arxiv.org/abs/1706.09516), [공식 회귀 문서](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor), [GPU 손실 지원](https://github.com/catboost/catboost/blob/master/catboost/docs/en/concepts/loss-functions-regression.md) | 최신 신경망이라고 포장하지 않는다. 다른 트리 학습 방식의 실용 대조군으로 GPU MAE를 구현했다. 입력·검증·군 구성을 동일하게 유지한다. |

우선순위는 **TabM + CatBoost + 작은 신경망**의 실제 동일 조건 비교다. 모델 규모나 논문 발표일만으로 우승 후보를 확정하지 않는다. 현재 실측 개선량은 GPU 학습 후에만 알 수 있다.

## 기존 조사 논문을 코드의 어디에 반영했는가

기존 `reference_application_review.md`의 R1~R7을 기준으로 반영/보류를 구분했다.

| 실험 ID | 실제 고정 설정 | 출처와 변경점 |
|---|---|---|
| `lgbm_l2` | trees=600, lr=.06, leaves=127, min_child=200, row/column sampling=.8, seed=42 | 기존 제공 baseline 값. **논문 최적값이 아니다.** 새 10/17 쌍 입력을 쓰는 공통 기준선. |
| `lgbm_l1` | 위 설정 유지, `objective=regression_l1`만 변경 | R1의 절대오차 대조. 개별 중앙값들의 합이 집합 중앙값이라는 가정은 하지 않고, 합산 W를 직접 평가한다. |
| `lgbm_revision` | L1 설정 유지; 17−10 예보 변화량 추가; 풍향은 sin/cos 및 그 차이 | R3의 제공 예보 기반 변형을 구현. 359°→1°를 −358°로 취급하지 않는다. 구름 영상 모델이나 고도별 구름 이동 모델을 재현했다는 뜻은 아니다. 인접 시각 차분·영상/AMV·강수량 추가는 이번 실험에 포함하지 않는다. |
| `catboost_mae` | 1200 trees, depth=8, lr=.03, L2=3, MAE, seed=42, GPU | **프로젝트 수동 제안값**. 특정 PV 논문에서 검증된 최적값이라고 주장하지 않는다. R4의 비-LightGBM 대조군. |
| `mlp10` | 은닉층 1개, hidden=10, ReLU, AdamW lr=.001/wd=0, MSE, batch=1024, 50 epochs | [Sharma §3.2](https://arxiv.org/html/2101.08236)의 작은 FCANN hidden=10을 채택. 원 논문의 과거 24시간·복사 입력 및 분위수 손실은 그대로 재현하지 않았다. ReLU/optimizer/lr/batch/epochs/MSE는 **프로젝트 변형값**. 작은 구조 대조 실험이다. |
| `tabm` | k=32, blocks=3, width=512, dropout=.1, ReLU; AdamW lr=.002/wd=.0003; MSE, batch=1024, 50 epochs | [공식 코드 기본값](https://raw.githubusercontent.com/yandex-research/tabm/main/tabm.py)과 [학습 안내](https://github.com/yandex-research/tabm). batch/epochs 및 현재 전처리·입력은 프로젝트 고정값. 각 멤버의 손실을 따로 계산한 뒤 평균하고, 추론에서만 멤버 출력을 평균한다. 논문의 전체 하이퍼파라미터 탐색을 재현하지 않는다. |

- **Gigoni**: [§III-D/F](https://arxiv.org/abs/1903.06800)의 QRF trees=300, minleaf=5, quantile=.4는 QRF 전용 설정이다. 이를 LightGBM/TabM에 옮기지 않았다. .4 분위수가 현재 대칭 절대오차 정산에 최적이라는 근거도 없다. QRF 자체 구현·P10/P90 구간 평가는 이번 실행군에 포함하지 않았다.
- **Lindberg**: [§2.2.1](https://uu.diva-portal.org/smash/get/diva2%3A1770882/FULLTEXT01.pdf)의 trees 400~2000/minleaf 10~20은 탐색 범위이며 고정 최적값이 아니다. 예측 오차와 거래 비용을 구분해 기록한다.
- **Simeunović**: [Appendix/Table III](https://arxiv.org/pdf/2107.13875v2)의 GCLSTM hidden32, GCTrafo kernel4/latent8/heads8, lr1e-4, 이웃15/24는 기록하되, 관측 이력이 없는 제출 조건에 원 그래프 모델을 억지로 끼워 넣지 않았다. 이번 지리 군 학습은 그래프 신경망이 아니다.
- **Nuño/R7**: 발전소별 독립 재표본 대신 모든 발전소를 합산한 동일 날짜의 후보/기준 오차를 함께 7일 블록으로 재표본한다. 7일·2000회는 프로젝트 선택값이며 논문이 보장한 최적 설정이 아니다.
- **R2** 태양 위치/계절 입력은 기존 공통 특징에서 유지. **R4의 블렌딩**, **R6의 집합 보정**은 이 고정 비교에 섞지 않는다. 기존 사이클과 후속으로 결합하려면 별도 과거 선택 구간이 필요하다.

## 지리 군: 큰 능선 분리 조건

사용자 확정 기준: 특정 산맥 이름을 지정하거나 우대하지 않고, 실제 지형에서 높은 능선을 우선적인 군 경계로 본다. 낮고 작은 능선은 무시할 수 있다. 현재 구현은 절대 고도와 양쪽 발전소 대비 상대 높이로 높은 장벽을 판별하고, 폭 조건으로 작은 지형을 걸러낸다. 아래 높이·폭 수치는 프로젝트 초기값이며 사용자가 지정한 수치는 아니다. 기준을 통과한 장벽은 모두 강제 분리하며, 통과한 장벽끼리 높이에 따라 추가 순위를 매기는 방식은 아니다.

`scripts/ridge_regions.py`는 기존 Copernicus GLO-30 DSM을 사용한다. DSM은 건물·수목이 포함되는 표면고도다. 1km 평균 격자로 옮기고 sigma=2km로 평활화하여 작은 지형의 영향을 줄인다.

공개 좌표 사이 거리가 80km 이하인 쌍에 대해 직선 경로를 약 1km 간격으로 조사한다. 양쪽 끝점보다 **300m 이상 높고**, 절대 고도 **500m 이상인 구간이 연속 3km 이상**이면 서로 다른 군에 배정해야 한다. 이 수치는 **수동 초기 지형 기준**이며 논문 최적값이 아니다. 실제 유역 분수령이나 명명된 태백/소백 산줄기를 추출한 결과는 아니다. 고지대·우회 경로·국지 기후는 이 단순 기준으로 완전히 표현되지 않는다.

좌표 오차(최대 약 2km)를 역추적하지 않는다. 거리 목적함수는 EPSG:5179 미터 좌표를 사용한다. 군의 공간적 연속성은 보장하지 않으며 먼 섬도 가장 가까운 군에 배정될 수 있다. 지도와 pair 목록을 검토할 수 있다. 군을 만들 때 발전량·평가 오차·가격은 사용하지 않는다.

고정 중심에서 거리 제곱합을 최소화하는 MILP와 중심 갱신을 최대 5회 반복한다. 능선 분리 조건은 항상 강제하며, 불가능한 K에서는 조건을 몰래 완화하지 않는다. 각 MILP 해의 가능성과 전체 군집 문제의 전역 최적성은 다른 개념이다.

현재 로컬 데이터 산출 결과:

| 구분 | K=2 | K=3 | K=4 |
|---|---|---|---|
| 거리만 | 187 / 66 | 111 / 63 / 79 | 107 / 35 / 77 / 34 |
| 능선 강제 분리 | 불가능 판정 | 불가능 판정 | **46 / 66 / 76 / 65** |

검사한 근거리 쌍 중 능선 장벽 1,111쌍, 지형 경로 결측 0쌍. 모든 장벽 쌍은 `ridge_k4`에서 다른 군이다. `data/ridge_regions_v1/`에 CSV/SVG/manifest를 생성했다. 이 디렉터리는 데이터 정책상 git 제외 대상이므로 학습 컴퓨터에서 생성하거나 함께 복사해야 한다. `manifest.json`이 배정 파일 해시를 보관한다.

이 **예측 군 4개**와 제출용 **G1/G2 2개**는 다르다. 이번 실험에서는 제출 배정을 reference 구성으로 고정하여 군별 학습 자체의 효과를 분리한다. 결측 발전소도 좌표로 배정되며 다른 군으로 오차 기반 재배정하지 않는다. CUDA1028 결과 없이 실행할 수 있다. 기존 오차 기반 sparse-peer/재편 실험은 별도 파이프라인에 남아 있다.

## 오차 감소를 판단하는 방식

기존 seasonal v2의 12개월(2024-06~2025-05)을 그대로 사용한다. 월별 학습 자료는 해당 월 origin의 관측 가용 cutoff 이전까지만 사용한다. 학습 대상 정규화/음수 clip/야간 0 처리는 공통 기존 규칙을 유지한다. 신경망의 중앙값 대치·평균·표준편차는 해당 학습 군의 과거 주간 자료에서만 fit하며, 각 열의 결측 지시자를 추가한다.

6개 고정 모델 × 5개 군 구성(global, distance K2/K3/K4, ridge K4) × 12개월이다. 현 구성은 총 1,008개 군별 모델을 fit하므로 수일 이상 걸릴 수 있다. 실행 시간은 각 작업의 done.json에 기록한다. wall-clock 예산이 같다는 비교는 아니며 군별 모델의 총 용량도 증가한다.

`fold_metrics.json`은 관측 발전소별 capacity-normalized MAE의 동일 가중 평균, 관측 site-hour 수, 실제 253개소가 모두 존재하는 시간 수, G1/G2별·전체 W/S를 남긴다. 결측 실측을 0으로 채워 점수를 만들지 않는다. W는 기존 README 식의 로컬 재현이며 공식 포털 채점값이라고 부르지 않는다.

`comparison.json`의 양수는 개선이다:

- `month_equal_nmae_reduction_pp`: 월 동일 가중 개별 NMAE 감소, 퍼센트포인트.
- `W_reduction_won`, `W_reduction_percent`: 동일 완전관측 시간의 합산 정산 오차 감소액·감소율.
- `nmae_months_improved`: 12개월 중 개선 월 수.
- `W_reduction_95pct_block_interval_won`: 월별로 7일 원형 블록을 2,000회 동시 재표본한 차이의 탐색적 구간. 여러 후보를 비교하므로 다중 검정 보정된 확정적 우위가 아니다. 구간이 0을 포함하면 안정적 개선이라고 단정하지 않는다.

기준은 global/LGBM L2, 같은 모델의 global, 능선 군에는 같은 K의 거리 군이다. 따라서 모델 변경·군 분리·능선 조건의 효과를 구분할 수 있다. 일부 월 실행은 `partial`; W의 완전관측 시간이 전혀 없으면 개선액은 null이다. 이후 파라미터를 이 점수에 맞춰 바꾸면 이 기간은 최종 독립 검증이 아니므로 새 holdout이 필요하다. 자동 우승 모델 선택·제출은 하지 않는다.

## 학습 컴퓨터 실행

`ai_sim` 디렉터리 기준. 기존 CUDA LightGBM 환경과 terrain 의존성을 사용한다. CatBoost와 TabM은 별도 설치한다. PyTorch는 학습 컴퓨터의 GPU/CUDA/아키텍처에 맞는 CUDA 빌드가 선행되어야 한다. CPU로 몰래 대체하지 않는다. 실제 설치 버전은 실행 manifest에 고정 기록된다.

```bash
python -m pip install catboost tabm==0.0.3
# 기존 terrain 데이터가 있고 ridge_regions_v1이 아직 없을 때만 실행
python scripts/ridge_regions.py

# 실제 GPU tiny fit/forward/backward 및 데이터·군 구성 검사
python scripts/run_literature.py --output-dir results/literature_fixed_v1 --preflight-only
# 전체 실험, 위 manifest 그대로 이어서 실행
python scripts/run_literature.py --output-dir results/literature_fixed_v1 --resume
```

구성 전체를 한 번에 실행해도 되고 다음처럼 두 묶음으로 나눠도 된다. 두 묶음의 공통 LGBM L2 기준선은 각각 재학습된다.

```bash
python scripts/run_literature.py --output-dir results/literature_trees_v1 --models lgbm_l2 lgbm_l1 lgbm_revision catboost_mae
python scripts/run_literature.py --output-dir results/literature_neural_v1 --models mlp10 tabm
```

중단 후 같은 명령에 `--resume`을 추가한다. 완료된 월/구성/모델의 예측·저장 모델 해시를 확인해 재사용하며, 중간에 실패한 작업은 해당 구성부터 다시 학습한다(신경망 epoch 중간 resume 아님). 모델 저장 후 다시 불러온 출력과 원래 출력을 비교한다. 입력·코드·버전·설정을 변경했다면 새 output-dir을 사용한다. 한 달 시험은 `--fold outer_2024_06`; 별도 디렉터리에서 실행하고 이를 12개월 결과로 해석하지 않는다.

현재 로컬에는 CUDA LightGBM 학습 기능과 torch/tabm/catboost가 준비되지 않아 실제 GPU 모델 성능은 미측정이다. 지형 분할과 CPU에서 가능한 데이터/평가/누출 방지 테스트는 실행했다. 학습 컴퓨터의 preflight 및 전체 학습 성공 전까지 TabM/CatBoost 실전 실행이 검증되었다고 주장하지 않는다.

검증 기록: 전체 unittest 72개 중 71개 통과·CUDA 통합 1개 건너뜀. 신규 6개 테스트는 능선 강제 제약/불가능 K, 풍향 원형 변화량, 제외 입력 불변성, 학습 전처리 격리, 시간 말단의 날짜 정렬 및 W 차이, 학습 cutoff·군 격리·저장 결과 재사용/변조 감지를 검사한다. 실제 입력과 생성된 253개소 군 배정 검사를 통과한 로컬 preflight는 `CUDA Tree Learner was not enabled in this build`에서 중단했다. 기존 지형 원본 manifest의 타일 해시도 대조했고 VRT 포함 28개 참조 파일 해시를 군 manifest에 기록했다.
