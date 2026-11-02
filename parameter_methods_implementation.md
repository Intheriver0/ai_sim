# 파라미터 방법론 구현: 첫 실행 단계

기존 `run_pipeline.py`, `run_literature.py`, LightGBM/CatBoost 고정 실험을 수정하지 않고 별도 실행기 `scripts/run_parameter_methods.py`를 추가했다. 이번 실행기는 **신경망** 대상이다. 기존 트리 탐색을 이 실행기로 교체한 것은 아니다.

## 구현 범위와 검증 수준

| 방법/모델 | 이번 상태 |
|---|---|
| 고정 설정, Random, TPE | 공통 목적함수에 연결, 실제 라이브러리 기반 작은 목적함수 테스트 |
| ASHA | Optuna SuccessiveHalvingPruner로 연결, 단일 작업자 순차 실행. 대규모 비동기 병렬 처리 속도를 검증했다는 뜻은 아님 |
| Hyperband | RandomSampler + HyperbandPruner 연결 |
| TPE + Hyperband | 별도 `tpe_hyperband` 이름. **원 BOHB 구현과 동일하다고 부르지 않음** |
| SMAC / SMAC multi-fidelity | SMAC3 공식 facade 연결, 2.4.1에서 작은 목적함수 실제 실행 검증 |
| PBT | 동기식 집단 학습 구현. 후보 하나가 모든 군·과거 검증 창의 모델 묶음. 하위 25%(최소 1개)가 상위 25%의 모델·optimizer 상태를 이어받고 lr/wd에 .8 또는 1.2 변이 적용 |
| MLP / TabM | 실제 작은 자료로 학습·저장·재개·예측 검증. TabM 멤버별 MSE 후 평균 |
| Hypernetwork head | 공통 encoder + 위치·지형 조건으로 작은 출력층 가중치 생성. 원 논문의 PV 재현이 아닌 프로젝트 변형. 실제 gradient/학습 검증 |
| MAML | 신경망에 정확한 고차 미분 inner update 적용. 발전소별 시간순 support/query로 meta-training. 실제 고차 gradient와 예측 검증 |
| DEHB | 공식 호출 adapter만 작성. 공식 0.1.2가 NumPy<2를 요구하여 현재 NumPy 2 환경에서 미검증. 기본 실행 목록에서 제외 |
| GP / 1-step hypergradient / learned optimizer | [2차 구현](parameter_methods_stage2.md)에 추가. 아래의 80개 테스트 기록은 1차 구현 시점 |
| 원 BOHB, p-diff, GHN, NNiT | [미결 목록](parameter_methods_pending.md)에 환경·자료·연구 이유를 구분 |

일반 Hypernetwork의 일부 가중치 생성과 MAML을 먼저 실행 가능한 형태로 만들었다. 더 큰 생성 모델에 필요한 체크포인트 bank/구조 자료 구축, 직접 생성 모델 비교는 후속 작업이다. 학습에 성공하는 것과 태양광 오차가 개선되는 것은 다르며, 실제 GPU 성능 수치는 아직 없다.

## 공통 학습과 검증

- 입력: 기존 10시/17시 쌍 입력. 기존 제외 기상 항목을 계속 제외한다.
- 기본 지역 구성: `ridge_k4`. 기존 거리 군·전체 통합 모델도 `--region`으로 선택 가능하다. 분리 불가능한 K를 요청하면 거절한다.
- 신규 신경망 공통 기준선: hidden64, lr=.001, weight_decay=.0003, batch1024, seed42. 기존 논문 실험의 MLP hidden10 기준선과 구분한다.
- 공통 탐색 공간: hidden ∈ {16,32,64,128,256}, lr ∈ [1e-4,.01] log, weight_decay ∈ [1e-6,.01] log. 이는 프로젝트 초기 공간이며 논문 최적값이 아니다. TabM의 k32/blocks3/dropout.1은 유지한다.
- 한 후보의 점수는 각 과거 창에서 **군 전체**를 예측해 계산한다. 군별 점수 크기로 다른 군의 모델을 무작정 복사하지 않는다.
- 외부 월마다 기존 seasonal v2의 이전 12개 `model_selection` 구간만 탐색에 사용한다. 그 이후 외부 월 학습 cutoff까지 새로 학습하고 외부 월을 평가한다. 외부 결과로 후보를 다시 고르지 않는다.
- 기본 선택 목적 `--objective nmae`: 관측 개소별·월별 동일 가중 NMAE proxy다. 실제 입력 검사에서 `outer_2024_06`의 과거 선택 구간에는 253개소 완전관측 시간이 없었다. 그래서 기본값을 명시적으로 proxy로 정했으며, W로 선택했다고 표현하지 않는다. `--objective W`를 명시하면 과거 완전관측 시간의 정산 W 합/시간 수로 선택하고 전체 선택 구간에 해당 시간이 없으면 오류로 중단한다. 자동 fallback은 없다. 외부 평가에서는 두 지표를 함께 기록한다.
- 저예산 후보의 점수와 전체 예산 후보의 점수를 섞어 우승 후보를 고르지 않는다. 최대 epochs까지 평가한 후보만 선택 대상이며 고정 기준 후보를 항상 포함한다. 탐색 예산이 작아 승격 후보가 없으면 기준선이 선택될 수 있다.
- Hypernetwork는 공개 좌표·log capacity와 기존 검증된 정적 terrain profile을 조건으로 받는다. 학습 구간에서만 전처리를 fit한다. 평가는 같은 모델의 고정 설정 대비이며, 별도 MLP 결과와 비교해야 Hypernetwork 구조 자체의 효과를 알 수 있다.
- MAML은 발전소별 과거 daylight 자료의 앞 75%에서 support, 뒤 25%에서 query를 뽑는다. 한 meta epoch는 100개 task update, support/query는 각각 최대64개, inner_lr=.01, inner_steps=1이다. 단순 MLP의 전량 1 epoch와 계산량이 같지 않다. 4개 미만 관측 발전소는 meta-training task에서 제외한다. 추론 적응에는 그 발전소의 마지막 과거 관측 최대256개만 쓰고, 관측이 없으면 공통 초기 모델을 쓴다. 평가 label은 적응에 넣지 않는다.

## PBT와 체크포인트

기본 epochs 단계는 5→15→45다. PBT 집단 크기는 4이며 각 단계에서 같은 과거 검증 창으로 평가한다. 구조 hidden64를 고정한 채 lr/wd만 변이한다. 최종 선택된 가중치를 외부 평가에 그대로 들고 가지 않고, **선택된 학습 스케줄을 외부 학습 자료에서 처음부터 재학습**한다.

모델·optimizer·epoch·스케줄·입력 해시·전처리 값을 저장한다. 매 epoch seed를 고정하므로 CPU 테스트에서 연속 학습과 중단/재개의 출력이 정확히 일치했다. GPU의 수치적 완전 일치는 아직 검증하지 않았다. 체크포인트 SHA 불일치나 학습 자료/설정 불일치는 중단한다. 저장 도중 장애로 체크포인트/해시가 어긋나도 조용히 넘어가지 않는다.

PBT 복제는 같은 검증 창·같은 군의 donor 상태를 받으며, 후보의 묶음 점수로 donor를 결정한다. 다른 시점의 학습 자료로 만든 상태를 로드하지 않는다. 모델 저장 후 재로딩 결과와 예측을 비교한다.

중단한 탐색은 같은 seed로 제안 과정을 재생하고, 완료된 후보/예산 예측 캐시를 재사용한다. 일부 라이브러리의 버전별 제안 순서 차이를 막기 위해 코드·설치 버전·입력 manifest가 달라지면 재개를 거절한다. 동일 output-dir에 여러 프로세스를 동시에 실행하지 않는다.

## 실행

학습 컴퓨터에서 CUDA PyTorch를 먼저 준비한 뒤 `ai_sim` 기준으로 실행한다. PyTorch를 이 requirements 파일로 CPU 버전까지 임의 교체하지 않는다.

```bash
python -m pip install -r requirements-parameter-methods.txt

# 우선 한 달 실행 전 데이터·GPU 검사. 이후 같은 명령에 --resume을 붙여 본실행.
python scripts/run_parameter_methods.py --model mlp --methods fixed random tpe asha pbt --fold outer_2024_06 --output-dir results/parameter_mlp_trial --preflight-only
python scripts/run_parameter_methods.py --model mlp --methods fixed random tpe asha pbt --fold outer_2024_06 --output-dir results/parameter_mlp_trial --resume

# 전체 12개월: 처음부터 별도 출력 경로 사용
python scripts/run_parameter_methods.py --model tabm --methods fixed random tpe smac smac_mf hyperband tpe_hyperband pbt --output-dir results/parameter_tabm_v1

# 구조/적응 모델은 먼저 고정 설정으로 별도 비교
python scripts/run_parameter_methods.py --model hyper_head --methods fixed --output-dir results/parameter_hyper_head_v1
python scripts/run_parameter_methods.py --model maml --methods fixed --output-dir results/parameter_maml_v1
```

`--trials 24`, `--min-epochs 5`, `--max-epochs 45`, `--population 4`, `--seed 42`, `--threads 4`가 기본이다. PBT 작업량은 trials가 아니라 population과 epochs 단계로 결정된다. `fixed`는 항상 포함된다. 한 달 시험 결과는 `partial`, 12개월 비교만 `complete`로 표시한다.

`fold_metrics.json`과 `comparison.json`에 선택 설정·과거 선택 점수·외부 NMAE·W 개선액/개선율·날짜 블록 신뢰구간을 남긴다. 비교 기준은 **같은 모델/지역의 고정 설정**이다. `trace.json`은 후보·budget·점수·호출 소요 시간을 기록한다. 캐시를 공유하므로 호출 시간만으로 최적화 알고리즘 속도 순위를 매기면 안 된다. 현재 실행기는 동일 GPU-hour 제한 비교 기능까지 구현한 상태가 아니며, trials 수 역시 알고리즘 간 완전히 같은 계산량을 뜻하지 않는다.

## 남은 검증과 제약

DEHB의 [공식 의존성](https://raw.githubusercontent.com/automl/DEHB/master/pyproject.toml)은 NumPy<2다. 기존 NumPy 2 환경을 강제로 낮추지 않았다. 호환 환경 분리와 adapter 실제 테스트가 끝나기 전 DEHB 실전 성공을 주장하지 않는다. BOHB와 나머지 생성 방법의 구현 상태는 위 표대로다.

로컬에 CPU PyTorch 2.15.0, TabM 0.0.3, SMAC 2.4.1, ConfigSpace 1.2.2를 설치해 작은 실행 테스트를 수행했다. production CLI는 CPU fallback을 제공하지 않는다. GPU 본학습/실제 개선율은 학습 컴퓨터에서 검증해야 한다.

전체 회귀 테스트 80개 중 79개 통과, CUDA 통합 1개 건너뜀. 신규 테스트는 4개 신경망의 실제 학습/재로딩, Hypernetwork 조건별 가중치 및 gradient, MAML 고차 gradient와 평가 정답 불변성, TabM 멤버별 손실, 연속/재개 학습 일치, PBT optimizer 복제 및 외부 스케줄 재학습 일치, 체크포인트/선택 파일 변조 감지, 과거 구간 제한, 저예산 점수의 최종 선택 배제를 확인했다.

실제 로컬 입력으로 NMAE preflight를 실행해 전처리·지역 배정·시간 분리 검사를 통과했고, CUDA PyTorch 부재에서 의도대로 중단했다. GPU에서 본학습이 시작된 상태는 아니다.
