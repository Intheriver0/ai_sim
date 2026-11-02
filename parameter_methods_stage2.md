# 2차 구현: 현재 환경에서 가능한 방법

2026-10-31. 새 venv 또는 기존 패키지 다운그레이드 없이 추가했다. 기존 모델과 실행기는 유지한다. 미구현 항목과 재개 조건은 [미결 목록](parameter_methods_pending.md)에 기록한다.

## GP Bayesian optimization

`--methods gp`: 설치된 Optuna GPSampler. 초기 random 10회 후 GP 제안을 사용한다. 기존 3축 공간과 최대 epochs 평가를 유지한다. trials가 10 이하이면 GP 제안까지 실행되지 않는다. GPU 연산의 완전 결정성을 가정하지 않도록 deterministic_objective=False를 명시했다. scipy/PyTorch만 사용하며 선택적인 greenlet이 없으면 순차 acquisition 최적화를 수행한다. [공식 설명](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.GPSampler.html)

## 한 단계 hypergradient

`--model mlp_hypergradient`: log(lr), log(L2)를 한 단계 SGD의 validation loss로 미분해 갱신한다. 각 학습 cutoff **안에서** 고유 timestamp의 앞 75%를 support, 뒤 25%를 내부 validation으로 분리한다. 모델 가중치는 support에서, 연속 설정은 내부 validation에서 학습한다. 외부 평가 및 HPO 선택 구간은 이 내부 분할에 들어가지 않는다.

초기 lr=.001/L2=.0003, hyper-Adam lr=.01, lr 범위 [1e-4,.01], L2 범위 [1e-6,.01]. 프로젝트 시작값이며 논문 최적값이 아니다. **1-step truncated bilevel 변형**으로, [Maclaurin 2015](https://proceedings.mlr.press/v37/maclaurin15.html)의 전체 궤적 reversible differentiation 재현이 아니다. 외부 학습 origin마다 모델과 controller를 새로 학습한다.

PBT는 controller의 설정과 충돌하므로 함께 사용하면 CLI에서 거절한다. `--methods fixed` 또는 `--methods fixed random tpe gp`를 명시한다. 고정 AdamW 및 고정 SGD MLP를 자동으로 외부 평가한다. 두 기준 모델은 학습 cutoff 전체 자료를 사용하므로 이는 전체 절차 비교이며 순수 optimizer만 바꾼 통제 실험은 아니다. SGD는 coupled L2, AdamW는 decoupled decay다.

## Coordinatewise LSTM learned optimizer

`--model mlp_learned_optimizer`: 2층 LSTM(hidden20)이 좌표별 gradient 이력으로 실제 가중치 갱신량을 생성한다. log/sign 전처리(p=10), 좌표 간 파라미터 공유, gradient 입력의 2차 미분 생략, unroll 경로를 통한 meta-gradient는 [Andrychowicz 2016](https://arxiv.org/html/1606.04474v2)을 따른다.

태양광 적용은 **프로젝트 변형**이다. 군별 학습 cutoff 내 발전소를 task로 사용한다. 각 task의 앞 75% support와 이후 25% query에서 meta-training한다. 각각 최대 64개, 기본 3 meta epochs × 100 task updates, unroll 5회, meta-Adam lr=.001, gradient norm clip1. 출력은 .01*tanh로 제한하고 lr/.001을 곱하며 decoupled weight decay를 적용한다. 초기 MLP는 task마다 새로 생성한다. 네 군만을 네 개의 meta 표본으로 취급하지 않는다.

meta-training 후 updater를 고정하고 해당 군의 전체 학습 자료로 예측 MLP를 학습한다. 평가 정답은 사용하지 않는다. 구조/seed/epochs가 같은 고정 AdamW도 자동 평가한다. meta-training 비용이 추가되므로 같은 epochs를 같은 계산 비용이라고 해석하면 안 된다. 발전소 holdout 일반화 및 미관측 지역 전이 성능은 아직 평가하지 않았다.

## 저장과 비교

체크포인트에는 hypergradient controller/Adam 상태 또는 학습된 LSTM/좌표별 recurrent 상태/meta 완료 여부를 저장한다. meta-training 도중 중단되면 같은 seed로 다시 수행한다. meta-training 완료는 epoch 0 체크포인트로 저장한다. 재개와 PBT 복제는 전체 상태를 함께 이동한다.

`comparison.json`의 각 행에 `reference`를 기록한다. fixed는 같은 방법의 고정 설정, reference_mlp는 고정 AdamW, reference_mlp_sgd는 고정 SGD다. 외부 점수로 우승 모델을 다시 선택하지 않는다. 실제 적용한 학습률은 군별 state.pt의 schedule에 저장한다. NMAE/W 및 날짜 블록 신뢰구간은 기존 평가를 유지한다. 동일 GPU-hour 비교는 아직 구현하지 않았다.

## 실행

`ai_sim`에서 실행한다. 기존 환경을 낮추는 설치 단계는 없다. 학습 머신에는 CUDA PyTorch가 있어야 한다.

```bash
python scripts/run_parameter_methods.py --model mlp --methods fixed random tpe gp smac smac_mf hyperband tpe_hyperband asha pbt --output-dir results/parameter_search_v2
python scripts/run_parameter_methods.py --model mlp_hypergradient --methods fixed --output-dir results/parameter_hypergradient_v2
python scripts/run_parameter_methods.py --model mlp_learned_optimizer --methods fixed --output-dir results/parameter_learned_optimizer_v2
```

먼저 `--fold outer_2024_06 --preflight-only`를 추가해 점검할 수 있다. 같은 출력 경로에서 본실행하려면 같은 fold/options를 유지하고 preflight-only 대신 `--resume`을 사용한다. 전체 12개월은 별도 출력 경로에서 시작한다.

BOHB 설치 시 netifaces 빌드 도구 부재로 실패했다. 별도 venv가 필수라고 단정한 것이 아니라 현재 머신의 빌드 문제다. 패키지 다운그레이드나 새 venv 생성은 하지 않았다.

## 검증 결과

전체 회귀 테스트 84개 중 83개 통과, CUDA 통합 1개 건너뜀. 추가 검증은 두 연속 설정의 hypergradient와 중앙 수치 미분 일치, LSTM meta-gradient 및 recurrence, 새 모델의 실제 학습/저장/재로딩, 평가 label 변경 불변성, 연속 학습 대 중단 재개/복제의 정확 일치, GP 초기 10회 이후 실제 모델 기반 제안까지 포함한다.

로컬 NumPy 2.5.4, pandas 3.1.0, torch 2.15.0+cpu, Optuna 5.0.0, ConfigSpace 1.2.2 버전은 유지됐다. 이는 CPU 기능 검증이며 본 데이터의 GPU 학습 또는 오차 개선 실증은 아니다.
