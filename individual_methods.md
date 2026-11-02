# 방법론별 개별 실행

전체 일괄 실행기 `run_all.py`는 제거했다. 아래 명령 중 원하는 방법 하나를 골라
`ai_sim` 폴더에서 실행한다. 명령 블록 전체를 한꺼번에 실행하는 안내가 아니다.
출력 폴더는 방법별로 분리하며, 통합 로그·통합 보고서를 자동 생성하지 않는다.
이미 생성한 결과는 삭제하지 않았다.

## 신경망: 먼저 방법 하나씩 고정 설정으로 비교

```bash
python scripts/run_parameter_methods.py --model mlp --methods fixed --output-dir results/mlp_fixed_v1
python scripts/run_parameter_methods.py --model tabm --methods fixed --output-dir results/tabm_fixed_v1
python scripts/run_parameter_methods.py --model hyper_head --methods fixed --output-dir results/hyper_head_fixed_v1
python scripts/run_parameter_methods.py --model maml --methods fixed --output-dir results/maml_fixed_v1
python scripts/run_parameter_methods.py --model mlp_sgd --methods fixed --output-dir results/mlp_sgd_fixed_v1
python scripts/run_parameter_methods.py --model mlp_hypergradient --methods fixed --output-dir results/hypergradient_fixed_v1
python scripts/run_parameter_methods.py --model mlp_learned_optimizer --methods fixed --output-dir results/lstm_optimizer_fixed_v1
```

`--methods`를 생략해도 이제 fixed 하나만 실행한다. Hypergradient는 고정 AdamW·SGD,
LSTM learned optimizer는 고정 AdamW 대조군을 함께 평가하는 기존 절차를 유지한다.

**fixed 단독 실행은 내부 12개월 후보 평가를 생략한다.** 외부 12개월 각각에서
그 월 이전 자료로 지역군별 모델을 새로 학습하고 평가한다. 기본 ridge_k4에서는
주 모델 학습이 624회에서 48회로 줄어든다(사전 점검·추가 대조군 제외).
모델의 자체 학습 과정, 기본 45 epoch, 외부 평가 범위는 바꾸지 않았다.
선택 점수는 `null`, 내부 평가 횟수는 0으로 저장하며, 외부 NMAE/W는 기존대로 기록한다.
다른 탐색 방법과 함께 실행하는 fixed 기준선은 내부 평가를 유지한다.

두 컴퓨터에 같은 코드·데이터·지역군·라이브러리 버전을 준비하고 다른 모델을 실행한다.
예를 들어 컴퓨터 1에서는:

```bash
python scripts/run_parameter_methods.py --model mlp --methods fixed --output-dir results/mlp_fixed_outer_v1
```

컴퓨터 2에서는:

```bash
python scripts/run_parameter_methods.py --model tabm --methods fixed --output-dir results/tabm_fixed_outer_v1
```

학습 예산·평가 범위 옵션을 추가할 필요는 없다. 완료 후 각 폴더 전체를 서로 다른 이름으로
모아 `fold_metrics.json`의 같은 외부 월별 성능을 비교한다. `comparison.json`은
해당 실행 내부의 기준선 대비 차이이므로 fixed 단독 실행의 자기 비교는 0이다.
이전 코드로 생성한 실행 폴더는 새 실행에 재사용하지 않는다.

## 탐색 방법도 하나씩 실행

모델과 탐색 방법을 각각 선택한다. 예를 들어 MLP에 TPE만 적용하려면:

```bash
python scripts/run_parameter_methods.py --model mlp --methods tpe --output-dir results/mlp_tpe_v1
```

`tpe` 자리에 `random`, `gp`, `asha`, `hyperband`, `tpe_hyperband`, `smac`,
`smac_mf`, `pbt` 중 하나를 지정한다. 결과 폴더명도 함께 바꾼다.
선택한 탐색 방법 외에 fixed 기준선은 비교를 위해 자동 포함된다.
Hypergradient와 PBT 조합은 지원하지 않는다. DEHB는 차단 상태이고 BOHB 등 미구현 방법은 실행하지 않는다.

기본은 외부 12개월, 내부 선택 12개월, trials 24, epochs 5~45, PBT population 4다.
방법 하나도 계산량이 크다. 축소 실행 예시는 다음과 같다.

```bash
python scripts/run_parameter_methods.py --model mlp --methods tpe --fold outer_2024_06 --trials 2 --min-epochs 1 --max-epochs 2 --output-dir results/mlp_tpe_small_v1
```

이는 작동 확인용 부분 결과이며 12개월 본실험 결과와 직접 비교하지 않는다.
GP는 초기 random 10회 이후부터 GP 제안을 사용하므로 trials 2로 GP 동작을 검증할 수 없다.

## 고정 모델 실험도 하나씩 실행

```bash
python scripts/run_literature.py --models lgbm_l2 --output-dir results/literature_lgbm_l2_v1
python scripts/run_literature.py --models lgbm_l1 --output-dir results/literature_lgbm_l1_v1
python scripts/run_literature.py --models lgbm_revision --output-dir results/literature_revision_v1
python scripts/run_literature.py --models catboost_mae --output-dir results/literature_catboost_v1
python scripts/run_literature.py --models mlp10 --output-dir results/literature_mlp10_v1
python scripts/run_literature.py --models tabm --output-dir results/literature_tabm_v1
```

`--models`를 생략하면 이제 lgbm_l2 하나만 실행한다. 다른 모델을 고르면
lgbm_l2 기준선을 함께 평가한다. 가능한 지역 군별 비교와 12개월 평가는 유지한다.
`--fold outer_2024_06`으로 한 달만 지정할 수 있다.

## 기존 그룹 재편 비교

```bash
python scripts/run_pipeline.py --stage compare --output-dir results/group_cycles_v1
```

이 실행기는 기존 별도/쌍 입력 기준선, 그룹 재편·재학습, 특성 군집 비교 절차를 유지한다.
새 신경망 실험을 이어서 실행하지 않으며 `--stage compare`에서는 제출 파일을 생성하지 않는다.
예산은 `--trials`, `--cycles`, `--trees`, `--search-steps`로 조절한다.

## 사전 점검·재개·로그

모든 실행기의 CPU 보조 스레드 기본값은 4다. 필요할 때만 `--threads`를 지정한다.
신경망 및 고정 모델 명령에 `--preflight-only`를 붙이면 사전 점검만 수행한다.
점검 후 같은 명령에서 `--preflight-only`를 빼고 `--resume`을 붙여 본실험을 진행한다.
그룹 재편 실행기는 `--stage preflight`로 점검한 후 `--stage compare --resume`으로 진행한다.
처음부터 본실험을 시작할 때는 새 출력 폴더를 사용하고 `--resume`을 붙이지 않는다.

각 방법의 터미널 출력을 저장하면서 백그라운드로 실행하려면, 예를 들어:

```bash
nohup python -u scripts/run_parameter_methods.py --model mlp_learned_optimizer --methods fixed --output-dir results/lstm_optimizer_fixed_v1 > lstm_optimizer_fixed_v1.log 2>&1 < /dev/null &
```

기존 출력 경로는 코드·입력·설정·라이브러리 일치 검사를 통과해야 재개할 수 있다.
이미 통합 실행으로 만든 결과도 이 검사를 우회해 재개하지 않는다.

## 결과를 직접 비교할 때

각 출력 폴더의 `comparison.json`과 신경망/고정 모델의 `fold_metrics.json`을 확인한다.
그룹 재편 실행기는 `COMPARISON.md`도 생성한다. 같은 데이터·외부 월·평가 방식·예산인지
확인한 뒤 비교한다. 부분 평가와 전체 평가, 지역 군이 다른 실험을 하나의 순위로 섞지 않는다.
같은 epochs/trials는 같은 GPU 계산량을 뜻하지 않는다.

데이터와 설치 환경은 이전과 같다. 전처리 자료, 지역 군 자료, 지형 자료가 필요하고,
그룹 재편 비교에는 완료된 CUDA1028 결과도 필요하다. 입력 항목은
[최종 입력 점검 기록](final_audit_2026-11-02.md), 추가 설치는
[신경망 설치 안내](parameter_methods_implementation.md)와 [고정 모델 안내](literature_experiments.md)를 따른다.
