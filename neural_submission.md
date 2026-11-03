# TabM → Hyper-head → MLP: 개별 G1/G2 검증과 제출

`scripts/run_neural_submission.py`는 모델 하나를 고정해 G1/G2 배정 선택,
외부 12개월 진단, 최종 재학습, 제출 CSV 생성과 재읽기 검증을 수행한다.
첫 실행은 TabM이다. 세 모델을 자동으로 섞거나 외부 점수를 보고 모델을 바꾸지 않는다.

## 첫 실행: TabM

CUDA가 준비된 학습 컴퓨터의 `ai_sim` 폴더에서 실행한다.
기존 [신경망 환경](parameter_methods_implementation.md)과 전처리 자료,
`data/ridge_regions_v1/`가 필요하다. Hyper-head는 `data/terrain_features_v2/`도 필요하다.

```bash
python -u scripts/run_neural_submission.py --model tabm --output-dir results/tabm_submission_v1
```

명령 하나로 사전 점검부터 제출 파일까지 진행한다. 기본값은 기존 실험과 같은
ridge_k4, hidden=64, lr=0.001, weight_decay=0.0003, 45 epoch, seed=42, threads=4다.
G1/G2 탐색은 각 시점에서 20,000회다. CPU로 자동 대체하지 않는다.

기존 TabM 결과 폴더에 **본학습 예측 cache까지 남아 있다면**, 첫 실행부터 다음처럼 지정한다.
JSON만 Git에서 받은 경우에는 위 기본 명령을 사용하면 된다.

```bash
python -u scripts/run_neural_submission.py --model tabm --prediction-run results/tabm_fixed_outer_v1 --output-dir results/tabm_submission_v1
```

기존 실행의 모델·학습 예산·입력·지역군·학습 코드·라이브러리·시간 분할을 대조한다.
일치하고 완료 해시가 유효한 월별 예측만 새 실행 폴더에 복사한다.
기존 12개월보다 앞선 과거 예측과 최종 제출 모델은 새로 학습한다.
누락된 과거 cache는 재학습하지만 변조된 완료 cache는 오류로 중단한다.
Hyper-head의 기존 manifest에 지형 해시가 없으면 재사용하지 않는다.

기본 달력에서 월별 학습은 2022-06~2025-05의 36개 월이다.
같은 월 모델은 배정 탐색·선택·외부 진단에서 공유하므로 전부 다시 학습하지 않는다.
재사용 없이 실행하면 지역군 4개 × (36개월 + 최종 모델 1회) = **148회** 본학습이며,
기존 12개월 예측을 모두 재사용하면 **100회**다. 각 본학습의 45 epoch는 줄이지 않는다.
사전 점검용 작은 학습은 별도다. 필요한 GPU 시간과 디스크 공간을 확보한다.

## 중단 후 재개 / 단계별 실행

중단된 명령에 `--resume`을 추가한다. 최초에 지정한 옵션은 유지한다.
예측 파일은 월별로, 모델은 epoch마다 저장한다. 최종 제출은 같은 cutoff에서 한 번
학습한 모델로 테스트 1년 전체를 예측한다.

```bash
python -u scripts/run_neural_submission.py --model tabm --output-dir results/tabm_submission_v1 --resume
```

사전 점검, 비교, 제출을 따로 실행할 수도 있다.

```bash
python -u scripts/run_neural_submission.py --model tabm --output-dir results/tabm_submission_v1 --stage preflight
python -u scripts/run_neural_submission.py --model tabm --output-dir results/tabm_submission_v1 --stage compare --resume
python -u scripts/run_neural_submission.py --model tabm --output-dir results/tabm_submission_v1 --stage submit --resume
```

비교 12개월이 완료되어야 제출 단계가 열린다. 코드·입력·환경·옵션이 바뀌면 새 결과
폴더를 사용한다. 동일 폴더 동시 실행은 `run.lock`으로 차단한다.
프로세스 강제 종료나 정전으로 lock이 남았다면 **기존 프로세스가 완전히 종료됐는지
확인한 후에만** 해당 결과 폴더의 `run.lock`을 지우고 재개한다.

백그라운드 실행이 필요하면:

```bash
nohup python -u scripts/run_neural_submission.py --model tabm --output-dir results/tabm_submission_v1 > tabm_submission_v1.log 2>&1 < /dev/null &
```

## 다음 두 모델

각각 별도 폴더에서 실행한다. GPU 메모리를 공유하므로 기본적으로 순서대로 실행한다.

```bash
python -u scripts/run_neural_submission.py --model hyper_head --output-dir results/hyper_head_submission_v1
python -u scripts/run_neural_submission.py --model mlp --output-dir results/mlp_submission_v1
```

## G1/G2 선택과 평가

학습 지역군 ridge_k4와 정산용 G1/G2는 별개다. 이번 실행은 학습 지역군과 모델 설정을
고정하고, 발전소별 예측을 정산용 두 그룹으로 합산하는 배정만 바꾼다.
배정 변경 때문에 같은 월 신경망을 다시 학습할 필요는 없다.

각 외부 평가 월에 대해:

1. 그 월의 24개월 전~12개월 전 월별 out-of-sample 예측으로 배정을 탐색한다.
2. 그다음 과거 12개월에서 제안 배정과 기존 배정을 비교한다. 동점은 기존 배정을 유지한다.
3. 선택을 `selected.json`에 먼저 확정하고 해당 외부 월을 평가한다.

각 월의 모델은 예측 시작 전날 17시까지 이용 가능한 실측만 학습한다.
그룹 탐색의 마지막 실측도 첫 선택 월의 학습 cutoff 이하로 잘라낸다.
최종 배정 선택에서도 제출 시점에 아직 이용할 수 없는 마지막 실측은 제외한다.

탐색·선택은 해당 기간 95% 이상 관측된 발전소의 공통 관측 시간에서
SMP×k로 가중한 그룹 절대오차를 사용한다. 이것은 **관측 패널 W 대리지표**이며
253개 전체의 정산금이 아니다. 탐색 패널에 없는 발전소는 기존 배정을 유지한다.
전체 253개 발전소의 용량으로 각 그룹 20,000 kW 이상 제약을 적용한다.
외부 평가는 전체 실측이 있는 시간의 W와 관측 발전소별 NMAE를 별도로 기록한다.

이미 2024-06~2025-05 성능을 보고 세 모델을 추렸으므로, 이 12개월 결과는
새 독립 최종 검증으로 주장하지 않는다. 모델 선택을 포함한 최종 일반화 성능은
미공개 테스트 실적 또는 별도의 새로운 자료에서만 확인할 수 있다.
G1/G2 선택 자체는 매 외부 월 이전 자료로 제한한다.

## 최종 제출 파일

TabM의 경우:

```text
results/tabm_submission_v1/submission/composition.csv
results/tabm_submission_v1/submission/forecast.csv
results/tabm_submission_v1/submission/validation.json
results/tabm_submission_v1/submission/done.json
```

`done.json`의 `status`가 `complete`일 때 **같은 폴더의 CSV 두 개를 한 쌍으로 제출**한다.
다른 모델의 composition과 forecast를 섞지 않는다. 포털 업로드는 자동 수행하지 않는다.

- `composition.csv`: `site_id,group`, PV001~PV253 중복·누락 없이 253행.
- `forecast.csv`: `group,ts,forecast_kwh`, G1/G2 × 테스트 8,760시간 = 17,520행.
- 타임존 포함 KST, 정확한 시간별 키, 비음수·유한 예측값, 두 그룹 각각 20MW 이상 확인.
- 저장된 CSV를 다시 읽어 같은 검증을 통과한 뒤에만 완료 표시 생성.
- `site_predictions.parquet`: 합산 전 예측 보존.
- `cache/deployment/`: 최종 지역군 모델·스케일러·checkpoint와 예측 보존.
- `comparison.json`: 외부 월별 기존/선택 배정의 NMAE, W 및 관측 시간 수.
- `deployment/group_selection/selected.json`: 최종 배정과 과거 선택 근거.

CSV만 다시 확인하려면:

```bash
python scripts/scoring.py --sites data/preprocessed_v1/sites.parquet --prices data/preprocessed_v1/test_prices.parquet --composition results/tabm_submission_v1/submission/composition.csv --forecast results/tabm_submission_v1/submission/forecast.csv
```

로컬 검증은 제공 형식·제약 검사다. 제출 포털의 공식 Validator와 실제 점수는 별도이며,
테스트 발전 실적이 없으므로 최종 W/S 점수를 만들어 표시하지 않는다.
