# 모든 구현 방법 통합 실행

CUDA 학습 컴퓨터의 `ai_sim`에서 실행한다. 기존 학습 환경, 전처리 자료,
`data/ridge_regions_v1`, `data/terrain_features_v2`, 완료된 CUDA1028 결과가 필요하다.
필요 패키지는 기존 `requirements-training.txt`와 `requirements-parameter-methods.txt`를 따른다.

```bash
python scripts/run_all.py
```

기본 결과 폴더는 `results/all_methods_v1`이다. GPU 작업은 한 번에 하나만 실행한다.

1. `run_literature.py`: LightGBM L2/L1/예보 수정량, CatBoost MAE, MLP10, TabM을 가능한 지역 군별로 비교.
2. `run_parameter_methods.py`: MLP, TabM, hypernetwork head, MAML, SGD MLP, 한 단계 hypergradient, LSTM learned optimizer 각각 실행.
   각 모델에 fixed/random/TPE/GP/ASHA/Hyperband/TPE+Hyperband/SMAC/SMAC multi-fidelity/PBT를 적용한다.
   충돌하는 hypergradient+PBT 조합은 제외한다. 기존 실행기의 대조군 평가도 유지한다.
3. `run_pipeline.py --stage compare`: 기존 기준선·쌍 입력·그룹 재편·특성 군집 비교.

각 실행기의 기본 예산과 12개 외부 월을 그대로 사용하므로 매우 큰 실험이다.
과거 별도 8개 후보 스크리닝이나 구버전 실행기를 중복 실행하지 않으며,
미구현 p-diff/GHN/NNiT 및 실행 차단된 DEHB/BOHB는 제외 사유를 결과에 기록한다.
제출 파일 생성이나 새 방법의 그룹 재편 사이클 내부 통합은 수행하지 않는다.

## 사전 점검·재개

```bash
python scripts/run_all.py --preflight-only
python scripts/run_all.py --resume
```

명령 목록만 확인하려면 `--dry-run`을 사용한다(파일 생성·학습 없음).
경로가 다르면 `--data-dir`, `--regions-dir`, `--terrain-dir`, `--config`,
`--error-run`, `--results-root`, `--output-dir`로 지정한다. 재개할 때 같은 옵션을 사용한다.

한 실행이 실패해도 다음 실행을 진행하고 전체 종료 코드는 1로 반환한다.
실패한 실행 안의 나머지 모델/방법은 실행되지 않을 수 있으므로 해당 문제를 해결하고 재개한다.
Ctrl+C는 현재 자식 프로세스를 종료하고 중단 상태와 부분 결과를 기록한다.
재개는 완료 실행도 기존 실행기에 다시 진입하여 입력·버전·저장 결과 검사를 수행한다.
검사를 통과한 캐시를 재사용하며, 변경된 코드·설정·입력으로 강제 재개하지 않는다.
같은 출력 폴더에서 두 실행을 동시에 시작하지 않는다.

## 통합 산출물

- `run.log`: 모든 실행의 stdout/stderr, UTC 시각, 실행 이름, 명령, 종료 상태. 재개 시 이어 쓴다.
- `results.json`: 실행별 상태·종료 코드·시간과 원래 `comparison.json`, `fold_metrics.json`, `preflight.json`의 내용.
- `SUMMARY.md`: 위 상태와 평가 결과를 한 문서에 모은 보고서.
- 실행별 하위 폴더: 원래 체크포인트·캐시·예측·평가 파일 보존.

로그는 실행 중 계속 기록되고 결과 보고서는 각 실행 전후에 갱신한다.
실패·중단 실행의 결과는 부분 결과나 이전 시도의 결과일 수 있으며 상태를 함께 확인한다.
사전 점검 성공은 `preflight_complete`로 표시하고 본실험 완료와 구분한다.
평가 절차가 다른 점수를 합쳐 단일 우승 모델을 고르지 않는다.
실행 시간은 동일 GPU-hour 비교 결과가 아니다.
