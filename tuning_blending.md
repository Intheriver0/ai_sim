# 5·6단계: 자동 튜닝과 예측 혼합

`scripts/train_tuned.py`가 K=1 공통 LightGBM의 파라미터 탐색, 기준선 유지 여부 선택, 혼합 비율 선택, 외부 월 평가를 순서대로 실행한다. 7단계 G1/G2 배분 최적화와 최종 대회 제출은 포함하지 않는다. 참조 배분을 고정한다.

## 선택 규칙

- 기존 특성, L2 학습 손실, 10시·17시 모델 평균을 유지한다. K·변수군·새 모델 계열은 탐색하지 않는다.
- 외부 월마다 이전 12개월의 1~10일 블록에서 파라미터를 선택하고, 11~20일 블록에서 혼합 비율을 선택한다. 각 블록의 모델은 그 블록 이전 자료만 사용한다. 21일 이후 배분 선택 블록은 7단계용으로 남긴다.
- 튜닝의 초기 학습은 블록 시작 D−8일 17시까지, 조기 종료는 D−7~D−2일, 선택된 반복 수로 재학습하는 범위는 D−1일 17시까지다. 외부 월도 동일한 조기 종료·재학습 규칙을 따른다. 기존 기준선은 600회를 유지하며 조기 종료하지 않는다.
- Optuna는 학습률 0.015~0.12, 잎 수 15~127, 최소 잎 표본 50~1000, 행·변수 샘플링 0.6~1.0, L1/L2 정규화 1e−8~10을 탐색한다. 기본 최대 반복 수는 1,200, 조기 종료 인내 횟수는 30이다. 이 범위는 이번 탐색 설정이며 문헌 최적값을 뜻하지 않는다.
- **선택 목적함수는 월 동일 가중 발전소별 용량정규화 MAE로 고정한다.** 완전 실적의 유무에 따라 목적함수를 바꾸지 않는다. 이는 공식 정산 W의 대리 지표다. W는 완전한 253개소 실적이 있는 시간에만 별도로 계산하며 그 외에는 N/A다.
- 기준선의 내부 점수보다 나아진 trial이 없으면 기준선을 선택한다. `selected`는 이 선택 결과이며 항상 튜닝 모델을 뜻하지 않는다.
- 혼합은 `alpha × fixed + (1−alpha) × selected`다. alpha는 0~1의 0.05 간격이며 모든 발전소·시간에 공통이다. 양 끝점도 포함하고 정확한 동점은 기준선 비율이 높은 쪽을 선택한다.
- 선택 결과를 저장한 뒤 외부 월을 평가한다. 외부 점수를 보고 설정을 변경하면 별도 개발 실험이며 독립적인 최종 성능으로 해석하지 않는다.

## Linux/GB10 실행

`ai_sim` 폴더와 기존 전처리 자료 `data/preprocessed_v1/`가 필요하다. Python 3.12 이상을 사용한다. 이번 구현은 CPU LightGBM이며 GPU 설정이 필요하지 않다.

```bash
source .venv/bin/activate
python -m pip install -r requirements-training.txt
python -m unittest discover -s tests -v
python -u scripts/train_tuned.py --output-dir results/tuning_smoke_local --threads 8 --smoke
```

`--smoke`는 외부 한 달, 내부 모델·혼합 각각 한 달, trial 하나, 기준선·튜닝 모두 최대 12회로 줄인 **실행 검사용**이다. 이 점수를 기존 600회 모델과 비교하지 않는다. 다른 월은 `--fold outer_2025_01`처럼 지정한다. 각 검사는 새 출력 폴더를 사용한다.

본 실행 예시는 외부 12개월, 월당 최대 20개 완료 trial, 월당 누적 탐색 예산 4시간이다. 실험 전에 이 설정을 확정하고 동일 명령으로 이어 실행한다.

```bash
python -u scripts/train_tuned.py --output-dir results/tuning_full_v1 --threads 8 --trials 20 --timeout 14400
```

터미널 종료 후에도 유지하려면 위 전경 실행과 **둘 중 하나만** 실행한다.

```bash
nohup python -u scripts/train_tuned.py --output-dir results/tuning_full_v1 --threads 8 --trials 20 --timeout 14400 > tuning_full_v1.log 2>&1 < /dev/null &
tail -f tuning_full_v1.log
```

같은 출력에 동시에 두 프로세스를 실행하지 않는다. 재개 전 기존 학습이 종료됐는지 확인하고 동일 명령에 `--resume`을 추가한다.

```bash
python -u scripts/train_tuned.py --output-dir results/tuning_full_v1 --threads 8 --trials 20 --timeout 14400 --resume
```

`--timeout`은 **trial 사이에서 검사하는 소프트 한도**다. 진행 중인 trial은 끝까지 완료하므로 초과할 수 있다. 기준선 학습·혼합·외부 평가는 이 예산에 포함하지 않으므로 전체 실행 시간은 4시간×12개월보다 길 수 있다. trial 하나도 내부 12개월의 두 라운드를 초기 학습·재학습하므로 최대 48회 학습이다. 축소 실행의 속도를 본 실험 총시간으로 단순 환산하지 않는다.

중단 trial은 실패 상태로 기록하고 같은 파라미터를 다시 대기열에 넣는다. 완료된 블록 예측은 해시 검증 후 재사용한다. 완료된 trial 수와 사용 시간을 누적하며 완료된 선택 결과는 고정한다. 소스·입력·라이브러리·예산·스레드 수 변경은 새 출력 폴더를 요구한다. 단일 프로세스 SQLite를 사용한다.

## 결과

```text
results/tuning_full_v1/
  manifest.json             # 실행 전 고정한 규칙·입력/소스 해시·버전
  git.json                  # 실행 시작 시 Git HEAD
  summary.json / SUMMARY.md # 기준선·선택 모델·혼합의 월별/집계 비교
  outer_YYYY_MM/
    study.db                # 로컬 Optuna 기록
    trials.json             # 가벼운 trial 결과
    selection.json          # 파라미터 선택·기준선 유지 여부
    blend.json              # alpha별 내부 곡선·선택된 alpha
    result.json             # 외부 평가 결과·모델 캐시 키
    blend_predictions.parquet
  cache/<hash>/
    metrics.json
    predictions.parquet
    models/                 # 외부 월 모델만 저장; 라운드별 txt
```

`all_12_months=true`, `smoke=false`를 확인한다. 표의 `fixed`, `selected`, `blend`를 비교하고 최악 월과 W를 함께 확인한다. W 집계는 가용 시간만의 값으로 여름·가을 전체 정산을 입증하지 않는다.

내부 예측 캐시는 중복 학습을 줄이지만 디스크를 사용한다. 내부 trial 모델 가중치는 저장하지 않으며 선택된 설정으로 재학습할 수 있다. 외부 모델은 저장·재로딩 예측 일치를 검사한다. 시간순 외부 혼합 예측은 7단계 개발의 입력으로 활용할 수 있으나, 각 외부 월의 배분 선택에는 그 월 이전 내부 배분 블록의 잔차를 별도로 생성해야 한다.

데이터·Parquet·캐시·SQLite·축소 실행 결과는 Git에서 제외한다. 본 실행의 JSON·Markdown 보고서는 사용자가 선택해 커밋할 수 있다. 재시작에는 실행 폴더 전체와 같은 전처리 입력이 필요하며 JSON만으로는 재개할 수 없다.

## 로컬 검증

Windows에서 실제 전처리 자료로 축소 실행을 검증한다. 이는 실행 경로 검증이며 12개월 본 튜닝 성능이나 GB10 실측 결과를 뜻하지 않는다. 테스트는 예측 키 정렬·오류 거부, 혼합 끝점·동점, 과거 학습/조기 종료 경계, 기준선 유지, trial 예산·중단 재개, 캐시 변조 거부를 포함한다.

2026-10-28 로컬 검증: 전체 테스트 33개 통과. 실제 자료의 2024년 6월(전체 정산 N/A)과 2025년 1월(완전 실적 744시간)에서 축소 실행을 완료했다. 최종 수정본은 `results/tuning_smoke_final/`의 1월 실행이며 재개 시 학습 없이 캐시를 재사용했다. 축소 trial은 기준선보다 나빠 기준선 유지와 alpha=1 경로를 확인했다. 개선 trial 선택·혼합의 중간 비율은 단위 테스트로 검증했으며 실제 본 실험에서 개선을 입증한 것은 아니다. Windows SQLite 연결 종료 문제를 수정하고 전체 검증을 통과했다.
