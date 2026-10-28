# ai_sim

**현재 모든 학습은 CUDA GPU 필수입니다.** [GB10 CUDA 설치·검사·실행](gpu_training.md)을 먼저 따릅니다. CUDA 초기 학습 검사가 실패하면 CPU로 대체하지 않고 종료합니다. 이전 CPU 결과에서 재개하지 않고 새 GPU 실행 폴더를 사용합니다.

**5·6단계 자동 튜닝·혼합:** [실행·재시작·결과 해석](tuning_blending.md). `scripts/train_tuned.py`가 K=1 기준선과 Optuna 모델을 비교하고 과거 별도 구간에서 혼합 비율을 선택합니다. 먼저 `--smoke`로 실행 경로를 확인한 뒤 GB10에서 본 실험을 실행합니다.

**다른 장비에서 발전소 그룹별 모델 실행:** [설치·데이터 이동·실행·결과 회수 안내](grouped_models.md). `scripts/train_grouped.py`가 K=1/2/3을 비교하고 결과를 `ai_sim/results/`에 저장합니다. Git push는 자동 실행하지 않습니다.

2026 OIBC 태양광 집합자원 구성·하루 전 발전량 예측 프로젝트의 문서와 로컬 점검 코드입니다. [master.md](master.md)에 목표와 현재 상태, [data_structure.md](data_structure.md)에 제공 파일의 구조와 README 대조 결과를 정리했습니다.

개발은 이 저장소에서, 실제 대규모 학습은 Git으로 연결된 **Linux 기반 GB10 Spark 장비**에서 수행합니다. GPU 모델 비교·자동 파라미터 탐색·시간순 평가·배분 최적화의 구현 순서는 [automl_plan.md](automl_plan.md)에 기록했습니다. seasonal_splits_v2는 2024-06~2025-05 외부 월 평가와 각 월 이전 12개월 내부 선택을 정의합니다. 4단계 고정 LightGBM과 테스트 20개를 구현·검증했고 로컬 CPU에서 월별 기준선 평가를 실행합니다. 실행 상태와 결과는 [4단계 안내](fixed_baseline.md)를 확인합니다. 학습 코드는 Linux에서도 실행할 수 있는 Python 명령줄 프로그램입니다.

대회 원본 데이터, 생성한 집계 JSON, 다운로드한 PDF·지형 자료와 학습 결과는 Git에 포함하지 않습니다. 데이터 사용 조건과 원본 보존 원칙은 [competition_guidelines.md](competition_guidelines.md)를 따릅니다. `data_structure.md`의 `data/dataset_audit.json` 링크 등은 로컬 점검 결과를 가리키므로 GitHub에서는 열리지 않습니다.

데이터를 사용할 수 있는 로컬 환경에서는 원본 폴더를 저장소 루트의 `OIBC_2026_DATA/handover 2/01_participant_dataset/`에 두고 다음 명령을 실행합니다. 명령은 Linux 셸과 Windows PowerShell에서 동일합니다.

```sh
python -m pip install -r requirements-inspection.txt
python scripts/inspect_dataset.py
```

점검 스크립트는 원본을 읽고 `data/dataset_audit.json`에 파일별 스키마·행 수·키·집계 통계를 기록합니다. 정산·제약 검사, 전처리, 시간순 검증 구간과 고정 모델(구현 순서 1~4단계)을 구현했습니다. 실행 방법은 [전처리 안내](preprocessing.md), [12개월 균등 검증 안내](seasonal_validation.md), [고정 모델 안내](fixed_baseline.md)에 있습니다. 5·6단계 튜닝·혼합 실행기도 구현했으며 대규모 성능 실험, 배분 최적화와 최종 대회 제출은 남아 있습니다. `refference/`의 문헌 PDF 원본도 로컬에만 보관합니다.

제출 파일은 운영팀이 공개한 [형식 Validator](https://app.notion.com/p/Validator-3fe10e0a193380ab925fc25abbf6a1ca)로 최종 점검합니다. [DataEn 공지](https://dataen.ai/challenge/2026/board/4984651b-a17a-4d2f-8350-fbf36ffc037b)에 따르면 이 도구는 점수를 계산하지 않으며 `scoring.py`는 배포되지 않습니다.
