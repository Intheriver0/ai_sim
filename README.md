# ai_sim

2026 OIBC 태양광 집합자원 구성·하루 전 발전량 예측 프로젝트의 문서와 로컬 점검 코드입니다. [master.md](master.md)에 목표와 현재 상태, [data_structure.md](data_structure.md)에 제공 파일의 구조와 README 대조 결과를 정리했습니다.

개발은 이 저장소에서, 실제 대규모 학습은 Git으로 연결된 **Linux 기반 GB10 Spark 장비**에서 수행합니다. GPU 모델 비교·자동 파라미터 탐색·시간순 평가·배분 최적화의 구현 순서는 [automl_plan.md](automl_plan.md)에 기록했습니다. 학습 코드는 Linux에서 바로 실행할 수 있는 Python 명령줄 프로그램으로 작성합니다.

대회 원본 데이터, 생성한 집계 JSON, 다운로드한 PDF·지형 자료와 학습 결과는 Git에 포함하지 않습니다. 데이터 사용 조건과 원본 보존 원칙은 [competition_guidelines.md](competition_guidelines.md)를 따릅니다. `data_structure.md`의 `data/dataset_audit.json` 링크 등은 로컬 점검 결과를 가리키므로 GitHub에서는 열리지 않습니다.

데이터를 사용할 수 있는 로컬 환경에서는 원본 폴더를 저장소 루트의 `OIBC_2026_DATA/handover 2/01_participant_dataset/`에 두고 다음 명령을 실행합니다. 명령은 Linux 셸과 Windows PowerShell에서 동일합니다.

```sh
python -m pip install -r requirements-inspection.txt
python scripts/inspect_dataset.py
```

점검 스크립트는 원본을 읽고 `data/dataset_audit.json`에 파일별 스키마·행 수·키·집계 통계를 기록합니다. 예측 모델 학습과 대회 제출은 아직 구현되지 않았습니다. `refference/`의 문헌 PDF 원본도 로컬에만 보관합니다.
