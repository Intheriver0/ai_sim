# ai_sim

2026 OIBC 태양광 집합자원 구성·하루 전 발전량 예측 프로젝트의 문서와 로컬 점검 코드입니다. [master.md](master.md)에 목표와 현재 상태, [data_structure.md](data_structure.md)에 제공 파일의 구조와 README 대조 결과를 정리했습니다.

대회 원본 데이터, 생성한 집계 JSON, 다운로드한 PDF·지형 자료는 Git에 포함하지 않습니다. 데이터 사용 조건과 원본 보존 원칙은 [competition_guidelines.md](competition_guidelines.md)를 따릅니다. `data_structure.md`의 `data/dataset_audit.json` 링크 등은 로컬 점검 결과를 가리키므로 GitHub에서는 열리지 않습니다.

데이터를 사용할 수 있는 로컬 환경에서는 원본 폴더를 저장소 루트의 `OIBC_2026_DATA/handover 2/01_participant_dataset/`에 두고 다음을 실행합니다.

```powershell
python -m pip install -r requirements-inspection.txt
python scripts/inspect_dataset.py
```

점검 스크립트는 원본을 읽고 `data/dataset_audit.json`에 파일별 스키마·행 수·키·집계 통계를 기록합니다. 예측 모델 학습과 대회 제출은 아직 구현되지 않았습니다. `refference/`의 문헌 PDF 원본도 로컬에만 보관합니다.
