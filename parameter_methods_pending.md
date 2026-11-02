# 파라미터 방법론 미결 목록

업데이트: 2026-10-31. 기존 Python 3.13 / NumPy 2 환경을 낮추거나 새 venv를 만들지 않는다.
환경 문제와 아직 수행하지 않은 연구 작업을 구분한다. 아래 방법을 구현 완료 또는 성능 검증 완료로 표시하지 않는다.

| 항목 | 상태와 확인한 이유 | 재개 조건 / 남은 작업 |
|---|---|---|
| DEHB | **환경 분리 후순위.** 공식 0.1.2가 NumPy<2를 요구. 기존 직접 호출 adapter는 있지만 검증되지 않았으며 production CLI에서 명시적으로 차단 | 추후 별도 환경 사용을 재개할 때 proposal/evaluation 연결과 실제 adapter 테스트 |
| 원 BOHB / HpBandSter 0.7.4 | **현재 환경 설치 불가로 보류.** 2026-10-31 dry-run은 다운그레이드를 요구하지 않았으나 실제 wheel 빌드에서 netifaces 0.11.0이 Microsoft Visual C++ 14+ 부재로 실패. 별도 venv가 반드시 필요하다는 의미는 아니며, 현재 머신 빌드 도구 문제 | 호환 wheel 또는 빌드 가능한 환경 확보 후 공식 KDE 제안과 Hyperband 연결을 검증. 임의 TPE+Hyperband를 BOHB로 이름만 바꾸지 않음 |
| p-diff | **연구 구현 미결, 의존성 충돌로 판정하지 않음.** 프로젝트용 체크포인트 bank, 동일 좌표계의 가중치 부분집합, bank 학습/선택 분리 준비가 안 됨 | 동일 학습 origin·전처리·구조의 bank 수집 → autoencoder → latent diffusion → 과거 구간에서 생성 후보 선택 → 외부 평가. 기존 CUDA1028 모델을 다른 시점에 무조건 재사용하지 않음 |
| GHN-2 | **연구 구현 미결.** 태양광 MLP 구조 그래프와 구조별 학습 과제 데이터 구축 전 | PV용 구조·가중치 학습 자료 및 별도 구조 holdout 구축. 이미지 pretrained GHN을 PV 재현으로 취급하지 않음 |
| NNiT | **관찰/연구 미결.** 조사된 근거는 preprint이며 사용자 조건인 검증된 방법 우선에 따라 후순위 | 독립 검증 및 PV 구조 토큰/가중치 생성 실험 설계 후 구현 여부 재검토 |
| 전체 학습 궤적 hypergradient | **미구현.** 추가된 것은 한 단계 truncated bilevel 변형 | 전체 궤적 미분/메모리 비용/근사 오차 비교가 별도로 필요 |
| 동일 GPU-hour 제한 비교 | **미구현.** trials/epochs는 방법 간 동일 총 계산량을 뜻하지 않음 | 캐시 재사용·meta-training·최종 refit 비용을 분리한 자원 예산 관리자 필요 |
| 본 데이터 CUDA 성능 검증 | **미실행.** 로컬은 CPU PyTorch | CUDA 학습 컴퓨터에서 preflight 후 12개 외부 월 실행. 실제 NMAE/W 개선량 확인 |

현재 실행 가능한 추가 항목은 GP Bayesian optimization, 1-step hypergradient, coordinatewise LSTM learned optimizer다.
자세한 알고리즘 범위·설정·실행 명령은 [구현 문서](parameter_methods_implementation.md)를 따른다.
