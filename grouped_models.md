# 발전소 유형별 모델 비교와 다른 장비에서 실행하기

## 이번 구현

발전소를 하나의 공통 모델로 예측하는 기준선(K=1)과 유형별 모델 2개(K=2), 3개(K=3)를 비교한다. **예측용 유형이며 제출용 G1/G2 배정과 별개**다. 정산 비교에는 기존 참조 G1/G2를 그대로 쓴다. 20MW 제약은 제출용 G1/G2에 적용되고 예측용 유형에는 적용하지 않는다.

`scripts/site_groups.py`는 발전소 유형을 학습하고, `scripts/train_grouped.py`는 유형별 LightGBM 학습·예측·평가·결과 저장·재시작을 담당한다. 공통 모델과 동일한 600회 학습, 특성, 두 예보 라운드 평균을 사용해 그룹화 효과를 비교한다. 종류별로 알고리즘까지 달리하는 실험은 이번 범위가 아니다.

### 그룹화에 쓰는 정보

- 제공 익명 위도·경도와 로그 설비용량.
- 해당 평가월 **이전** 유효 발전량으로 계산한 오전(9~11시), 정오(12~14시), 오후(15~17시)의 평균 용량 대비 발전량. 비율은 기존 기준선과 동일하게 0~1.2 범위로 사용한다.
- 세 시간대에 각각 30개 이상 과거 표본이 있는 발전소로 표준화와 KMeans를 학습한다. 6개 변수를 각각 표준화하고 시드 42·초기화 20회로 고정한다. [KMeans 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).
- 실적이 없거나 부족한 발전소는 위치·용량만으로 가장 가까운 군 중심에 배정한다. 253개소를 모두 한 유형에 배정하며 누락하지 않는다.
- 그룹 수별 실루엣 지표도 저장하지만, 최종 판단은 예측 성능을 기준으로 한다. 크기를 1/N로 강제하지 않으므로 군별 발전소·학습 행 수를 확인해야 한다.

과거 운영 기간과 계절별 자료량 차이가 유형에 영향을 줄 수 있다. 이번 프로필은 첫 비교 방법이고 설비의 고유 특성을 완전히 복원한 결과가 아니다. 평가월별로 그룹화를 새로 학습하므로 `cluster=0`의 의미가 다른 달에도 같다고 가정하면 안 된다. 원좌표나 시설 식별을 위한 외부 자료는 사용하지 않는다.

## 다른 장비 준비 — Linux/GB10

1. Git의 **소스·설정·의존성·테스트 파일**을 새 장비에 가져온다. Git push/pull은 사용자가 수행한다.
2. Python 3.12 이상 가상환경을 만든다(현재 Windows 검증은 Python 3.13). 현재 잠금 파일의 NumPy가 Python 3.12 이상을 요구한다.
3. 원본 자료 또는 이미 만든 전처리 폴더는 별도로 복사한다. `.gitignore` 때문에 Git만 가져오면 데이터는 없다.

아래 명령은 `ai_sim` 폴더에서 실행한다. CPU LightGBM이므로 CUDA 설치 없이 시작할 수 있다. GB10에서도 먼저 CPU 기준으로 비교하며 ARM64 장비에서 설치·속도를 직접 검증한 상태는 아니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
python -m unittest discover -s tests
```

Linux에서 `libgomp.so.1` 관련 로딩 오류가 나면 시스템의 OpenMP 런타임(`libgomp1`)을 설치한다. 패키지 설치 오류가 발생하면 Python 버전과 `uname -m` 결과부터 확인한다.

### 방법 A: 기존 전처리 자료 복사

기존 `ai_sim/data/preprocessed_v1/` 폴더를 새 장비의 같은 위치로 복사한다. 그룹 실험에 필요한 파일은 다음과 같다.

```text
data/preprocessed_v1/
  report.json
  sites.parquet
  reference_composition.parquet
  train_targets.parquet
  train_prices.parquet
  train_round10.parquet
  train_round17.parquet
```

이 실험에는 실측 기상이나 대회 평가기간 test 예보가 필요하지 않다. 이후 최종 제출까지 진행하려면 원본·나머지 전처리 결과도 옮긴다.

### 방법 B: 새 장비에서 원본 전처리

출력 폴더가 아직 없을 때만 실행한다. 원본 경로는 실제 위치로 바꾼다.

```bash
python scripts/preprocess.py --data-dir "/path/to/01_participant_dataset" --output-dir data/preprocessed_v1
```

## 실행 명령

처음에는 한 달로 설치와 경로를 확인한다. 실제 파라미터 600회로 1·2·3그룹 모두 실행한다.

```bash
python scripts/train_grouped.py --fold outer_2024_06 --output-dir results/grouping_check --threads 4
```

전체 비교는 별도 폴더에서 실행한다. 기본값은 12개월 × K=1,2,3 × 예보 2라운드, 총 **144개 모델**이다. 전월 말 17시까지의 전체 학습 이력을 이용하며 평가월 정답은 학습하지 않는다.

```bash
python scripts/train_grouped.py --output-dir results/grouping_full_v1 --threads 8
```

중단 후에는 **같은 인수에 `--resume`만 추가**한다.

```bash
python scripts/train_grouped.py --output-dir results/grouping_full_v1 --threads 8 --resume
```

완료된 월/K 조합은 파일 해시 확인 후 건너뛴다. 중단된 월/K 조합만 처음부터 다시 학습한다. 설정·소스·입력·라이브러리 버전이 바뀌면 재사용하지 않고 새 출력 폴더를 요구한다. 스레드 수 역시 같은 값이어야 한다. `--clusters 1 2`처럼 후보를 줄일 수 있지만 다른 후보 설정은 새 실행 이름을 사용한다. `--data-dir`로 다른 위치의 전처리 자료를 지정할 수 있다.

Windows 임베디드 Python에서는 상위 `sun_elec` 폴더에서 다음처럼 실행한다.

```powershell
.\.tools\python\python.exe -c "import sys; sys.path.insert(0,'ai_sim/scripts'); import train_grouped; train_grouped.main()" --output-dir ai_sim/results/grouping_full_v1 --threads 4
```

## 결과를 이 컴퓨터로 가져오기

출력은 반드시 **`ai_sim/results/실행이름/` 내부**다. 작업 디렉터리가 달라도 기본 출력은 스크립트가 속한 `ai_sim/results/grouping_v1/`로 고정된다.

```text
results/grouping_full_v1/
  manifest.json                   # 실행 설정, 코드·입력 해시, 라이브러리 버전
  summary.json                    # K별 집계 및 완료 여부
  SUMMARY.md                      # 바로 읽을 수 있는 K별·월별 비교표
  outer_2024_06/
    k1/                           # k2/, k3/도 생성
      groups.json                 # 253개소 프로필·배정·군 중심·학습 마감
      metrics.json                # 월별 오차·정산·표본 수·시간·완료 파일 해시
      site_metrics.json           # 발전소별 오차, 관측 시간 수
      site_predictions.parquet    # 발전소·시각별 예측
      models/*.txt                # 군별·라운드별 LightGBM 모델
  ... 나머지 11개월
```

- **Git으로 돌아와 분석:** `SUMMARY.md`와 JSON 보고서는 Git 제외 대상이 아니므로 사용자가 선택해 커밋·push할 수 있다. 이 컴퓨터에서 pull하면 비교·배정·발전소별 오차를 검토할 수 있다. 자동 Git 작업은 없다.
- **시간별 상세 분석·모델 재사용:** `.parquet`과 `models/`는 Git에서 제외된다. 이 파일까지 필요하면 실행 폴더 전체를 외장 디스크·파일 전송 등으로 이 컴퓨터의 `ai_sim/results/`에 복사한다. **Git만으로 대용량 파일이 돌아오지는 않는다.**
- 다른 장비에서 재시작하려면 입력 자료뿐 아니라 완료된 모델·예측 파일도 함께 있어야 한다. 요약만 복사한 폴더는 분석용이며 `--resume` 검증을 통과하지 않는다.

## 결과 판단

`summary.json`의 `all_12_months=true`를 먼저 확인한다. 월별 실적 보유 발전소 수가 달라도 각 발전소의 용량정규화 MAE를 평균하고, 다시 12개월을 동일 가중으로 평균한다. 전체 253개소 정산은 완전 실적 3,802시간만 계산하며, 여름·가을의 불완전 정산은 N/A다. 출력 예측도 내부 검증의 실적 보유 행 기준이다.

`exploratory_best_k`는 12개월이 모두 끝났을 때 월 동일 가중 오차가 가장 작은 K를 알려준다. **세 후보를 탐색적으로 비교한 결과이며, 같은 평가 결과로 고른 K의 성능을 독립 검증 성능으로 주장하지 않는다.** 최종 채택 전에는 기존 v2의 과거 내부 선택 구간에서 K를 고르고 외부 월에서 확인하는 단계가 필요하다. 현재 432개 내부 선택 창을 전부 학습하는 코드는 이번 실행에 포함하지 않는다.

실루엣이 높아도 예측 성능이 좋아진다는 보장은 없다. K=1보다 월별 오차·가용 정산 W가 개선되는지, 군별 표본이 충분한지, 특정 계절이나 신규 발전소에서 나빠지는지 함께 확인한다. K=2/3이 더 나쁘면 공통 모델을 유지할 수 있다.

## 이 컴퓨터에서 확인한 결과

- 테스트 **23개 통과**: 모든 253개소 배정, 그룹화 재현, 이력 부족 발전소 처리, 구분 불가능한 프로필 거부, 미래 실적 변경이 과거 프로필에 영향을 주지 않는지 포함한다.
- 실제 2024년 6월 한 달을 대상으로 K=1/2/3의 **12개 모델**을 학습했다. 각 후보 모두 동일한 153,311개 발전소·시간 행을 예측했다. [실행 요약](results/grouping_smoke_v1/SUMMARY.md).
- K=1 예측은 기존 4단계 같은 월의 예측 153,311개와 최대 차이 0이다. 그룹별 저장 모델을 다시 불러온 예측도 확인했다.
- 같은 설정의 `--resume`이 세 후보를 모두 재학습 없이 건너뛰었다. 스레드 설정을 바꾼 재시작은 거부했다. 모델·Parquet은 Git 제외, JSON·Markdown 요약은 Git으로 옮길 수 있음을 확인했다.

| 후보 | 발전소 수 | 용량정규화 MAE |
|---|---|---:|
| K=1 | 253 | 4.564% |
| K=2 | 185 / 68 | 4.692% |
| K=3 | 59 / 151 / 43 | 4.767% |

이 달에는 전체 253개소 정산 가능한 시간이 없어 W는 N/A다. K=3의 마지막 군은 발전소 43개지만 과거 유효 학습 행은 라운드당 15,487개뿐이다. **발전소 개수와 실제 학습 자료량은 다르다.** 각 군의 `training_rows`와 이력 부족 여부를 함께 확인해야 한다. 한 달만으로 그룹 수를 확정하지 않으며, 전체 12개월 비교와 GB10 실행은 새 장비에서 진행할 작업으로 남겨 두었다.
