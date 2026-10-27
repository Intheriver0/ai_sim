# 1~2단계: 정산 검사와 데이터 전처리

구현 범위는 README 정산식 기반 로컬 계산, 구성·예측·실적 검사, 발전소 메타와 A/B 예보 결합이다. 모델 학습, 시간순 폴드 선택, 튜닝, 배분 최적화는 후속 단계다. 운영팀의 점수 계산 프로그램을 재현했다고 주장하지 않으며 공식 형식 Validator 검사는 제출 단계에서 별도로 수행한다.

## 실행

일반 Python 환경에서 저장소 루트 `ai_sim`을 작업 디렉터리로 사용한다. Linux에서도 같은 명령을 쓴다. 데이터 경로는 실제 보관 위치로 지정한다.

```sh
python -m pip install -r requirements-training.txt
python -m unittest discover -s tests -v
python scripts/preprocess.py --data-dir "../OIBC_2026_DATA/handover 2/01_participant_dataset" --output-dir data/preprocessed_v1
```

현재 Windows 작업공간의 `.tools/python`은 임베디드 Python으로 스크립트 폴더를 자동으로 모듈 검색 경로에 넣지 않는다. 상위 작업공간 `sun_elec`에서 실행할 때는 다음 명령을 쓴다.

```powershell
.\.tools\python\python.exe -m unittest discover -s ai_sim/tests -v
.\.tools\python\python.exe -c "import sys; sys.path.insert(0, 'ai_sim/scripts'); import preprocess; preprocess.main()" --data-dir "OIBC_2026_DATA/handover 2/01_participant_dataset" --output-dir ai_sim/data/preprocessed_v1
```

출력 폴더는 새 경로여야 하며 원본 데이터 폴더 내부에는 만들 수 없다. 재실행은 `preprocessed_v2`처럼 다른 경로로 지정한다. 실패 시 부분 파일이 남을 수 있으며, **`report.json`의 `status=complete`가 있는 실행만 사용**한다. DuckDB 메모리 제한은 1GB, 스레드는 2개다. 중간 계산은 디스크로 넘길 수 있으며 이 제한은 전체 프로세스 메모리 상한을 뜻하지 않는다.

## 처리 정책

| 대상 | 처리 |
|---|---|
| 시간 | 타임존이 있는 시각만 허용. KST 기준 정각과 키 유일성 검사 |
| 대상일·발행시각 | 우측닫힘 규약으로 `target_day=date(ts-1시간)`. 전날 10시 또는 17시를 `issued_at`으로 계산. 실제 발행 메타데이터를 새로 확보한 것은 아님 |
| 학습 행 | 원본 발전량에 실제 존재하는 `(site_id, ts)`만 사용. 없는 발전량은 행 생성·0 대체하지 않음 |
| 음수 발전량 | `energy_raw_kwh`에 원값 보존. `target_kwh=NULL`, `negative_energy=true`, `training_eligible=false`. 학습에서 제외해야 하며 로컬 정산에는 원본 실적을 그대로 사용 |
| B 비정각 예보 | 반올림하지 않고 별도 `*_off_hour.parquet`에 보관. 정각 결합에서는 제외 |
| 예보 누락·결측 | left join으로 발전소·시각 행 보존. `wa_missing`·`wb_missing`은 예보 행 부재 표시. 개별 변수 NULL은 그대로 유지하고 NaN도 NULL로 통일 |
| 풍속 | A는 m/s 유지, B 풍속·돌풍은 3.6으로 나눔. 열 이름을 `*_ms`로 변경 |
| 강설 | A는 mm 유지, B는 cm에서 mm로 환산. 열 이름을 `*_mm`로 변경. 단위 통일만 하며 두 제공사의 측정 정의가 같다고 가정하지 않음 |
| 범위 밖 예보 | 습도 105 등 원값 유지. 문서의 관측 범위를 임의 clipping 기준으로 사용하지 않음. 무한값은 오류로 중단 |
| 실측 기상 | `clean_observed.py`로 별도 정제. README 이상값과 비유한값은 셀 단위 NULL 및 `*_invalid` 표시. 예보 특성에는 결합하지 않음. 중복 키는 `duplicate_key`로 표시하고 교사 학습에서 제외 |
| SMP·정답 | 예보 특성 파일과 별도 저장. 평가용 SMP와 발전량이 자동으로 모델 입력에 들어가지 않게 함 |

평균·중앙값 보간 등 학습되는 전처리는 수행하지 않는다. 향후 필요하면 각 학습 폴드에서만 적합한다. 라운드별 파일에 동일한 발전소·시각이 반복되므로 모델 평가 때 라운드별 정산금을 더하면 안 된다. 두 라운드의 예측을 결합한 뒤 시간당 단일 예측으로 정산한다.

## 생성 파일

| 파일 | 내용 |
|---|---|
| `sites.parquet` | 원본 발전소 메타 |
| `reference_composition.parquet` | 참조 A/B를 G1/G2로 변환하고 20MW 제약 검증 |
| `train_targets.parquet` | `site_id, ts, energy_raw_kwh, target_kwh, negative_energy, training_eligible` |
| `train_round10.parquet`, `train_round17.parquet` | 학습 시각의 발전소 메타·정규화 예보·결측 표시. 정답은 키로 별도 결합 |
| `test_round10.parquet`, `test_round17.parquet` | 전체 발전소 × 평가 시각의 동일 스키마 특성 |
| `train_prices.parquet`, `test_prices.parquet` | `ts, smp, k`. KST 11~15시 k=3, 나머지 1 |
| `train_wb_off_hour.parquet` | 제외된 비정각 B 원본 예보 |
| `report.json` | 원본 경로, DuckDB 버전, 정책, 행 수, 결측 수, 참조 용량, 완료 여부 |

특성 파일 전체를 무조건 모델에 넣는 방식은 사용하지 않는다. 후속 모델 단계에서 ID·시간·메타·날씨의 입력 열을 명시적으로 선택한다. 출력은 Git 제외 경로인 `data/`에 보관한다.

## 정산·제약 검사

CSV 또는 Parquet 입력을 받는다. `--prices`의 시간축을 검사·계산 대상으로 사용하므로 최종 제출 검사에는 반드시 원본 평가 `smp.csv`를 지정한다.

```sh
python scripts/scoring.py --sites data/preprocessed_v1/sites.parquet --composition data/preprocessed_v1/reference_composition.parquet --forecast path/to/forecast.csv --prices path/to/validation_prices.parquet --actual "../OIBC_2026_DATA/handover 2/01_participant_dataset/train/generation.parquet"
```

`forecast` 스키마는 `group,ts,forecast_kwh`, 실적은 `site_id,ts,energy`다. 검증용 가격 파일은 평가할 시간만 포함해야 한다. 원본 학습 가격 전체에는 실적이 불완전한 시각이 있으므로 전체를 지정하면 계산을 거부한다. 실제 검증 구간 선택은 3단계에서 한다.

- 구성: PV001~PV253 전부 정확히 한 번, G1/G2만 허용, 양쪽 용량 20,000kW 이상.
- 예측: 각 그룹에 가격 시간축이 정확히 한 번씩 존재하고 타임존·정각·유한값·비음수 조건 만족.
- 실적: 대상 시각마다 모든 발전소의 실적 필수. 대상 밖 과거 시각은 계산에서 제외. 실적 누락을 부분 집합 합계로 평가하지 않음.
- 정산: 발전소 실적을 집합별로 **먼저 합산**한 뒤 `W=Σ0.1×SMP×k×|F−A|`, `S=ΣSMP×A−W` 계산. 결과는 그룹별·전체 원 단위.
- `--actual`을 생략하면 구조만 검사하며 점수를 계산하지 않음.

테스트는 손계산 점수, 집합 내 오차 상쇄, 0원 SMP, KST 가중치 경계, 누락·중복·NaN·무한대·음수 예측·용량 위반, 작은 원본 파일을 이용한 전처리 전체 실행을 확인한다.

## 실제 실행 결과 — 2026-10-27

테스트 6개를 통과하고 제공 원본 전체 전처리를 완료했다. [실행 보고서](data/preprocessed_v1/report.json)는 로컬에 저장되며 Git에 포함하지 않는다.

| 항목 | 확인 결과 |
|---|---|
| 학습 특성 | 라운드별 4,392,566행, 키 중복 없음 |
| 평가 특성 | 라운드별 2,216,280행 = 253개소 × 8,760시간, 키 중복 없음 |
| 음수 발전량 | 40행 원값 보존, 학습 정답 결측 처리 |
| 비정각 B 예보 | 92행 별도 보관 |
| 평가 A 예보 누락 영향 | 라운드 10에서 발전소·시각 872행, 라운드 17에서 188행. 해당 특성 행은 삭제하지 않음 |
| 평가 B 예보 누락 영향 | 0행 |
| 전체 발전소 실적이 있는 시간 | 3,802시간. 검증 구간 선택은 후속 단계 |
| 참조 배정 용량 | G1 20,465.655kW, G2 22,189.715kW |

이 결과는 전처리·검사 결과이며 모델 예측 점수는 아니다.

## 실측 이상값 필수 전처리 추가 — 2026-10-27

배포 README의 실측 이상값 안내를 프로젝트의 필수 학습 전처리로 적용한다. 대회가 특정 제거 방법을 강제한 것은 아니다. 임의의 `9999` 전역 치환 대신 변수·자료 종류·원단위별로 판정한다. 예보의 관측 최솟값·최댓값을 제거 기준으로 사용하지 않는다.

| 실측 조건(원단위) | NULL 처리된 셀 수 |
|---|---:|
| A 풍속 > 50 m/s (9999 포함) | 14 |
| B 운량 > 100% | 433 |
| B UV > 15 | 394 |
| B 시정 > 50 km | 896 |
| B 풍속 > 200 km/h | 230 |
| B 기압 < 900 hPa | 428 |

원본과 행 수는 보존하고 이상 셀 2,395개를 NULL로 바꿨다(고유 행 수가 아님). 정제 실측의 단위는 원본과 같다. A 중복 6키의 12행은 모두 표시했다. 향후 보정 학습은 `duplicate_key=false`이면서 해당 교사 변수 값이 NULL이 아닌 과거 학습 구간만 사용해야 한다. 실측은 평가 추론 입력으로 사용할 수 없다.

실제 추가 실행 결과는 `data/observed_clean_v1/observed_wa.parquet`, `observed_wb.parquet`, [report.json](data/observed_clean_v1/report.json)에 있다. 기존 `preprocessed_v1`은 변경하지 않았다. 이후 `preprocess.py` 실행은 같은 정제를 필수로 호출해 새 출력의 `observed_clean/`에 저장한다. 기존 예보 전처리 결과에 실측 정제만 추가할 때는 다음을 실행한다.

```sh
python scripts/clean_observed.py --data-dir "../OIBC_2026_DATA/handover 2/01_participant_dataset" --output-dir data/observed_clean_v1
```
