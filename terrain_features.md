# 지형 가림 추정 자료 v1 / v2

2026-10-29 사용자 요청으로 만드는 별도 실험용 자료다. 기존 학습·자동 튜닝·후보 큐에는 연결하지 않는다.

## v2: 원본 지형과 누락 타일 보완

v1의 대다수 결측은 남한 행정경계 마스킹 때문이었다. v2는 마스킹 전 원본 26개와 추가 다운로드한 `N34_E129` 타일을 연결한 27개 타일을 사용한다. 새 파일 크기는 4,158,318 bytes이며 SHA-256은 `5eed44d79aa4f08e4fced5224e951d08ff3a30f6c29d9055cd8e427e1a993837`이다. 원본 고도를 사용하며 바다나 빈 영역을 임의로 0으로 채우지 않는다.

입력은 `../data/terrain/original_expanded.vrt`, 출처·해시는 `original_expanded_manifest.json`에 기록된다. v2 출력은 `data/terrain_features_v2/`이고, v1 파일은 보존한다. 반경·방위·거리 간격·태양 위치·시간별 컬럼은 v1과 동일하다. 아래 마스킹 관련 설명과 기존 결측 수치는 v1에 해당한다.

작업공간 `sun_elec` 루트에서 v2를 생성·검증한다:

```powershell
.\.tools\python\python.exe scripts\prepare_expanded_terrain.py
.\.tools\python\python.exe scripts\test_terrain_features.py
.\.tools\python\python.exe scripts\build_terrain_features.py --terrain-source original-expanded --output-dir ai_sim/data/terrain_features_v2
.\.tools\python\python.exe scripts\validate_terrain_features.py --data-dir ai_sim/data/terrain_features_v2
```

기존 v2 출력 폴더가 있으면 덮어쓰지 않는다. 제공 좌표의 최대 2km 이동이라는 한계는 v2에서도 동일하다.

### v2 검증 완료

- 253개소의 중심 고도와 182,160개 방향 모두 유효하다.
- train 4,392,566행 중 기존 결측 657,642행, test 2,216,280행 중 기존 결측 327,061행을 모두 해소했다. 총 6,608,846행의 지형 각도·각도 차이·가림 판정 결측은 0행이다.
- v1에서 유효했던 방향별 각도와 시간별 지형 특성은 전부 동일하다. v1 및 v2 산출물의 해시를 확인했다.
- 추가 타일의 12,960,000개 픽셀이 모자이크에서 원본과 정확히 일치한다. 기존 학습 입력·모델·후보 큐는 바꾸지 않았다.
- 합성 테스트 5개, 전체 행의 키·계산식·null 정책 검사가 통과했다. 별도 5개소의 촘촘한 계산과 비교한 지점별 차이 95백분위는 약 0.03~0.62°, 최대는 약 0.87°다. 실제 좌표·DSM 정확도에 대한 검증은 아니다.

## 가장 중요한 한계

제공 좌표는 실제 위치에서 최대 2km 이동하고 소수 네 자리로 반올림했다. 따라서 **제공 좌표에서 계산한 지형 특성**이지 실제 발전소 음영의 정답이 아니다. 실제 위치를 추정하거나 관측소 거리로 좌표를 복원하지 않는다. 원본 DSM은 나무·건물을 포함한다. 패널 위치·설치 높이·방향·각도는 알려져 있지 않다.

과거 문서의 DEM 비활성 방침은 이번 별도 자료 생성 요청에 한해 변경했다. 대회 제출 사용 적격성을 이 자료 생성으로 보증하지 않는다.

## 생성 방식

- 자료: 기존 `data/terrain/south_korea_dem_30m.vrt`와 남한 경계 마스킹 DSM 타일. 원본 manifest의 26개 타일 해시를 확인한다.
- 중심: 제공 위경도. 중심 고도는 DSM 양선형 보간값이며 실제 발전소 고도로 부르지 않는다.
- 반경: 20km. 방위는 진북 0°, 시계 방향(동쪽 90°), 0.5° 간격의 720개 광선.
- 거리: 30m부터 30m 간격, 마지막 20,000m 포함. 중심 30m 이내 및 20km 밖 장애물은 계산하지 않는다.
- 거리·방향: WGS84 기반 중심별 방위정거리 투영(AEQD). 경위도 차이를 미터로 사용하지 않는다.
- 지형: 양선형 보간. 가림 각도는 `atan2(주변 DSM − 중심 DSM − d²/(2R), d)`이며 R=6,371,008.8m이다. 대기 굴절은 모델링하지 않는다.
- 각 광선의 최대 각도를 저장하고 음수는 0°로 제한한다. 따라서 관측점이 높은 경우의 음의 지평선까지 예측하는 모델은 아니다.
- 방위 사이에는 원형 선형 보간한다. 0.5° 광선 사이의 좁은 봉우리·건물은 놓칠 수 있다. 촘촘한 검산 결과를 별도로 남긴다.
- NoData는 해수면 0m가 아니라 알 수 없음이다. 중심이 누락되면 해당 발전소의 가림 각도가 없다. 광선에 누락 구간이 있으면 부분 최대값은 참고용으로 저장하되 시간별 학습값은 null로 남긴다.
- 태양 위치는 기존 `solar_time.py`와 `fixed_model.features`에 맞춘 시간 구간 중앙 근사값이다. 시간 전체의 가림 비율이나 발전 손실률이 아니다.

## 출력

`ai_sim/data/terrain_features_v1/`:

| 파일 | 내용 |
|---|---|
| `horizon_profiles.parquet` | 발전소 × 방위별 최대 각도·방향별 자료 충실도 |
| `site_quality.json` | 중심 DSM 유효성, 최대 각도, 누락 방향, 2km 이내 장애물 지배 방향 수 |
| `train_terrain.parquet` | 기존 train과 `site_id, ts`로 결합하는 시간별 지형 특성 |
| `test_terrain.parquet` | 기존 test와 동일하게 결합하는 시간별 지형 특성 |
| `manifest.json` | 파라미터·버전·해시·출력 건수·완료 여부 |
| `validation.json` | 저장 결과 검증 및 촘촘한 샘플링 비교 |
| `horizon_preview.png` | 일부 제공 좌표의 가림 곡선 확인용 그림 |

### 방향별 필드

- `horizon_sampled_deg`: 유효한 샘플에서 찾은 최대 높이각. `ray_complete=false`이면 누락 방향에 더 높은 장애물이 있을 수 있어 전체 가림 각도로 확정할 수 없다.
- `valid_sample_fraction`: 조사 거리 샘플 중 유효 비율.
- `ray_complete`: 중심과 모든 거리 샘플이 유효한지 여부. 실제 위치 정확성이나 광선 사이의 완전성을 보증하지 않는다.
- `max_blocker_distance_m`: 양의 최대 각도를 만드는 샘플까지 거리. 학습 필수가 아닌 검산용이다. 가림 각도 0이면 null이다.

### 시간별 필드

- `solar_elevation_deg`, `solar_azimuth_deg`: 현재 모델과 일치하는 태양 위치.
- `terrain_horizon_deg`: 해당 태양 방위의 보간 지형 높이각. 필요한 광선이 불완전하면 null.
- `terrain_clearance_deg`: 태양 고도각 − 지형 높이각. 낮에 음수이면 직접광 가림 추정.
- `terrain_direct_blocked`: 낮이며 태양이 지형보다 낮을 때 true. 밤에는 false, 방향 자료가 불완전하면 null.
- `terrain_direction_valid`: 시간별 지형값 산출이 가능한지. null을 0이나 false로 덮어쓰지 않는다.

좌표와 시각만 쓰므로 10시·17시 예보에 같은 파일을 결합한다. 두 예보 회차의 키와 정확히 일치하는지 생성 시 확인한다. 원래 발전량·기상 입력을 복사하거나 수정하지 않는다. 라벨·오차·그룹 선택은 생성 과정에서 사용하지 않는다.

## 재현

`sun_elec` 작업공간 루트에서:

```powershell
.\.tools\python\python.exe scripts\test_terrain_features.py
.\.tools\python\python.exe scripts\build_terrain_features.py
.\.tools\python\python.exe scripts\validate_terrain_features.py
```

출력 폴더가 이미 있으면 덮어쓰지 않고 종료한다. 새로 만들 때는 `--output-dir`로 새 버전을 지정한다. `manifest.json`의 `status=complete`를 확인한다. 사용 라이브러리는 기존 환경의 numpy, pandas, scipy, rasterio, duckdb이며 원자료 해시는 manifest에 기록한다.

## v1 생성·검증 결과

- 방향별 182,160행(253개소 × 720방위), train 4,392,566행, test 2,216,280행을 생성했다.
- 중심 DSM이 유효한 곳은 250개소다. `PV014`, `PV123`, `PV131`은 제공 좌표에서 고도를 얻지 못해 지형값을 null로 남겼다.
- 모든 720방위가 완전한 곳은 143개소이며, 전체 방향의 85.57%가 완전하다.
- 시간별 지형값이 불명인 행은 train 657,642개, test 327,061개다. 이를 가림 없음으로 해석하지 않는다.
- 두 예보 회차와의 키 일치, 키 중복 없음, 산출식·null 정책·값 범위를 검사했다. 시간별 정책 위반은 0행이다.
- 합성 지형 및 태양 위치 일치 테스트 5개가 통과했다.
- 가림 정도가 다른 5개소를 0.25°·15m 간격으로 재계산했다. 같은 내부 제외 반경 30m와 외부 반경 20km를 유지했다. 비교 가능한 방향에서 지점별 차이의 95백분위는 약 0.03~0.43°, 최대 차이는 약 0.67°였다. 이 수치는 샘플링 민감도이며 실제 위치·DSM 자체의 정확도를 의미하지 않는다.
- 자료 파일은 약 70MB다. GPU 학습이나 기존 자료 파일은 수정하지 않았다.

## 출처

produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.

The organisations in charge of the Copernicus programme by law or by delegation do not incur any liability for any use of the Copernicus WorldDEM-30.

남한 마스킹 경계: 통계청 SGIS 원자료, vuski/admdongkor 가공본, CC BY 4.0. 상세 출처·원본 이용 조건은 `../data/terrain/README.md`와 `sources/`의 라이선스를 따른다.
