# 태양광 하루 전 예측·집합자원 구성 참고자료

정리일: 2026-10-26. 폴더명은 사용자 요청대로 `refference`를 사용한다.

프로젝트 적용 순서와 구체적인 예측 설계는 [발전량 예측 계획](../forecast_plan.md)을 함께 참고한다.

2026-10-27 [문헌 반영 검토](../reference_application_review.md)에 현재 코드·계획의 반영 수준, 원문에서 확인한 설정, 적용이 어려운 입력 조건과 후속 실험 R1~R7을 기록했다. 아이디어 목록과 실제 학습·검증 완료를 구분한다.

2026 대회의 공개 규정·데이터 안내·제출·평가·공지·FAQ 조사와 기존 문서 대조는 [DataEn 2026 OIBC 공개 자료 조사](dataen_2026/README.md)에 정리했다. 출처별 접근 상태는 [source manifest](dataen_2026/source_manifest.json)에 기록했다.

구름 이동·발달과 일사량 예측은 [별도 조사: cloud_dynamics](cloud_dynamics/README.md)에 정리했다. 연구 4편과 국내 AMV 기술자료 1건, 입력 후보와 하루 전 적용 한계를 포함한다. PDF 4개를 보관하고 나머지 1편은 원문 링크로 제공한다.

**규정 정정:** [운영팀 Q&A](https://dataen.ai/challenge/2026/board/cfe4c0fc-7951-4ee9-b293-4730e42029ec)에 따라 자유롭게 공개되고 상업적 이용이 허용된 외부 소스·모델은 사용할 수 있다. 이 폴더의 논문·PDF는 방법 이해를 위한 연구 참고이며, 실제 데이터 입력에는 별도로 원자료의 라이선스와 전날 17시 입수 가능성을 확인한다. 공식 식은 `W = Σ_gΣ_t 0.1×SMP[t]×k[t]×|A[g,t]-F[g,t]|`, `S = 매전수익 − W`이며 `k` 경계는 제공 파일로 확인한다. 논문의 RMSE·MAE 개선이 대회 S 개선을 보장하지 않는다.

## 먼저 읽을 자료

| 순서 | 자료 | 읽는 목적 | 예측 범위 |
|---|---|---|---|
| 1 | 08 IEA PVPS 지역 예측 보고서 | 기상예보→전처리→집합 발전량 예측 전체 흐름 | 1~3일 전 |
| 2 | 01 Gigoni 등 | 단순 모델과 복잡한 모델, 기상예보의 영향을 비교 | 하루 전 시간별 |
| 3 | 04 Lindberg 등 | 예측 정확도와 거래 가치의 차이, 확률예측 | 하루 전 |
| 4 | 06 Nuño 등 | 집합 예측오차의 공간·시간 의존성 | 하루 전 오차 시나리오 |
| 5 | 07 Yang 등 | 발전소별 합산 예측과 집합 직접 예측 정합화 | 하루 전 및 1분 전 별도 사례 |
| 국내 사례 | 09 Jeon 등 | 한반도 관측소의 공간·시간·다변량 그래프 | 하루 전 시간별 일사 |
| 보조 | 02 pvlib / 03 Sharma / 05 GNN | 물리 특성 / 확률모델 비교 / 시공간 구조 | 각 항목 참조 |

## 01. 하루 전 시간별 예측의 기본 비교

- **서지:** Lorenzo Gigoni, Alessandro Betti, Emanuele Crisostomi, Alessandro Franco, Mauro Tucci, Fabrizio Bizzarri, Debora Mucci (2018). *Day-Ahead Hourly Forecasting of Power Generation from Photovoltaic Plants*. IEEE Transactions on Sustainable Energy, 9(2), 831–842.
- **DOI:** [10.1109/TSTE.2017.2762435](https://doi.org/10.1109/TSTE.2017.2762435)
- **확인 출처:** [저자 공개본 arXiv](https://arxiv.org/abs/1903.06800). 저널 발행은 2018년, arXiv 업로드는 2019년이다.
- **공개 PDF:** [로컬 파일](pdf/01_gigoni_2018_day_ahead.pdf) / [원본](https://arxiv.org/pdf/1903.06800)
- **내용:** 규모·기술이 다른 32개 태양광 발전소의 1년 자료로 예측 방법과 기상 조건·예보의 영향을 비교한다.
- **프로젝트 적용 제안:** 같은 기간·입력으로 지속성 기준, 트리 모델, 물리 보정 모델을 비교한다. 모델 복잡도에 앞서 기상예보와 시간 정렬의 영향을 확인한다.
- **한계:** 국내 기상·대회 정산식·2개 그룹 배분을 검증한 자료가 아니다. 공개본은 저자 업로드 판본이며 arXiv의 공개 열람이 임의 재배포 허가를 뜻하지 않는다.

## 02. Python 물리 모델과 특성 생성

- **서지:** William F. Holmgren, Clifford W. Hansen, Mark A. Mikofski (2018). *pvlib python: a python package for modeling solar energy systems*. Journal of Open Source Software, 3(29), 884.
- **DOI·확인 출처:** [10.21105/joss.00884](https://joss.theoj.org/papers/10.21105/joss.00884)
- **공개 PDF:** [로컬 파일](pdf/02_holmgren_2018_pvlib.pdf) / [원본](https://joss.theoj.org/papers/10.21105/joss.00884.pdf)
- **내용:** 태양 위치, 청천 일사, 경사면 일사, DC 출력 및 AC 변환 등의 Python 구현을 소개한다. 특정 예측 시간대의 성능 비교 논문은 아니다.
- **연구 참고:** 태양 위치·청천 기준 출력·온도 영향을 분리하는 사고방식을 참고한다. 제공 timestamp·좌표의 수학적 태양 위치와 제공 예보, 학습구간 패턴을 기준선으로 쓰고 외부 탁도·기후 자료는 라이선스·발행 시점 확인 후 별도 후보로 비교한다.
- **한계:** 패널 방위·경사·인버터 정보가 없으면 물리 출력 자체에 가정이 들어간다. DEM 경사와 패널 경사를 동일시하지 않는다. 논문은 CC BY 4.0, 소프트웨어는 논문에 BSD 3-clause로 명시되어 있으며 실제 사용 버전도 확인한다.

## 03. 확률예측에서 복잡한 모델이 필요한지 비교

- **서지:** Vinayak Sharma, Jorge Angel Gonzalez Ordiano, Ralf Mikut, Umit Cali (2021). *Probabilistic Solar Power Forecasting: Long Short-Term Memory Network vs Simpler Approaches*. arXiv:2101.08236.
- **DOI:** [10.48550/arXiv.2101.08236](https://doi.org/10.48550/arXiv.2101.08236)
- **확인 출처:** [arXiv 원문 페이지](https://arxiv.org/abs/2101.08236)
- **공개 PDF:** [로컬 파일](pdf/03_sharma_2021_probabilistic.pdf) / [원본](https://arxiv.org/pdf/2101.08236)
- **내용:** GEFCom14 공개 자료로 하루 전 확률예측에서 LSTM과 더 간단한 방법을 비교한다.
- **프로젝트 적용 제안:** 평균 하나에 더해 분위수 예측을 만들고, 단순한 분위수 모델을 신경망과 비교한다. 구간의 보정 상태와 최종 정산 손실을 별도 평가한다.
- **한계:** arXiv 페이지에 학회 제출본으로 표시되므로 해당 페이지로 동료심사 확정을 주장하지 않는다. 대회 정산식에 적합한 제출값은 별도 결정해야 한다. 페이지의 라이선스 링크는 CC BY 4.0이다.

## 04. 정확도뿐 아니라 거래 가치까지 평가

- **서지:** O. Lindberg, D. Lingfors, J. Arnqvist, D. van der Meer, J. Munkhammar (2023). *Day-ahead probabilistic forecasting at a co-located wind and solar power park in Sweden: Trading and forecast verification*. Advances in Applied Energy, 9, 100120.
- **DOI:** [10.1016/j.adapen.2022.100120](https://doi.org/10.1016/j.adapen.2022.100120). DOI에 2022가 있지만 발행연도는 2023이다.
- **확인 출처·공개 PDF:** [대학 저장소 원문](https://uu.diva-portal.org/smash/get/diva2%3A1770882/FULLTEXT01.pdf) / [로컬 파일](pdf/04_lindberg_2023_trading.pdf)
- **내용:** 스웨덴의 풍력·태양광 결합 단지를 대상으로 NWP 후처리, 하루 전 확률예측, 집합 효과와 거래 가치를 분석한다.
- **프로젝트 적용 제안:** 예측 정확도와 금전적 손실을 나누어 평가하고, 확률분포를 이용해 정산식상 기대 페널티가 작은 제출값을 고르는 실험을 설계한다.
- **한계:** 풍력·태양광 결합과 스웨덴 시장 사례다. 태양광만 있는 국내 2그룹 구성의 최적 비율이나 성과로 그대로 옮기지 않는다. CC BY 4.0 공개 논문이다.

## 05. 발전소 사이 시공간 관계를 그래프로 학습

- **서지:** Jelena Simeunović, Baptiste Schubnel, Pierre-Jean Alet, Rafael E. Carrillo (2021 공개본 v2). *Spatio-temporal graph neural networks for multi-site PV power forecasting*. arXiv:2107.13875v2; IEEE Transactions on Sustainable Energy 게재 승인본으로 표시.
- **관련 저널 DOI:** [10.1109/TSTE.2021.3125200](https://doi.org/10.1109/TSTE.2021.3125200)
- **공개본 DOI·확인 출처:** [10.48550/arXiv.2107.13875](https://arxiv.org/abs/2107.13875)
- **공개 PDF:** [로컬 파일](pdf/05_simeunovic_2021_spatiotemporal.pdf) / [원본 v2](https://arxiv.org/pdf/2107.13875v2)
- **내용:** 스위스 실제 304개·모의 1,000개 시스템에서 발전 시계열의 그래프 관계를 사용하는 GCLSTM·GCTrafo를 제시한다. 검증 범위는 최대 6시간 전이며 NWP 단일지점 방법 대비 비교는 최대 4시간이다.
- **프로젝트 적용 제안:** 발전소의 상대 위치·과거 출력 관계를 이용하는 모델 구조를 참고한다. 하루 전 모델에서는 미래 기상예보를 함께 쓰는 별도 검증이 필요하다.
- **한계:** 이 논문은 하루 전 우수성의 근거가 아니다. 좌표 2km 오차가 있는 우리 데이터의 근거리 그래프 연결을 정밀한 실제 위치로 간주하지 않는다. 공개 열람과 재배포 허가는 구분한다.

## 06. 집합 예측오차의 공간·시간 의존성

- **서지:** Edgar Nuño [DTU 서지명: Nuño Martinez], Matti Koivisto, Nicolaos Cutululis, Poul Sørensen (2018). *On the simulation of aggregated solar PV forecast errors*. IEEE Transactions on Sustainable Energy, 9(4), 1889–1898.
- **DOI:** [10.1109/TSTE.2018.2818727](https://doi.org/10.1109/TSTE.2018.2818727)
- **확인 출처·공개 PDF:** [DTU 저자 원고](https://backend.orbit.dtu.dk/ws/files/149828757/revision_new_2_1_.pdf) / [로컬 파일](pdf/06_nuno_2018_aggregated_errors.pdf)
- **내용:** 면적·용량·기후가 다른 지역의 하루 전 집합 예측오차 시나리오를 만들며, 오차 분포·자기상관·지역 간 상관을 함께 다룬다.
- **프로젝트 적용 제안:** 개별 출력의 상관보다 검증 예측 잔차의 동시간대 공분산을 살펴본다. 두 그룹 후보를 동일한 잔차 시나리오로 평가하고 공식 W로 순위를 정한다.
- **한계:** 목적은 현실적인 오차 시나리오 시뮬레이션이며, 가장 정확한 점예측이나 이 대회의 2분할 최적해를 제시하는 연구가 아니다. DTU는 개인 학습·연구용 1부 다운로드를 허용하며 추가 배포·상업적 사용 제한을 명시한다. 자료 자체를 제출 코드에 포함하지 않는다.

## 07. 발전소별 예측과 집합 예측의 일관성

- **서지:** Dazhi Yang, Hao Quan, Vahid R. Disfani, Licheng Liu (2017). *Reconciling solar forecasts: Geographical hierarchy*. Solar Energy, 146, 276–286.
- **DOI:** [10.1016/j.solener.2017.02.010](https://doi.org/10.1016/j.solener.2017.02.010)
- **확인 출처:** [출판사 초록·서지 페이지](https://www.sciencedirect.com/science/article/abs/pii/S0038092X17301020)
- **보관 상태:** 링크와 요약만 보관. 공개 PDF를 확보하지 않았으며 유료 접근을 우회하지 않았다.
- **내용:** 서로 다른 수준의 예측을 공분산 기반 정합화로 맞추어 하위 예측의 합과 상위 예측을 일치시킨다. 318개 모의 PV의 NWP 기반 하루 전 사례와 17개 일사 센서의 1분 전 사례가 별도로 포함된다.
- **프로젝트 적용 제안:** 발전소별 예측 합산, 그룹 직접 예측, 전체 합계 예측을 비교하고 필요하면 정합화한다. 잔차 공분산은 시간 순서를 지킨 검증 예측에서 추정한다.
- **한계:** 예측 정합화는 발전소의 그룹 배정을 정하는 최적화와 다르다. 저자들의 정확도 개선 결과가 우리 W의 개선을 보장하지 않는다. 전문을 로컬로 확보하지 않아 이 설명은 출판사 공개 초록·본문 발췌 범위다.

## 08. 지역 단위 하루 전 예측의 공식 기술 보고서

- **서지:** M. Pierro, C. Cornaro, D. Moser, A. Betti, M. Moschella, E. Collino, D. Ronzio, D. van der Meer, J. Widen, L. Visser, T. AlSkaif, W. van Sark (2020). *Regional solar power forecasting*. IEA PVPS, Report T16-01:2020, 76쪽, ISBN 978-3-906042-88-6.
- **DOI:** 확인한 공식 보고서에 기재 없음. 보고서 번호·ISBN으로 식별한다.
- **확인 출처·공개 PDF:** [IEA PVPS 원문](https://iea-pvps.org/wp-content/uploads/2020/07/IEA_PVPS_T16_regional_forecast.pdf) / [로컬 파일](pdf/08_iea_2020_regional_forecasting.pdf)
- **내용:** 네덜란드 150개 시스템과 이탈리아 지역·전국 자료에서 1~3일 전 지역 발전량 예측을 비교한다. NWP, 전처리, 특성 생성, 트리·신경망·아날로그 앙상블, 모델 결합을 다룬다.
- **프로젝트 적용 제안:** 개별 발전소 정보가 부족할 때의 집합 직접 예측과, 동일 기상예보·동일 기간을 사용한 공정한 모델 비교를 참고한다. 우선 3장 데이터, 4장 방법, 5.2.1절 면적과 오차, 5.4절 모델 결합을 읽는다.
- **한계:** 확률예측 평가는 보고서 범위 밖이며 한국 정산식 최적화 연구가 아니다. 공식 공개 배포본이지만 별도의 개방형 재배포 라이선스는 확인하지 않았다. 서지 저자 표기는 대학 포털 오기 대신 공식 PDF를 기준으로 했다.

## 09. 한반도 관측자료를 활용한 다변량 시공간 그래프

- **서지:** Hyeon-Ju Jeon, Min-Woo Choi, O-Joun Lee (2022). *Day-Ahead Hourly Solar Irradiance Forecasting Based on Multi-Attributed Spatio-Temporal Graph Convolutional Network*. Sensors, 22(19), 7179.
- **DOI:** [10.3390/s22197179](https://doi.org/10.3390/s22197179)
- **확인 출처:** [PubMed 서지·초록·그림 설명](https://pubmed.ncbi.nlm.nih.gov/36236280/). 무료 전문 링크: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9572285/), [출판사](https://www.mdpi.com/1424-8220/22/19/7179).
- **보관 상태:** 링크와 요약만 보관. 이번 확인에서 PMC는 브라우저 확인 화면, 출판사는 요청 제한(429)을 반환해 전문 PDF를 확보하지 않았다.
- **내용:** 한반도 ASOS 관측소의 일사와 기상변수로 동적 그래프를 만들고 시간 변화를 학습한다. 공간적 인접성, 시간 변화, 변수 다양성의 기여를 비교한다. 거리와 과거 일사 상관에 기반한 그래프를 그림에서 비교한다.
- **프로젝트 적용 제안:** 한국 공간 자료로 검증된 모델 구조의 참고 사례다. 거리 그래프와 과거 잔차/출력 관계 그래프를 비교하고 구름·풍속·습도 등 다변량의 추가 효과를 검증한다.
- **한계:** 예측 대상은 일사이며 발전량·정산금 자체가 아니다. 현재 검토는 공개 초록·그림 설명 범위로, 전문의 입력 관측 구간·발표 시각·검증 분할을 확인하기 전에는 우리 하루 전 제출 조건에 그대로 부합한다고 단정하지 않는다. 이 논문이 위치 2km 오차나 2개 VPP 분할을 검증한 것은 아니다.

## 이 프로젝트에서 구분할 것

다음은 위 문헌을 참고한 **우리의 적용 제안**이며 논문이 그대로 입증한 결과가 아니다.

1. 하루 전 예측에서는 제출 시점에 이미 발표·입수된 기상예보를 사용한다. 사후 관측·재분석을 당시 예보인 것처럼 쓰지 않는다.
2. DEM은 공개·상업적 이용 조건과 익명 좌표 오차를 확인한 뒤 별도 검증 후보로 비교한다. 실제 발전소 위치를 복원하지 않는다.
3. 제공자료 기반 발전소 특성화 → 시간순 OOF 잔차 → 유사도 군 → 2개 VPP 배분 → 후속 홀드아웃 순으로 비교한다. 중간 군은 최종 VPP가 아니며 예측기가 바뀌면 잔차와 배분의 우열도 바뀔 수 있다.
4. 모든 발전소를 정확히 하나의 그룹에 배정하는 제약을 지키고 두 그룹의 합산 S로 평가한다. 지리적 분산·공분산·RMSE는 후보 생성 또는 보조 지표다.
5. 분위수·시나리오를 쓸 때 집합의 불확실성은 발전소 간 의존성을 보존해 계산한다. 개별 발전소의 같은 분위수를 단순 합하면 일반적으로 집합의 그 분위수가 되지 않는다.
6. 대회 W의 정확한 식이 확인되면 예측 점값을 결정하는 손실함수와 그룹 탐색의 목적함수를 그 식에 맞춘다.

## 파일·재현 정보

- `pdf/`: 공개 출처에서 받은 PDF. 원본을 수정하지 않는다.
- `downloads.json`: 원본/최종 URL, 수신 시각, 크기, SHA-256, 다운로드 성공 여부.
- `download_references.py`: Python 표준 라이브러리로 공개 PDF를 받는 스크립트. 응답이 `%PDF-`로 시작하는지 확인하고 체크섬을 기록한다.
- 실행: 작업공간 루트에서 `.\.tools\python\python.exe refference/download_references.py`.
- 다운로드 상태의 최종 근거는 `downloads.json`이다. 공개 논문과 소프트웨어의 라이선스는 서로 다르며, 이 폴더의 PDF를 자동으로 외부에 재배포하지 않는다.
