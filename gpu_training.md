# GB10 CUDA 필수 학습

2026-10-28 사용자 요청으로 `train_fixed.py`, `train_grouped.py`, `train_tuned.py` 모두 CUDA 학습을 필수로 변경했다. 기존 CPU 실행 기록은 이력이며 앞으로의 기본 설정이 아니다.

- `device_type='cuda'`를 기준선·튜닝·조기 종료·재학습 모두에 지정한다. CPU나 OpenCL로 자동 전환하지 않는다.
- 각 실행기는 입력 해시 계산 전에 작은 CUDA 모델을 실제로 학습한다. 실패하면 오류로 끝나며 본 학습을 시작하지 않는다. `nvidia-smi`에 장비가 보이는 것만으로 성공으로 판단하지 않는다.
- `CUDA preflight PASSED`와 실행 manifest의 `gpu.device_type=cuda`, `gpu.probe=passed`를 확인한다.
- CPU 전용 `deterministic`, `force_col_wise` 설정을 제거했다. CUDA와 CPU 점수의 비트 단위 동일성을 보장하지 않는다. 이전 CPU 출력 폴더에서 `--resume`하지 말고 새 GPU 출력 폴더를 사용한다.
- 데이터 읽기·특성 생성·평가 및 현재 예측 호출은 CPU 작업을 포함한다. GPU 사용률이 매 순간 100%일 필요는 없다. `--threads`는 CPU 보조 작업의 스레드 수이며 GPU 전환 옵션이 아니다.

## GB10 준비

기존 학습 프로세스는 자동으로 CUDA로 전환되지 않는다. 실행 중인 프로세스·환경·결과는 자동 수정하지 않는다. 아래 설치는 별도 `.venv-cuda` 환경에서 수행한다. 코드 업데이트는 기존 CPU 작업을 종료한 뒤 수행하는 편이 실행 코드 기록을 일관되게 보존하기 좋다. CPU 결과 폴더는 삭제하지 않는다.

`ai_sim` 폴더에서 장비·툴킷을 확인한다.

```bash
git pull origin main
nvidia-smi
nvcc --version
cmake --version
g++ --version
dpkg-query -W libnccl2 libnccl-dev
nvcc --list-gpu-code | grep sm_121
```

LightGBM의 CUDA 빌드에는 CUDA Toolkit, CMake, 호환되는 C++ 컴파일러가 필요하다. `nvidia-smi`의 CUDA 표시는 드라이버 지원 버전이며 `nvcc` 설치 확인을 대신하지 않는다. 컴파일러가 없거나 native GPU 아키텍처를 지원하지 못하면 해당 오류부터 해결하고 CPU 빌드로 우회하지 않는다.

4.7.0 소스 재검토에서 추가 확인한 조건: CMake 3.28 이상과 NCCL 개발 파일이 필요하다. 기본 NCCL 검색은 `nccl.h`와 `libnccl_static.a`를 찾으므로 Ubuntu에서는 CUDA 버전에 맞는 `libnccl-dev` 설치 여부를 확인한다. 사용자 지정 경로 설치는 `NCCL_ROOT`로 지정할 수 있다. 패키지가 없다고 최신 NCCL을 무조건 설치하면 CUDA 관련 패키지가 함께 변경될 수 있어 현재 `nvcc`·패키지 버전에 맞춰 설치한다. [4.7.0 빌드 설정](https://github.com/lightgbm-org/LightGBM/blob/v4.7.0/CMakeLists.txt), [NCCL 탐색 설정](https://github.com/lightgbm-org/LightGBM/blob/v4.7.0/cmake/modules/FindNCCL.cmake).

`native`는 현재 GB10에 맞는 GPU 코드를 생성하는 공식 옵션이다. `CMAKE_BUILD_PARALLEL_LEVEL=4`는 컴파일 작업 수만 제한하며 이후 학습 성능을 4스레드로 제한하지 않는다. 4.7.0 Python 패키지는 기본 Release 빌드이며 CUDA 컴파일에 `-O3`를 사용한다. 설치 명령이 적합하다는 것과 전체 파이프라인의 GPU 가속 배율이 검증됐다는 것은 다르다.

```bash
python3 -m venv .venv-cuda
source .venv-cuda/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
CMAKE_BUILD_PARALLEL_LEVEL=4 python -m pip install --force-reinstall --no-deps --no-cache-dir --no-binary=lightgbm --config-settings=cmake.define.USE_CUDA=ON --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=native lightgbm==4.7.0
python scripts/gpu_training.py
```

마지막 명령이 실패하면 본 학습을 실행하지 않는다. `CUDA Tree Learner was not enabled`는 CPU 빌드, `no kernel image` 등은 GPU 아키텍처/툴킷 호환성 확인이 필요한 경우다. 설치 후 `requirements-training.txt`의 같은 버전을 일반 재설치해 CUDA 빌드를 덮어쓰지 않는다. 환경을 재설치했다면 항상 위 GPU 검사를 다시 수행한다.

공식 근거: [LightGBM CUDA Python 빌드](https://github.com/lightgbm-org/LightGBM/blob/main/python-package/README.rst#build-cuda-version), [CUDA 설치·native 아키텍처 설정](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html#build-cuda-version). 이 안내는 GB10 실측 성공 기록이 아니라 해당 장비에서 수행할 절차다.

## 검사와 본 실행

```bash
RUN_CUDA_TESTS=1 python -m unittest discover -s tests -v
python -u scripts/train_tuned.py --output-dir results/tuning_smoke_cuda_v1 --threads 8 --smoke
```

GPU 검사와 축소 실행 모두 성공한 뒤 본 학습을 실행한다.

```bash
nohup python -u scripts/train_tuned.py --output-dir results/tuning_cuda_v1 --threads 8 --trials 20 --timeout 14400 > tuning_cuda_v1.log 2>&1 < /dev/null &
tail -f tuning_cuda_v1.log
```

별도 터미널에서 `watch -n 1 nvidia-smi`로 GPU 활동을 확인한다. CUDA 사전 검사 성공과 학습 로그도 함께 확인한다. 본 실행은 한 번만 시작한다. 재시작은 프로세스가 종료된 뒤 같은 명령·환경·출력에 `--resume`을 추가한다.

## 개발 환경 검증

Windows에서 39개 테스트 중 38개 통과, 실제 CUDA 학습·저장 재현 테스트 1개는 제외했다. 장비에서는 `RUN_CUDA_TESTS=1`로 이 검사를 반드시 활성화한다. CPU/OpenCL 파라미터 거부, CUDA 사전 검사 실패 시 재시도·CPU 대체 금지, 기준선과 모든 trial의 CUDA 설정 전달을 검증했다. 실제 Windows CPU 빌드에 사전 검사를 실행해 CUDA 미지원 오류와 종료 코드 1을 확인했다. GB10 CUDA 학습 성공·속도·점수는 아직 측정하지 않았다.
