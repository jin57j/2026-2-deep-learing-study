# ResNet - Python 실습

준비된 mini 데이터와 최종 가중치로 기본 분류, 도메인 fine-tuning, 데이터 증강,
개 품종 분류, 머리 위치 회귀를 확인한다. 데이터와 가중치는 이 폴더에 포함하며,
Python 환경은 저장소 루트의 공통 uv 설정을 사용한다. 학습과 추론은 CPU로 실행한다.

전체 구성과 빠른 실행은 [통합 안내](../README.md), 브라우저 추론과 레이어별 계산값 탐색은
옆의 [웹 실습 안내](../resnet-web/README.md)를 따른다.

## 폴더 구성

```text
resnet-classroom/
  data/                # 제공 mini 이미지, label CSV, 출처
  a_base/              # A: 기본 6품종 분류
  b_domain/            # B: 원본 30장으로 도메인 fine-tuning
  c_augmentation/      # C: 원본 30장 + 변형본 30장으로 fine-tuning
  d_dog_breeds/        # D: 개 25품종 분류
  e_localization/      # E: 머리 위치 회귀
  infer.py             # 다섯 모델의 결과를 한 번에 확인
  RESULTS.md           # 제공 최종 가중치의 결과
```

공통 설정, 데이터 읽기, 모델 코드는 [week4/shared](../../shared/)를 과제와 함께 사용한다.

각 모델 폴더에는 `train.py`, `infer.py`, `final.weights.h5`가 있다.
E의 공간 특징 선택과 좌표 Head는 [e_localization/model.py](e_localization/model.py)에 둔다.
A, B, C는 전체 모델 가중치, D, E는 **A의 특징에 맞춰 학습한 Head 가중치**다.
D, E는 `a_base/final.weights.h5`를 읽어 encoder를 구성한다.

| 모델 | 학습하는 부분 | 출력 |
|---|---|---|
| A | 고정 ImageNet ResNet50 뒤 새 Head | 6품종 logits |
| B | A의 conv4~5 Conv + 기존 Head | 같은 6품종 logits |
| C | B와 같은 범위, A에서 별도로 시작 | 같은 6품종 logits |
| D | A의 고정 2,048차원 특징 뒤 새 Head | 개 25품종 logits |
| E | A의 고정 conv4 공간 특징 뒤 새 Head | 머리 사각형 좌표 4개 |

B, C는 conv1~3과 모든 BatchNorm 가중치와 통계를 고정한다.
고양이 3품종을 평가할 때도 기존 6개 전체 출력에서 선택하며 대상 외 출력은 오답이다.
D, E가 A의 특징을 서로 다른 출력에 재사용하는 과정을 representation learning과 연결한다.

## 설치와 결과 확인

**저장소 루트에서** 환경을 준비한 뒤 이 폴더로 이동한다. uv는 상위 디렉토리의
[pyproject.toml](../../../pyproject.toml)과 [uv.lock](../../../uv.lock)을 사용하고,
Python 3.12와 루트 `.venv`를 다른 주차와 함께 사용한다.

```sh
uv sync --locked
cd week4/resnet/resnet-classroom
uv run --locked python infer.py
```

Windows에서도 같은 명령을 사용한다.
제공 가중치로 추론할 때는 다운로드나 학습이 필요 없다.

`infer.py`는 평가 수치와 예측 그림을 보여준다. 같은 간격으로 고른 Test 사진 3장을 사용하며
분류 그림은 정답 품종과 예측 품종을, 위치 그림은 정답 초록 box와 예측 빨강 box를 표시한다.
수치와 해석은 [최종 결과](RESULTS.md)에 모았다.

특정 모델만 확인할 수도 있다.

```sh
uv run --locked python -m a_base.infer
uv run --locked python -m b_domain.infer
uv run --locked python -m c_augmentation.infer
uv run --locked python -m d_dog_breeds.infer
uv run --locked python -m e_localization.infer
```

하위 모듈은 `resnet-classroom/`을 작업 경로로 두고 `-m`으로 실행한다.

노트북에서 이 모듈을 사용할 때는 TensorFlow 연산 전에 `shared.common`을 먼저 불러오면
이 실습의 CPU와 스레드 설정을 적용할 수 있다. 이미 TensorFlow가 초기화된 경우에는
안내 경고를 표시하고 기존 실행 설정으로 계속 진행한다. 따라서 이 경우 장치와 스레드 수는
현재 노트북 설정을 따른다. CPU 설정을 다시 적용하려면 커널을 재시작한다.

## 학습 코드 읽기와 재학습

각 `train.py`는 **데이터 -> 특징/모델 -> 학습 범위 -> compile -> fit -> 최종 저장** 순서다.
[week4/shared/data.py](../../shared/data.py)는 CSV, RGB, label 읽기와 batch 만들기를 담당한다.
[week4/shared/models.py](../../shared/models.py)는 모델 구조 정의와 가중치 읽기 및 저장을 담당한다.
필요한 실행 환경 설정은 [week4/shared/common.py](../../shared/common.py)에 모았다.
학습 범위 설정과 `compile`, `fit`은 각 모델의 학습 파일에서 직접 볼 수 있다.

```sh
uv run --locked python -m a_base.train
uv run --locked python -m b_domain.train
uv run --locked python -m c_augmentation.train
uv run --locked python -m d_dog_breeds.train
uv run --locked python -m e_localization.train
```

재학습은 해당 폴더의 `final.weights.h5`를 덮어쓴다.
A를 재학습했다면 B, C, D, E도 새 A에서 다시 학습한다.
A의 첫 학습에서만 ImageNet 초기 가중치를 Keras가 읽거나 내려받는다.

| 모델 | 학습 입력 | Epoch | Batch | Adam learning rate |
|---|---|---:|---:|---:|
| A | 기본 분류 view 1,080장 | 3 | 32 | 1e-3 |
| B | 도메인 원본 30장 | 4 | 8 | 1e-5 |
| C | 원본 30장 + 고정 변형본 30장 | 4 | 8 | 1e-5 |
| D | 개 원본 1,000장 | 10 | 32 | 1e-3 |
| E | 기본 원본 360장 | 27 | 16 | 1e-4 |

모든 학습은 seed 42에서 시작하고 정해진 마지막 epoch를 저장한다.
분류는 Sparse CE, E는 Huber $\delta=0.1$과 clipnorm=1을 사용한다.
`fit()`과 `predict()`는 `verbose=0`으로 실행한다.

## 제공 데이터와 증강

| CSV | Train / Validation / Test | 용도 |
|---|---|---|
| `data/base/classification.csv` | 1,080 / 90 / 90 | A의 제공 증강 view 포함 분류 학습 |
| `data/base/manifest.csv` | 360 / 90 / 90 | A 원본 평가, E 학습과 평가 |
| `data/domain/manifest.csv` | 30 / 120 / 150 | B, C 학습, A/B/C 도메인 평가 |
| `data/dog_breeds/manifest.csv` | 1,000 / 500 / 500 | D 학습과 평가 |

모든 제공 이미지는 RGB $160 \times 160$이다. 모델 내부 `ResNetPreprocess`에서 전처리하므로
읽은 픽셀 0~255를 미리 0~1로 rescale하지 않는다.
기본 6품종 ID는 Birman=0, Ragdoll=1, Siamese=2, Beagle=3, Pug=4, Shiba Inu=5다.
개 25품종은 별도 ID 0~24이며 [classes.json](data/dog_breeds/classes.json)을 따른다.
위치 label은 0~1 범위의 `(xmin, ymin, xmax, ymax)`다.

C는 원본 사진마다 변형본 하나를 추가한다. 품종별로 밝기와 채도를 바꾼 5장과 shift 5장으로,
총 밝기와 채도 15장 + shift 15장을 만든다. 두 변형을 한 사진에 연속 적용하지 않는다.
밝기는 최대 $\pm 12.75$, 채도는 0.9~1.1배, shift는 가로와 세로 최대 $\pm 8\%$다.
Shift의 빈 영역은 회색 128, 보간은 bilinear다.
사진 선정 seed 141, 밝기와 채도 seed 142, shift seed 143을 사용한다.
변형본은 메모리에서 한 번 만든 뒤 모든 epoch에서 유지하며 Validation/Test에는 적용하지 않는다.

출처, 가공 방법, 라이선스는 [data/SOURCE.md](data/SOURCE.md)에 있다.
이 폴더에는 mini 데이터와 다섯 최종 가중치, 코드와 문서를 포함한다.
