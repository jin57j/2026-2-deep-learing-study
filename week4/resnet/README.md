# Week 4 - ResNet 전이학습 실습

준비된 이미지와 A~E 최종 모델로 분류, 도메인 적응, 데이터 증강, 특징 재사용,
머리 위치 회귀를 확인한다. 웹에서 레이어의 실제 계산값을 살펴보거나 Python으로
추론하거나 재학습할 수 있다. Python 실습은 **저장소 루트의 공통 uv 환경에서 CPU로 실행한다.**

## 구성

```text
week4/
  shared/                  # Python 실습과 과제가 함께 쓰는 공통 코드
  resnet/
    README.md
    resnet-classroom/       # CPU용 Python 학습 및 추론 코드와 최종 H5 가중치
      data/                # 두 실습이 함께 사용하는 이미지와 CSV, 클래스 목록
      a_base/              # A: 기본 6품종 분류
      b_domain/            # B: 도메인 fine-tuning
      c_augmentation/      # C: 증강 데이터를 사용한 fine-tuning
      d_dog_breeds/        # D: A의 특징으로 개 25품종 분류
      e_localization/      # E: A의 공간 특징으로 머리 위치 회귀
    resnet-web/             # 브라우저 추론과 레이어 탐색
      assets/              # 로컬 TensorFlow.js와 backend, 라이선스
      models/              # 웹용 GraphModel과 가중치
      serve.py             # Python 표준 라이브러리 서버
      ...
```

두 하위 폴더의 이름과 배치를 유지한다. 웹은 `resnet-classroom/data/`를 참조한다.
Python의 H5와 웹의 모델 가중치는 형식이 달라 각각 필요하다.
Python 환경은 루트의 [pyproject.toml](../../pyproject.toml), [uv.lock](../../uv.lock),
[.python-version](../../.python-version)을 함께 사용한다.

## 웹으로 실행

저장소 루트에서 다음을 실행한다. Python 3만 필요하며 npm이나 TensorFlow 설치는
필요 없다. 제공 파일로 추론할 때 인터넷에 접속하지 않는다.

```sh
cd week4/resnet
python3 resnet-web/serve.py --port 8000
```

Windows에서는 마지막 명령을 `py resnet-web/serve.py --port 8000`으로 실행할 수 있다.
브라우저에서 [실습 화면](http://localhost:8000/resnet-web/)을 열고 모델과 이미지를 선택한 뒤
`추론 실행`을 누른다. 레이어와 채널을 선택해 계산값을 확인한다. 종료는 터미널의 Ctrl+C다.
HTML을 더블클릭하지 않고 서버를 통해 접속한다.

자세한 조작법은 [웹 실습 안내](resnet-web/README.md)에 있다.

## Python으로 실행

**저장소 루트에서** 다음을 실행한다. uv가 루트의 Python 3.12 설정과 `uv.lock`을 사용해
공통 `.venv`를 준비한다. 다른 주차와 같은 환경을 사용한다.

```sh
uv sync --locked
uv run --locked python week4/resnet/resnet-classroom/infer.py
```

Windows에서도 같은 uv 명령을 사용한다. 학습과 추론은 CPU에서 실행한다.

추론은 제공된 최종 가중치를 사용한다. 재학습 방법, 데이터와 학습 설정은
[Python 실습 안내](resnet-classroom/README.md), 제공 모델의 성능과 해석은
[최종 결과](resnet-classroom/RESULTS.md)를 따른다.

재학습은 해당 `final.weights.h5`를 덮어쓴다. A를 바꾸면 B~E도 다시 학습해야 한다.
웹 모델은 별도로 변환한 배포 파일이므로 Python 재학습 결과가 자동 반영되지 않는다.
웹의 WebGPU, WebGL, WASM backend는 Python의 CPU 실행 설정과 별개다.

## 공유 및 출처

웹을 공유할 때는 `resnet/` 전체를 함께 복사한다. Python 실습은 `week4/shared/`와 이 저장소의 루트 환경
설정을 함께 사용한다.

이미지 출처, 가공 방법, 배포 조건은 [data/SOURCE.md](resnet-classroom/data/SOURCE.md),
TensorFlow.js 라이선스는 [THIRD-PARTY.txt](resnet-web/assets/THIRD-PARTY.txt)와
[Apache-2.0](resnet-web/assets/APACHE-2.0.txt)에 있다.
