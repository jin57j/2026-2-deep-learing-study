# ResNet · 웹 실습

A~E 모델과 Test 이미지를 선택해 실제 추론을 실행하고, 레이어별 계산값을 채널별 격자로
탐색한다. 모델·JS·이미지를 함께 제공하므로 인터넷·npm·TensorFlow 설치가 필요 없다.

## 실행

부모 `resnet/` 폴더에서 실행한다.

```sh
python3 resnet-web/serve.py --port 8000
```

Windows에서는 `py resnet-web/serve.py --port 8000`으로 실행할 수 있다.
브라우저에서 [실습 화면](http://localhost:8000/resnet-web/)을 연다. 종료는 터미널의 Ctrl+C다.
서버는 이 컴퓨터의 loopback 주소에서만 접속을 받는다. HTML을 더블클릭해서 열지 않는다.
서버는 스크립트 위치를 기준으로 파일을 찾으므로 다른 작업 경로에서 시작해도 된다.

## 사용 순서

1. A~E 모델을 선택하고 문제·입력·출력·학습 정보를 확인한다.
2. Test 목록에서 이미지를 고르면 미리보기와 정답이 표시된다.
3. `추론 실행`을 누르면 레이어별 계산값과 최종 예측을 확인할 수 있다.
4. 완료 후 레이어 목록·채널을 선택해 같은 실행의 값을 다시 살펴본다.
5. 격자 셀을 가리키거나 클릭하면 정확한 숫자를, 펼친 숫자 표에서는 선택 채널 전체를 확인한다.

`레이어 이름 풀이 · 도움말`을 펼치면 `conv4_block5_add` 같은 이름을 구성 요소별로
확인할 수 있다. Shortcut, BN·ReLU·pooling, 분류·위치 Head의 연산도 설명한다.
Dense 이름 끝의 자동 식별 번호는 레이어 순서나 뉴런 수를 뜻하지 않는다.

블록 구조 패널은 현재 stage, 잔차 블록 순서와 bottleneck의 주 경로·shortcut·합류 위치를
표시한다. 현재 연산을 강조하며 계산된 연산을 누르면 해당 출력으로 이동한다. Conv는
공간 커널 격자, 필터 하나의 크기, 필터 개수, stride와 padding을 표시한다. 커널·풀링
설정은 제공 모델 그래프에서 읽은 `catalog.json`의 정보다. BN·ReLU·Add에는 합성곱
커널이 없으며, Stem·Head는 잔차 블록과 구분해 표시한다.

모델·이미지를 바꾸거나 새 실행을 시작하면 이전 실행값을 교체한다. 페이지를 새로 열면
이전 실행을 복원하지 않는다. 화면에 표시된 backend가 실제 연산 backend다.
WebGPU 초기화·모델 확인 실패 시 WebGL, WASM 순으로 시도한다.

## 모델과 공유 데이터

| 모델 | 내용 | 선택 가능한 Test |
|---|---|---|
| A | 기본 6품종 분류 | base 90장 · domain 150장 |
| B | A에서 도메인 원본으로 fine-tuning | domain 150장 · base 90장 |
| C | A에서 도메인 원본·변형본으로 fine-tuning | domain 150장 · base 90장 |
| D | A의 고정 GAP 특징 + 개 25품종 Head | dog_breeds 500장 |
| E | A의 고정 conv4 공간 특징 + 좌표 Head | base 90장 |

웹은 `../resnet-classroom/data/`의 이미지·CSV를 읽는다. 두 폴더의 이름과 상대 위치를
유지하고 `resnet/` 전체를 함께 복사한다. `assets/`, `models/`, `catalog.json`,
HTML·CSS·JS, `serve.py`는 실행에 필요하다.

웹 가중치는 Python의 H5와 다른 형식이며 `models/`에서 읽는다. Python에서 재학습해도
이 파일은 자동으로 바뀌지 않는다. 학습 코드·설정은 [Python 실습 안내](../resnet-classroom/README.md),
제공 가중치의 성능·해석은 [최종 결과](../resnet-classroom/RESULTS.md)를 따른다.

TensorFlow.js와 backend는 Apache-2.0이며
[고지](assets/THIRD-PARTY.txt)·[라이선스](assets/APACHE-2.0.txt)·원래 JS
라이선스 헤더를 유지한다. 이미지 출처와 배포 조건은
[data/SOURCE.md](../resnet-classroom/data/SOURCE.md)에 있다.
