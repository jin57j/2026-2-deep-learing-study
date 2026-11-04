# Pet mini 데이터 출처와 가공

Oxford-IIIT Pet: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar,
*Cats and Dogs*, CVPR 2012.

- 원본: https://www.robots.ox.ac.uk/~vgg/data/pets/
- 라이선스: CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/
- 이미지 저작권은 원 소유자에게 있다. 이 재가공 데이터도 CC BY-SA 4.0으로 배포한다.

제공 이미지는 RGB 160×160 JPEG다. 원본의 종횡비를 유지한 크기 조정과
회색 128 padding이 적용되어 있다. 원본 ID·분할·label은 각 CSV에 기록되어 있다.

| 자료 | 원본 pool | 교육용 Train / Validation / Test |
|---|---|---|
| base | trainval의 고양이 3품종·개 3품종 | 원본 360 / 90 / 90 |
| domain | test의 Birman·Ragdoll·Siamese | 원본 30 / 120 / 150 |
| dog_breeds | test의 개 25품종 | 원본 1,000 / 500 / 500 |

이는 교육용 재분할이며 Oxford 공식 benchmark Test가 아니다.
세 pool의 원본 ID는 겹치지 않는다. domain은 품종별 Train 10장 / Validation 40장 /
Test 50장, dog_breeds는 품종별 Train 40장 / Validation 20장 / Test 20장이다.

기본 품종 ID는 Birman=0, Ragdoll=1, Siamese=2, Beagle=3, Pug=4, Shiba Inu=5다.
domain도 같은 ID를 사용한다. dog_breeds는 별도 0~24 label이며 이름은 `classes.json`에 있다.

기본 `classification.csv`는 Train 원본 360장과 제공된 두 증강 view 720장,
Validation/Test 원본 각 90장을 포함해 총 1,260행이다.
제공 증강은 좌우 반전·밝기 변화, 머리 보존 crop이다. 파생 view는 원본과 같은 split에 속한다.
머리 위치 회귀에는 `base/manifest.csv`의 원본 540장을 사용한다.
Box는 resize/padding에 맞춰 변환한 normalized `(xmin,ymin,xmax,ymax)`다.

C는 domain Train의 원본 30장에 밝기·채도 변형본 15장과 shift ±8% 변형본 15장을 메모리에서 한 번 만들어 추가한다.
이미지 파일을 추가 생성하지 않으며 Validation/Test와 위치 회귀 사진에는 이 변형을 적용하지 않는다.
