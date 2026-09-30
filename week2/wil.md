이번 주에는 모델이 학습하는 전체 흐름을 조금 더 구체적으로 이해했다. 처음에는 Loss, Gradient, Backpropagation를 각각 따로 알고 있었는데, 실제로는 이들이 하나의 과정으로 이어진다는 점이 중요했다. 모델은 먼저 현재 parameter를 이용해 예측값을 만들고, Cost Function을 통해 실제값과 예측값의 차이를 계산한다. 이후 Loss를 줄이기 위해 각 parameter가 Loss에 얼마나 영향을 주는지를 Gradient로 구하고 Optimizer가 이 Gradient를 이용해 parameter를 업데이트한다. 이를 반복하는 것이 학습의 기본 구조라는 것을 이해했다.
Gradient는 각 parameter에 대한 Loss의 변화율이다. 예를 들어 모델이
y=asin^2(x)+bx
이고 Cost Function이 MSE라면 이때 a와 b에 대한 Gradient는 Chain Rule을 이용해 계산할 수 있다.
즉 여러 데이터가 하나의 parameter에 미치는 영향을 합쳐서 최종 Gradient를 구하는 것이다.
Backpropagation은 Gradient 그 자체가 아니라 최종 Loss에서 시작해 Chain Rule을 이용하여 앞쪽 parameter까지 Gradient를 효율적으로 전달하며 계산하는 방법이다. 모델이 여러 층으로 구성되어 있을 경우 앞쪽 weight가 Loss에 직접 연결되어 있지 않기 때문에, 중간 연산의 미분값들을 곱해가며 거꾸로 계산한다. 따라서 Forward Pass에서는 입력으로부터 예측값과 Loss를 계산하고, Backward Pass에서는 Loss로부터 각 parameter의 Gradient를 계산한다고 정리할 수 있었다.
또한 Gradient와 Optimizer의 역할도 구분하게 되었다. Gradient는 어느 방향으로 Loss가 증가하는가를 알려주는 정보이고, Optimizer는 이를 이용해 실제 parameter를 어떻게 수정할지를 결정한다.  
또한 Gradient를 계산한할 때 컴퓨터가 미분연산을 어떻게 처리할지 궁금해져서 찾아봤다. NumPy로 직접 작성한 Gradient 함수는 사람이 미분한 수식을 코드로 옮긴 것이며, TensorFlow의 자동미분은 각 연산의 미분 규칙과 Chain Rule을 이용해 Gradient를 계산한다. 반면 수치미분은 parameter를 아주 조금 변화시킨 뒤 Loss의 차이를 이용해 Gradient를 근사하는 방법이다.
수치미분은 실제 학습에 주로 사용하는 방식이라기보다, 직접 구현한 Gradient가 맞는지를 확인하는 Gradient Checking 용도로 사용할 수 있다는 점도 알게 되었다. 해석적으로 계산한 Gradient와 수치미분으로 근사한 Gradient가 비슷하게 나오면 미분식과 구현이 올바를 가능성이 높다. 반면 parameter가 많아질수록 모든 parameter에 대해 +epsilon, -epsilon을 계산해야 하기 때문에 실제 딥러닝 학습에는 비효율적이라는것을 알았다.
