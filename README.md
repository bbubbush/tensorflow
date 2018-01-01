# tensorflow
SUNG KIM의 강의를 보고 따라하는 tensorflow

Day 1 (17.12.19)

#### [ LAB1. TensorFlow Basics ]
- tf에서 출력되는 'b'는 Bytes 타입이라는 표시다
- 실행은 Session.run을 통해 실행

tf의 큰 흐름
1. graph를 정의한다
2. session을 통해 실행한다
3. 결과값을 출력한다

Tensor는 Array를 말함
핵심 개념
1. Ranks : 차원의 수
2. Shapes : 각 엘리먼트의 개수 
3. Types : 데이터 타입

#### [ Lecture 2. Linear Regression ]
Linear(선형) : y의 범위가 몇개의 단계로 나눠지는 방법 traning data 필요
	- H(x) = Wx + b의 형태로 나온다고 가설을 세움
	- 좋은 가설은 실제 데이터와 가설간의 데이터 갭이 적은 것이다
	- (H(x)-y)^2을 통해 차이를 비교  (이것을 cost function이라고 함)
	- minimize cost(W, b)를 가장 작게 하는 W, b를 구하는 것이 학습의 목표

#### [ Lecture 3. How to minimize cost ]
Hypothesis and Cost
H(x) = Wx + b
cost(W,b) = sum((H(x^i)-y^i)^2) / m

이것이 기본 식인데 간편함을 위해 b값을 제거
H(x) = Wx
cost(W) = sum((Wx^i - y^i)^2) / m (Hypothesis의 값을 대체하지 않고 그대로 사용)

이 식을 minimize 하기 위해 cosw(W)의 값을 찾아보기

|x|y|
|---
|1|1|
|2|2|
|3|3|

*위와 같은 dataset이 있을 때 W = 1, cost(W)의 값은?*
직접 대입해 본다
(1*1-1)^2 + (1*2-2)^2 + (1*3-3)^2 = 0,  m으로 나누지 않아도 0이므로 cost(W) = 0이 됨

*W = 0, cost(W)의 값은?*
(0*1-1)^2 + (0*2-2)^2 + (0*3-3)^2 = (1 + 4 + 9) / 3 = 14/3 = 4.667

*W = 2, cost(W)의 값은?*
(2*1-1)^2 + (2*2-2)^2 + (2*3-3)^2 = (1 + 4 + 9) / 3 = 4.667

위 3개의 문제를 통해 x축을 W, y축을 cost로 하는 2차함수를 그리면 (1,0)을 꼭지점으로 하는 2차함수가 그려진다. 이렇듯 cost의 최소값을 찾기 위해서는 기계적으로 2차함수를 그려 가장 작은 cost값을 찾으면 되는데 여기에 사용되는 알고리즘이 **Gradient descent algorithm** 이다.