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
|1|1|
|2|2|
|3|3|
위와 같은 dataset이 있을 때 W = 1, cost(W) = ?
