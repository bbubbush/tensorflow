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
**Hypothesis and Cost**

H(x) = Wx + b
cost(W,b) = sum((H(x^i)-y^i)^2) / m

이것이 기본 식인데 간편함을 위해 b값을 제거
H(x) = Wx
cost(W) = sum((Wx^i - y^i)^2) / m (Hypothesis의 값을 대체하지 않고 그대로 사용)

이 식을 minimize 하기 위해 cosw(W)의 값을 찾아보기

 x | y 
---|---
 1 | 1 
 2 | 2 
 3 | 3 

*위와 같은 dataset이 있을 때 W = 1, cost(W)의 값은?*

직접 대입해 본다
(1\*1-1)^2 + (1\*2-2)^2 + (1\*3-3)^2 = 0,  m으로 나누지 않아도 0이므로 cost(W) = 0이 됨

*W = 0, cost(W)의 값은?*

(0\*1-1)^2 + (0\*2-2)^2 + (0\*3-3)^2 = (1 + 4 + 9) / 3 = 14/3 = 4.667

*W = 2, cost(W)의 값은?*

(2\*1-1)^2 + (2\*2-2)^2 + (2\*3-3)^2 = (1 + 4 + 9) / 3 = 4.667

위 3개의 문제를 통해 x축을 W, y축을 cost로 하는 2차함수를 그리면 (1,0)을 꼭지점으로 하는 2차함수가 그려진다. 이렇듯 cost의 최소값을 찾기 위해서는 기계적으로 2차함수를 그려 가장 작은 cost값을 찾으면 되는데 여기에 사용되는 알고리즘이 **Gradient descent algorithm** 이다.

cost를 minimize하는데 사용하지만 이 외에도 많은 곳에서 사용되는 알고리즘이다.

**How it works?**

현재의 cost값보다 작은 cost 값을 발견하면 그것을 cost 값으로 갖는다. 이를 반복하여 가장 작은 방향으로 W를 조금 변경한다. 이를 꾸준히 반복하되, W의 값을 너무 크거나 작게 주면 최소값을 못찾고 헤맬 수 있다.

경사도를 구하는 방법은 미분을 이용한다.

**W := W - a(sum((Wx^i - y^i)x^i) / m)** 이 최종적인 Gradient descent algoritym이 된다. 미분을 통해 기울기를 구해 W 값에서 다음 기울기로 이동하면서 비교하게 한다.

**Convex function**

x축을 W, y축을 b, z축을 cost(W, b)의 값으로 할 때, 밥그릇이 뒤집어진 모양의 형태를 Convex function이라고 한다.

이 때는 어떤 cost값으로 시작을 해도 가장 안쪽에 파여있는 한 점을 찾을 수 있다. 하지만 Convex function의 형태가 아니라 산의 모양을 하면 어디서 시작하냐에 따라 최소값이 달라질 수 있어 의도한 결과를 벗어날 수 있다. 따라서 항상 Linear Regresstion을 적용하기 전에 cost(W, b)의 형태가 Convex function의 형태를 하는지 확인해야 한다.

#### [ Lecture 4. Multivariable Linear regression ]

지금까지 한 개의 input(x)만 가지고 했지만 이번에는 두 개 이상의 input(x1, x2...)의 Linear regression을 구하는 방법을 진행. 여러개의 변수가 있을 땐

>H(x1, x2, x3) = w1x1 + w2x2 + w3x3 + b 

이렇게 기존의 식에서 옆으로만 늘어날 뿐 형태는 그대로 유지한다. n이 늘어나면 수식 역시 길어지기 때문에 이를 편하게 관리하기 위해 Matrix의 형태로 관리한다. 이러면 곱셈을 통해 기존의 수식을 만들어 낼 수 있다.

[x1, x2, x3] * [[w1], [w2], [w3]]를 곱할 경우 행렬의 곱셈으로 인해 x1w1 + x2w2 + x3w3의 값이 나온다. 위의 수식과 동일한 값이지만 행렬의 경우 x가 w보다 먼저 나오므로 행렬의 표시를 위해 H(X) = XW의 수식을 쓴다. (기존에는 H(X) = WX의 형태였다)

인스턴스가 [5, 3]이고 웨이트가 [3, 1]개 있으면 H(X)는 [5, 1]이 된다. 형식으로 나타내면 instance[A, B] weight[B, C] Hypothesis[A, C]의 형태이다.

여러개의 값이 입력된다면 [n, 3]이 된다. numpy에서는 -1 혹은 None으로 적어 n개가 들어옴을 나타낼 수 있다.

여기서부터가 본격적인 ML의 입문이다.

### [ Lecture 5-1. Logistic (regression) classification ]

뉴럴네트워킹과 딥러닝에 활용되는 개념.

이번 강의에서 하는 것은 Birary Classification, 즉 둘 중 하나를 고르는 것이다. 스팸메일 or 햄메일, 맞춤게시물을 보여줄지 or 말지 등

기존 Linear regression을 사용하게 되면 튀는 데이터가 등장했을 때 다른 데이터에 큰 영향을 끼칠 수 있다. 또한 H(x) = Wx + b의 식은 0~1의 값을 도출해야 하는 binary에 적합하지 않아 가설을 수정해야 한다. 

Logistic Hypothesis(sigmoid 혹은 logistic function이라고 부름) : H(X) = 1/1+e^(-WX)

이제는 cost함수의 형태를 봐야 하는데 sigmoid의 형태를 그래프로 그리면 완곡한 2차함수가 아닌 울퉁불퉁한 2차함수가 만들어진다. 따라서 시작하는 위치에 따라서 최소값이 제각기 다르게 되어(local minimum) 우리가 찾고자 하는 진짜 최소값(global minimum)을 못구한다.

따라서 이번에도 Linear와 다른 새로운 cost function이 필요하다.

C(H(z), y) = y = 1일 때 -log(H(x)), y = 0 일 때 -log(1-H(x))

두가지 케이스로 나누어서 적용시킨다. 지수의 상극, 반대가 되는 것이 log기 때문에 log로 지수의 값이 커지는 것을 잡아준다.

Y = 1, H(x) = 1일 때, cost를 계산하면 -log(1) = 0
Y = 1, H(x) = 0일 때, cost를 계산하면 -log(0) = 무한대(log(x)의 x값이 무한히 작아지면 결과값은 무한히 작아지는데 앞에 -가 붙어 무한히 커지게 변한다)

Y = 0, H(x) = 1일 때, cost를 계산하면 -log(0) = 무한대
Y = 0, H(x) = 0일 때, cost를 계산하면 -log(1) = 0

그러나 이렇게 분기로 식을 두면 코딩을 할 때 복잡해지므로 식을 하나로 합쳤다. 

>c(H(x), y) = -ylog(H(x))-(1-y)log(1-H(x))

Minimize cost는 앞서 cost function을 매끄러운 2차함수로 만들었기 때문에 기존의 방법을 통해 최소화 시키면 된다.

