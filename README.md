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

### [ Lecture 5. Logistic (regression) classification ]

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

### [ Lecture 6. Softmax Regression ]

(이 부분을 학습하면서 내가 얼마나 얉게 알고 지나가는지 알게되었다. 그래도 완벽한 이해를 하며 넘어가기 보다 지금은 큰 그림을 보고 심화학습을 추후에 해야할거같단 생각이 들어 진도를 먼저 진행해보겠다)

x라는 입력을 W을 가지고 계산을 한 값을 z값인데 이것을 sigmod에 넣어 Y`값을 만드는데 Y`는 0~1까지의 값을 나타낸다.

Logistic regression은 결국 이분법이기때문에 두 값의 경계를 나타내는 선분의 기울기를 찾으면 된다. Multinomial classificatiion역시 마찬가지다 Y값이 여러개가 되면 각각의 Y값을 구분하는 경계선을 각각 찾아보는 것이다. 

이렇게 수식을 정리하면 행렬형태로 나오는데 Y값 하나당 수식 하나씩 나오는 것을 합쳐서 Y값 n개의 수식을 n, X값 m개를 행렬로 표기하면 

(n,m) * (m,1)의 형태로 나오게 된다. 이 값은 아직 sigmod가 되지 않아 1 이상의 수치가 나올 수 있어서 각각의 값을 Softmax regression을 활용해 수치를 0~1의 값을 갖게 조정한다.

Softmax 특징은 1. 항목에 대한 값이 0~1 사이의 값이 나오고, 2. 모든 항목의 값을 합치면 1이 되어야 한다. 사실 softmax는 점수형태로 나온 출력값을 확률형태로 변경해주는 것에 불과하다.

이렇게 0~1 사이의 값을 One-Hot Encoding 기법을 통해 가장 수치가 큰 값을 1로, 나머지 값을 0으로 변환하여 하나의 결과가 나오게 표기해준다.

cost function은 cross-entropy를 사용한다. 복잡해 보이지만 Logistic cost function( -ylog(H(x))-(1-y)log(1-H(x)) )이 사실상 cross-entropy였다.

다음으로는 Gradient descent를 활용해 cost를 최소화 시킨다. 

추가사항 

>cross entropy = D(S, L) = reduce_mean(reduce_sum(L * -log(S)))

Y의 값이 lable로 입력되는 것이 아니라 [[0,0,1], [1,0,0]...] 이런식으로 입력되면 one_hot 이라 한다. lable로 입력될 경우 reshape를 활용해서 차원을 한단계 낮추는 기술이 필요하다.(tf에서 제공하는 one_hot은 return값의 r이 +1 되어 나오기 때문에 -1을 해주기 위해 필요하다)

### [ Lecture 7-1. Application & Tips: Learning rate, data preprocessing, overfitting ]

세 방법 모두 오류를 줄이고 보다 바른 학습을 위해 사용하는 개념이다.

1. Learning rate  
우리가 cost function을 minimize하기 위해 사용하는 learning_rate는 너무 크면 overshooting이 일어난다. 내려가는 정도가 너무 강해 최소값으로 가지 못하고 그 근처를 왔다갔다하거나, 너무 크게 되면 밖으로 나가게 된다. 이것을 overshooting이라고 한다. 이럴 경우 값이 숫자가 나오지 않거나, 값이 점점 커지면 의심해봐야한다.

반대로 너무 작으면 for문을 다 돌아도 최소값에 도달하지 못하게 된다. 너무 작게 움직이기 때문이다. 따라서 처음에는 0.01을 기준으로 값이 발산하면 값을 줄이고, cost가 아직도 큰 값을 유지하는것 같다면 값을 키우는 등 유연하게 대처할 수 있다.

2. Data preprocessing(데이터 선처리)  
x의 data set 중 편차가 심한 인자가 포함되면 값의 편차가 너무 크게된다. 이는 후에 오차로 나타나게 되는데 이를 줄이기 위해 데이터를 선처리한다. 보통 zero-centered data를 시켜 영점을 잡고, normalized data를 통해 편차를 줄인다.(일정한 범위 안에서 다 표현이 되도록 변화시킨다)

[Nomalizetion 중 아래의 형태를 Standardizetion이라고 한다]
x_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()

3. Overfitting(가장 큰 문제가 됨)
학습 데이터에만 딱 맞는 모델이 나오게 되어 실제 데이터를 활용함에 있어 제약이 생기는 경우를 Overfitting이라고 한다.

[overfitting을 줄이는 방법]
- training data를 많이 확보한다
- features의 중복을 줄이고 개수를 줄인다
- Regularization을 활용한다

Regularization이란 model의 구분선이 곡선이 많을 수록 실 데이터에 사용하기에 문제가 있기 때문에 weight를 낮은 값으로 주어 곡선을 최소화 시킨다. 이것을 하기 위해선 cost function에 regularization_strength*tf.reduce_sum(tf.square(W)) 를 더한다. (regularization_strength값이 0에 가까울 수록 적은 변화를, 높을수록 큰 변화를 준다)

### [ Lecture 7-2. Application & Tips: Learning and test data sets ]

training set을 통해 학습한 모델에 다시 training set을 가지고 물어보면 100%의 답을 낼 것이다. 하지만 이것은 나쁜 방법이다. 모든 데이터를 경험했기 때문에 model은 100%의 답만 도출할 것이다. 따라서 전체 training set을 뒤에 1/3정도를 test set으로 사용하며 이 값은 training 하지 않는다. 그럼 약 2/3로 학습된 데이터를 통해 나온 model에 test set을 넣어 신뢰도를 측정할 수 있다. 

여기서 더 나아가면 위의 Training set의 뒤의 30%를 Validation set으로 활용한다. 이처럼 크게 둘로 나누는 방법과 셋으로 나누는 방법이 많이 사용하는 방법이다. 

Online learning은 training set이 너무 많을 경우 한번에 다 학습하는데 큰 자원과 시간이 소모된다. 따라서 이를 여러 등분해서 학습을 하면서 학습된 결과를 model에 남기는 방법을 Online learning이라고 한다. 매번 데이터가 추가될 때 마다 추가된 내용만 학습하면 되기 때문에 좋은 방법이다. 대표적인 예제가 MNIST Dataset이다.

이제 test set이 얼마나 맞았는지 확인하여 정확도를 산출하며 image쪽에서는 95~99%의 정확도가 유의미하게 사용된다.

**arg_max와 argmax의 차이**

>WARNING:tensorflow:From F:\PYTHON\tensorflow\lab7-3.py:24: arg_max (from tensorflow.python.ops.gen_math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `argmax` instead

arg_max함수를 사용하다보면 이런 경고문이 출력이 된다. 버전간 호환을 위해 기존에 사용하던 arg_max함수를 지우지 않고 유지하면서, argmax를 사용하길 권고하는 안내이자 경고문이다. 두 함수의 차이가 존재하는지 바꿔서 사용해도 경고문의 출력 여부만 다를뿐 결과에 대한 차이는 없다. 그러니 앞으로는 tf 개발자가 권고한대로 argmax를 사용하겠다.

### [ Lecture 8. Deep Neural Nets for Everyone ]

**Neural Network의 역사**
'복잡한 문제를 사람을 대신해서 기계가 해주면 얼마나 좋을까?'라는 생각에서 출발했다. 사람의 신경을 연구하던 학자들이 뉴런이 생각보다 단순하게 동작한다는 것을 알게 되고, 이는 하나의 수식으로 정리가 되었다.(Activation Functions)  
당시 And 나 Or의 구분을 할 수 있으면 이것을 조합해서 생각이 가능하다고 판단했다. 그래서 이런 논리사고를 하는 것을 중요하게 생각했는데 NN은 잘 해결했다. 그러나 XOR라는 새로운 문제에 부딧혀 NN분야는 침체기에 들어갔다. XOR은 이차원 상에서 선으로 구분을 지을 수 없기 때문에 아무리 해도 50%의 성공률을 넘지 못했다.(Linear의 한계)  
후에 MIT의 Minsky교수가 여러개의 NN을 사용하면 가능함을 이야기 하며서 동시에 각 NN의 W과 b를 학습시킬수 없다고 하며 기존의 NN의 한계를 규정했다.  
훗날 Backpropagation을 통해 XOR문제가 해결이 되었다. (Convolutional NN의 통해 해결하는 다른 방법도 있다. Alpha Go도 이 방법을 사용했다.)  
하지만 Backpropagation 역시 새로운 문제가 발견되는데 여러 NN layer를 지날수록 앞쪽으로 돌아오는 값의 의미가 약해져 layer의 depth가 깊을수록 문제해결능력이 떨어졌다. 이것이 두번째 침채기가 되어 NN은 다시 한번 사람들에게 외면받았다.

그 후 기존의 Backpropagation에서 초기값을 잘 주어서 애초에 값을 다시 reversing시키지 않거나 그 폭을 줄이는 방법을 택하면서 사람들의 관심을 끌기 위해 기존의 NN에서 deep learning이라고 명하게 되었다.  
가장 화제가 된 사건은 ImageNet이란 사이트에서 주어진 사진을 NN을 통해 설명하여 성능을 측정하는 과정에서 기존에 26.2%의 오답률을 15.3%로 끌어내는 사건이다. 2015년의 경우 사람이 판단한 정답률을 DL이 처음으로 넘어서기도 했다.(사람: 5%, DL: 4%)

Deep learning의 동의어: Neural Network의 발전된 형태.

### [ Lecture 9. NN for XOR ]

NN의 초기에 골치가 되었던 XOR을 어떻게 해결했을까?

단일 Unit으로는 절대 못푼다는 것이 수학적으로 증명도 되고 무수한 실패를 만들었다.

그래서 다수의 (최소 3개의) Unit을 통해 XOR을 풀었다.

y1 = sigmoid(x1*w1 + b1) 
y2 = sigmoid(x2*w2 + b2)

y` = sigmoid((y1, y2)*w3 + b3)

즉 두번의 sigmoid의 결과값을 다른 sigmoid의 인자로 넣어 계산한다.

위의 결과를 matrix를 통해 한번의 수식으로 계산할 수 있다.

>K(x) = sigmoid(X*W1 + b1)
>y` = sigmoid(K(x)*W2 + b2)

위와 같이 한줄의 코드로 치환이 가능하다.(이해가 안된 상태에서는 그냥 암기지만 이해가 되면 왜 저 수식이 나온지 감이 온다)

**[backpropagation]**
x1, x2... 수많은 x값 각각이 최종 결과 y`에 어떤 영향을 끼치는지 알기 위해선 앞서 수행했던 결과를 역으로 수행해보면 된다. 말이 쉽지 실제로는 그 과정 하나하나 되돌려야 하는데 엄청난 연산을 필요로 한다.

그래서 기존의 계산결과를 graph(tensor)로 저장해 둔것을 chain rule을 활용해서 x와 b의 값을 미분하면 이 값이 결과에 미치는 비율을 구할 수 있다. 이것을 활용하여 x의 값들, b의 값들을 조정할 수 있다. b의 미분값이 1이면 결과값 f에 대해 1:1이 된다. 즉 b가 1 증가하면 f도 1이 증가한다. 만약 b의 미분값이 5라면  b가 1 증가하면 f는 5가 증가할 것이다.

이렇게 결과값을 미분해가며 뒤로 돌아가면 최종적으로 x의 값들, b의 값들이 미치는 영향을 알 수 있다.

왜 고성능의 컴퓨터가 필요한지 여기서 조금 이해되었다. tensorflow에서는 결과값을 구하는 과정에서 발생하는 tensor를 저장해두고 후에 필요한 연산을을 위해 남겨둔다. 여기서 발생하는 메모리와 수많은 연산들, 컴퓨터 성능에 영향을 크게 받게 된다. 

### [ Special Lecture : 10분안에 미분 정리하기 ]

'f를 x로 미분한다'는 의미는 식으로 df/dx가 된다. x의 변화가 f에 미치는 영향이 어떤지 확인한다는 의미가 되기도 한다. 

f((x + a) - f(x)) / a (a는 델타 x라고 함) 이렇게 적기도 하며 델타x가 아주 작은 값일 때, 값의 순간변화율을 확인하는 것이 미분이라고 할 수 있다. 이는 기울기를 나타나게 되며 Gradient descent 알고리즘의 핵심이 된다. 

그럼 x, w, b의 값들이 의미하는 것은 무엇인가? 하면 결과값에 대한 비율이다. 미분해서 나온 b의 값이 1이라고 하면 결과값 'f에 대해 b가 1이 증가하면 f도 1이 증가한다'는 의미가 된다. 마찬가지로 x1의 값이 5라면 x가 1만큼 바뀌면 f가 5만큼 변한다는 것이다.

**[Basic Derivative]**

델타x == 0.01 일 때 :

f(x) = 3 == df/dx == (f(x + 0.01)-f(x)) / 0.01 == (3 - 3) / 0.01 == 0, 상수함수는 x가 없으므로 항상 상수값을 리턴하는데 이로 인해 분자의 값이 항상 0이 되어버린다. 그래서 상수를 미분하면 0이 된다.

f(x) = x == df/dx == 0.01 / 0.01 == 1, x의 값이 존재하면 x의 상수곱만 남게 되어 결과값이 된다. 같은 원리로 f(x) = 2x일 경우엔 2가 된다.

**[Partial Derivative]**

f(x,y) = xy, (af/ax) == y를 보면 파셜x의 값의 변화에 따른 파셜f의 값을 보고싶어하기 때문에 x의 상수값이 중요하다. x = 1이기 때문에 결국 f = 1이되고 x값 역시 x가 중요하고 y는 상수취급을 하게 되므로 y값만 남게 되므로 f(x,y) = xy, (af/ax) == 1이 되게 된다.

반대의 경우도 마찬가지.  
f(x,y) = xy, (af/ay)는 y가 영향력을 끼치므로 y의 상수곱, 1이 값이 된다.

f(x,y) = x+y, af/ax의 값은 x만 영향을 끼치므로 y값은 상수취급받아서 0, x는 상수곱의 값을 취한다. 즉 1+0, 1이 된다.

**[Chain rule]**

f(g(x)) == af/ax == (af/ag) * (ag/ax) 이 된다.  
식으로 적으면 어려운데 의미를 생각해보면 'g에 대한 x의 영향력과 f에 대한 g의 영향력'이라고 생각하면 조금 더 쉽게 와닿는다. 이렇게 두번에 걸쳐 구한 미분값을 서로 곱하면 f(g(x))의 값이 된다. 

### [ Special Lecture : Tensor Board ]

기존의 숫자를 사용하여 cost를 확인했는데, 시각화 시켜서 보기위해 tensor board를 사용한다.

**[사용순서]**

1. From TF graph, decide which tensors you want to log.(어떤 tensor를 log할지 정함)

	<code>
	foo_hist = tf.summary.histogram('foo', foo)
	
	cost = tf.summary.scalar('cost', cost)
	</code>

2. Merge all summaries.(한번에 summary 하기 위해 모두 합침)

	<code>
	summary = tf.summary.merge_all()
	</code>

3. Create writer and add graph.(기록 위치를 정하고 graph를 추가)

	<code>
	writer = tf.summary.FileWriter('./logs')

	writer.add_graph(sess.graph)
	</code>

4. Run summary merge and add_summary.(run시에 summary도 함께 해주며 기록해준다.)

	<code>
	s, _ = sess.run([summary, optimizer], feed_dic=blah)

	writer.add_summary(s, global_step=global_step)
	</code>

5. Launch TensorBoard.(실행한다.)

	> cmd에서 tensorboard --logdir=./logs   후 localhost:6006에 접속

## ~~(취업준비로 인해 당분간 잠정 중단)~~ 