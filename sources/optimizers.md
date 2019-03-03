
## 최적화 사용

최적화 방식은 케라스 모델 컴파일에 꼭 필요한 두 인자 중 하나이다.

```python
from keras import optimizers

model = Sequential()
model.add(Dense(64, kernel_initializer='uniform', input_shape=(10,)))
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
```

위 예처럼 최적화 인스턴스를 만들어서 `model.compile()`로 전달할 수도 있고 그냥 이름으로 호출할 수도 있다. 후자에서는 그 최적화의 기본 매개변수들을 쓰게 된다.

```python
# 이름으로 최적화 주기: 기본 매개변수 사용
model.compile(loss='mean_squared_error', optimizer='sgd')
```

---

## 모든 케라스 최적화에 공통인 매개변수

매개변수 `clipnorm`과 `clipvalue`는 모든 최적화에서 경사 클리핑을 제어하는 데 쓸 수 있다.

```python
from keras import optimizers

# 모든 매개변수 경사를 최대 norm 1로 자른다.
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
```

```python
from keras import optimizers

# 모든 매개변수 경사를 최댓값 0.5와
# 최솟값 -0.5로 자른다.
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
```

---

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L157)</span>
### SGD

```python
keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
```

확률적 경사 하강(stochastic gradient descent) 최적화.

모멘텀, 학습률 감쇄, 네스테로프 모멘텀 지원 포함.

__인자__

- __lr__: float >= 0. 학습률.
- __momentum__: float >= 0. 적절한 방향으로 SGD를 가속시키고
    진동을 감쇠시키는 매개변수.
- __decay__: float >= 0. 각 갱신마다의 학습률 감쇄.
- __nesterov__: bool. 네스테로프 모멘텀 적용 여부.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L220)</span>
### RMSprop

```python
keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
```

RMSProp 최적화.

이 최적화의 매개변수들을 기본값 그대로
두기를 권한다.
(단 학습률은 마음대로 조정할 수 있다.)

이 최적화는 일반적으로 순환 신경망에 좋은 선택지이다.

__인자__

- __lr__: float >= 0. 학습률.
- __rho__: float >= 0.
- __epsilon__: float >= 0. 퍼징 인자. `None`이면 `K.epsilon()` 사용.
- __decay__: float >= 0. 각 갱신마다의 학습률 감쇄.

__참고 자료__

- [rmsprop: Divide the gradient by a running average of its recent magnitude](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L287)</span>
### Adagrad

```python
keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
```

Adagrad 최적화.

Adagrad는 매개변수별로 학습률이 있는 최적화로,
훈련 중 매개변수가 얼마나 자주 갱신되는가에 따라
학습률을 조정한다. 매개변수가 많이 갱신될수록
갱신 폭이 작아진다.

이 최적화의 매개변수들을 기본값 그대로
두기를 권한다.

__인자__

- __lr__: float >= 0. 초기 학습률.
- __epsilon__: float >= 0. `None`이면 `K.epsilon()` 사용.
- __decay__: float >= 0. 각 갱신마다의 학습률 감쇄.

__참고 자료__

- [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L351)</span>
### Adadelta

```python
keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
```

Adadelta 최적화.

Adadelta는 Adagrad를 더 견고하게 확장한 것으로,
모든 과거 경사를 누적하는 대신 경사 갱신의 이동 윈도에 기반해서
학습률을 조정한다. 그렇게 해서 Adadelta는 갱신이 많이 이뤄진
상태에서도 계속 학습을 할 수 있다. Adagrad에서와 달리
Adadelta 원래 버전에서는 초기 학습률을 설정할 필요가 없다.
이 버전에서는 다른 대부분의 케라스 최적화처럼
초기 학습률과 감쇄 인자를 설정할 수 있다.

이 최적화의 매개변수들을 기본값 그대로
두기를 권한다.

__인자__

- __lr__: float >= 0. 초기 학습률, 기본값은 1.
    기본값 그대로 두기를 권한다.
- __rho__: float >= 0. Adadelta 감쇄 인자. 각 time step에서
    경사를 어느 비율만큼 남겨 둘 것인지에 해당.
- __epsilon__: float >= 0. 퍼징 인자. `None`이면 `K.epsilon()` 사용.
- __decay__: float >= 0. 초기 학습률 감쇄.

__참고 자료__

- [Adadelta - an adaptive learning rate method](http://arxiv.org/abs/1212.5701)

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L433)</span>
### Adam

```python
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
```

Adam 최적화.

기본 매개변수 값들은 원저 논문에 제시된 값을 따른다.

__인자__

- __lr__: float >= 0. 학습률.
- __beta_1__: float, 0 < beta < 1. 일반적으로 1에 가까움.
- __beta_2__: float, 0 < beta < 1. 일반적으로 1에 가까움.
- __epsilon__: float >= 0. 퍼징 인자. `None`이면 `K.epsilon()` 사용.
- __decay__: float >= 0. 각 갱신마다의 학습률 감쇄.
- __amsgrad__: bool. 논문 "On the Convergence of Adam and
    Beyond"에 있는 이 알고리듬의 AMSGrad 버전 적용 여부.

__참고 자료__

- [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
- [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L522)</span>
### Adamax

```python
keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
```

Adam 논문 7절에서 온 Adamax 최적화.

무한 norm을 기반으로 한 Adam의 변형 버전이다.
기본 매개변수 값들은 논문에 제시된 값을 따른다.

__인자__

- __lr__: float >= 0. 학습률.
- __beta_1/beta_2__: floats, 0 < beta < 1. 일반적으로 1에 가까움.
- __epsilon__: float >= 0. 퍼징 인자. `None`이면 `K.epsilon()` 사용.
- __decay__: float >= 0. 각 갱신마다의 학습률 감쇄.

__참고 자료__

- [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L599)</span>
### Nadam

```python
keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
```

네스테로프(Nesterov) Adam 최적화.

Adam의 핵심이 RMSprop에 모멘텀을 더한 거라면
Nadam은 Adam RMSprop에 네스테로프 모멘텀을 더한 것이다.

기본 매개변수 값들은 논문에 제시된 값을 따른다.
이 최적화의 매개변수들을 기본값 그대로
두기를 권한다.

__인자__

- __lr__: float >= 0. 학습률.
- __beta_1/beta_2__: floats, 0 < beta < 1. 일반적으로 1에 가까움.
- __epsilon__: float >= 0. 퍼징 인자. `None`이면 `K.epsilon()` 사용.

__참고 자료__

- [Nadam 보고서](http://cs229.stanford.edu/proj2015/054_report.pdf)
- [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)

