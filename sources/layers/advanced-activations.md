<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L18)</span>
### LeakyReLU

```python
keras.layers.LeakyReLU(alpha=0.3)
```

정류 선형 단위(Rectified Linear Unit)의 누출 버전.

유닛이 활성이 아닐 때 작은 경사를 허용한다.
x < 0이면 `f(x) = alpha * x`,
x >= 0이면 `f(x) = x`.

__입력 형태__

마음대로. 이 층을 모델의 첫 번째 층으로 쓸 때는
키워드 인자 `input_shape`(정수들의 튜플, 표본 축은
제외)를 써야 한다.

__출력 형태__

입력과 같은 형태.

__인자__

- __alpha__: float >= 0. 음수 경사 계수.

__참고 자료__

- [Rectifier Nonlinearities Improve Neural Network Acoustic Models](https://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L57)</span>
### PReLU

```python
keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
```

매개변수형 정류 선형 단위(Rectified Linear Unit).

x < 0이면 `f(x) = alpha * x`,
x >= 0이면 `f(x) = x`.
`alpha`는 x와 형태가 같은 학습되는 배열.

__입력 형태__

마음대로. 이 층을 모델의 첫 번째 층으로 쓸 때는
키워드 인자 `input_shape`(정수들의 튜플, 표본 축은
제외)를 써야 한다.

__출력 형태__

입력과 같은 형태.

__인자__

- __alpha_initializer__: 가중치 초기화 함수.
- __alpha_regularizer__: 가중치 정칙화.
- __alpha_constraint__: 가중치 제약.
- __shared_axes__: 활성 함수를 위한 학습 가능
    매개변수들을 공유할 축.
    예를 들어 들어오는 피쳐 맵이 출력 형태가
    `(batch, height, width, channels)`인
    2차원 합성곱에서 온 것인데 매개변수들을
    공간 상에서 공유해서 각 필터마다
    매개변수 세트가 하나씩 있게 하고 싶다면
    `shared_axes=[1, 2]`라고 설정하면 된다.

__참고 자료__

- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L152)</span>
### ELU

```python
keras.layers.ELU(alpha=1.0)
```

지수 선형 단위(Exponential Linear Unit).

x < 0이면 `f(x) = alpha * (exp(x) - 1.)`,
x >= 0이면 `f(x) = x`.

__입력 형태__

마음대로. 이 층을 모델의 첫 번째 층으로 쓸 때는
키워드 인자 `input_shape`(정수들의 튜플, 표본 축은
제외)를 써야 한다.

__출력 형태__

입력과 같은 형태.

__인자__

- __alpha__: 음수 스케일 인자.

__참고 자료__

- [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289v1)

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L191)</span>
### ThresholdedReLU

```python
keras.layers.ThresholdedReLU(theta=1.0)
```

역치 정류 선형 단위(Rectified Linear Unit).

x > theta이면 `f(x) = x`,
아니면 `f(x) = 0`.

__입력 형태__

마음대로. 이 층을 모델의 첫 번째 층으로 쓸 때는
키워드 인자 `input_shape`(정수들의 튜플, 표본 축은
제외)를 써야 한다.

__출력 형태__

입력과 같은 형태.

__인자__

- __theta__: float >= 0. 활성의 역치 위치.

__참고 자료__

- [Zero-Bias Autoencoders and the Benefits of Co-Adapting Features](http://arxiv.org/abs/1402.3337)

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L230)</span>
### Softmax

```python
keras.layers.Softmax(axis=-1)
```

소프트맥스 활성 함수.

__입력 형태__

마음대로. 이 층을 모델의 첫 번째 층으로 쓸 때는
키워드 인자 `input_shape`(정수들의 튜플, 표본 축은
제외)를 써야 한다.

__출력 형태__

입력과 같은 형태.

__인자__

- __axis__: int. 소프트맥스 정규화를 적용하는 축.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L262)</span>
### ReLU

```python
keras.layers.ReLU(max_value=None)
```

정류 선형 단위(Rectified Linear Unit) 활성 함수.

__입력 형태__

마음대로. 이 층을 모델의 첫 번째 층으로 쓸 때는
키워드 인자 `input_shape`(정수들의 튜플, 표본 축은
제외)를 써야 한다.

__출력 형태__

입력과 같은 형태.

__인자__

- __max_value__: float. 최대 출력 값.
