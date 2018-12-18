<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L762)</span>
### Dense

```python
keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

많이 쓰는 바로 그 밀집 연결 NN 층.

`Dense`에서 구현하는 연산은
`output = activation(dot(input, kernel) + bias)`이다.
여기서 `activation`은 항목별 활성 함수이며
`activation` 인자로 전달한다. `kernel`은 층에서 생성한
가중치 행렬이고 `bias`는 층에서 생성한 편향 벡터이다
(`use_bias`가 `True`인 경우에만 해당).

참고: 층 입력의 랭크가 2보다 크면
`kernel`과의 도트곱 전에 편평하게 만든다.

__예시__


```python
# Sequential 모델의 첫 번째 층:
model = Sequential()
model.add(Dense(32, input_shape=(16,)))
# 그러면 모델이 (*, 16) 형태의 입력 배열을 받고
# (*, 32) 형태의 배열을 출력하게 됨

# 첫 번째 층 이후로는 입력의 크기를
# 지정해 주지 않아도 된다:
model.add(Dense(32))
```

__인자__

- __units__: 양의 정수. 출력 공간의 차원수.
- __activation__: 사용할 활성 함수
    ([활성](../activations.md) 참고).
    아무것도 지정하지 않으면 아무 활성도 적용하지 않음
    (즉 "선형" 활성: `a(x) = x`).
- __use_bias__: 불리언. 층에서 편향 벡터를 쓸지 여부.
- __kernel_initializer__: `kernel` 가중치 행렬의 initializer
    ([초기화](../initializers.md) 참고).
- __bias_initializer__: 편향 벡터의 initializer
    ([초기화](../initializers.md) 참고).
- __kernel_regularizer__: `kernel` 가중치 행렬에
    적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __bias_regularizer__: 편향 벡터에 적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __activity_regularizer__: 층의 출력에 ("활성"에)
    적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __kernel_constraint__: `kernel` 가중치 행렬에
    적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).

__입력 형태__

`(batch_size, ..., input_dim)` 형태의 n차원 텐서.
가장 흔한 경우는 `(batch_size, input_dim)` 형태의
2차원 입력일 것이다.

__출력 형태__

`(batch_size, ..., units)` 형태의 n차원 텐서.
예를 들어 `(batch_size, input_dim)` 형태의 2차원 입력에 대해
출력은 `(batch_size, units)` 형태가 될 것이다.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L272)</span>
### Activation

```python
keras.layers.Activation(activation)
```

출력에 활성 함수를 적용한다.

__인자__

- __activation__: 사용할 활성 함수 이름
    ([활성](../activations.md) 참고),
    또는 테아노나 오픈플로우의 연산.

__입력 형태__

마음대로. 이 층을 모델의 첫 번째 층으로 쓸 때는
키워드 인자 `input_shape`(정수들의 튜플, 표본 축은
제외)를 사용하라.

__출력 형태__

입력과 같은 형태.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L80)</span>
### Dropout

```python
keras.layers.Dropout(rate, noise_shape=None, seed=None)
```

입력에 드롭아웃을 적용한다.

드롭아웃은 훈련 동안 각 갱신마다
입력 유닛들 중 `rate` 부분만큼을 무작위로 0으로 설정하는 것이며
과적합을 막는 데 도움이 된다.

__인자__

- __rate__: 0과 1 사이의 float. 버릴 입력 유닛들의 비율.
- __noise_shape__: 입력에 곱하게 되는 이진 드롭아웃 마스크의
    형태를 나타내는 1차원 정수 텐서.
    예를 들어 입력의 형태가 `(batch_size, timesteps, features)`이고
    모든 timestep에 대해 드롭아웃 마스크가 같게 하고 싶다면
    `noise_shape=(batch_size, 1, features)`라고 할 수 있다.
- __seed__: 난수 시드로 사용할 파이썬 정수.

__참고 자료__

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L457)</span>
### Flatten

```python
keras.layers.Flatten(data_format=None)
```

입력을 편평하게 만든다. 배치 크기에는 영향을 주지 않는다.

__인자__

- __data_format__: 문자열.
    "channels_last"(기본값) 또는 "channels_first".
    입력에서 차원들의 순서.
    이 인자의 목적은 한 데이터 형식에서 다른 형식으로
    모델을 전환할 때 가중치 순서를 유지하는 것이다.
    "channels_last"는 `(batch, ..., channels)` 형태의
    입력에 해당하고 "channels_first"는
    `(batch, channels, ...)` 형태의 입력에 해당한다.
    지정하지 않으면 케라스 설정 파일 `~/.keras/keras.json`에
    있는 `image_data_format` 값을 쓴다.
    그 값을 설정한 적이 없으면 "channels_last"를 쓰게 된다.

__예시__


```python
model = Sequential()
model.add(Conv2D(64, (3, 3),
                 input_shape=(3, 32, 32), padding='same',))
# 그러면: model.output_shape == (None, 64, 32, 32)

model.add(Flatten())
# 그러면: model.output_shape == (None, 65536)
```

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/engine/input_layer.py#L113)</span>
### Input

```python
keras.engine.input_layer.Input()
```

`Input()`을 통해 케라스 텐서 인스턴스를 만든다.

케라스 텐서란 기반 백엔드(테아노, 텐서플로우, CNTK)에서
온 텐서 객체에 어떤 속성들을 추가한 것이다.
그 속성들 때문에 모델의 입력과 출력만 알면
케라스 모델을 구성할 수 있게 된다.

예를 들어 a, b, c가 케라스 텐서라고 하면
`model = Model(input=[a, b], output=c)`라고
하는 게 가능해진다.

추가되는 케라스 속성들은 다음과 같다.

- `_keras_shape`: 케라스 측 형태 추론을 통해
전파되는 정수 형태 튜플.
- `_keras_history`: 텐서에 적용된 마지막 층.
그 층을 가지고 재귀적으로 층 그래프 전체를
얻어 올 수 있다.

__인자__

- __shape__: 형태 튜플(정수). 배치 크기는 포함하지 않음.
    예를 들어 `shape=(32,)`는 예상 입력이
    32차원 벡터들의 배치들이라는 뜻이다.
- __batch_shape__: 형태 튜플(정수). 배치 크기 포함.
    예를 들어 `batch_shape=(10, 32)`는 예상 입력이
    32차원 벡터 10개짜리 배치들이라는 뜻이다.
    `batch_shape=(None, 32)`는 임의 개수의 32차원 벡터들로 된
    배치들을 뜻한다.
- __name__: 선택적. 층의 이름 문자열.
    모델 내에서 유일해야 한다. (같은 이름을 재사용하지 말 것.)
    제공해 주지 않으면 자동 생성된다.
- __dtype__: 문자열. 입력에서 기대하는 데이터 타입.
    (`float32`, `float64`, `int32`...)
- __sparse__: 불리언. 생성할 플레이스홀더가 희소해야 하는지
    여부를 나타냄.
- __tensor__: 선택적. `Input` 층으로 집어 넣을 기존 텐서.
    설정 시 층에서 플레이스홀더 텐서를 만들지 않게 된다.

__반환__

텐서.

__예시__


```python
# 케라스에서의 로지스틱 회귀
x = Input(shape=(32,))
y = Dense(16, activation='softmax')(x)
model = Model(x, y)
```

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L306)</span>
### Reshape

```python
keras.layers.Reshape(target_shape)
```

출력의 형태를 특정 형태로 바꾼다.

__인자__

- __target_shape__: 목표 형태. 정수들의 튜플.
    배치 축은 포함하지 않음.

__입력 형태__

마음대로. 다만 입력 형태의 모든 차원들이 정해져 있어야 한다.
이 층을 모델의 첫 번째 층으로 쓸 때는
키워드 인자 `input_shape`(정수들의 튜플, 표본 축은
제외)를 사용하라.

__출력 형태__

`(batch_size,) + target_shape`

__예시__


```python
# Sequential 모델의 첫 번째 층
model = Sequential()
model.add(Reshape((3, 4), input_shape=(12,)))
# 그러면: model.output_shape == (None, 3, 4)
# 참고: `None`은 배치 차원

# Sequential 모델의 중간 층
model.add(Reshape((6, 2)))
# 그러면: model.output_shape == (None, 6, 2)

# 차원에 `-1`을 사용해 형태 추론도 지원함
model.add(Reshape((-1, 2, 2)))
# 그러면: model.output_shape == (None, 3, 2, 2)
```

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L405)</span>
### Permute

```python
keras.layers.Permute(dims)
```

주어진 패턴에 따라 입력의 차원들을 치환한다.

가령 RNN과 convnet을 연결할 때 유용하다.

__예시__


```python
model = Sequential()
model.add(Permute((2, 1), input_shape=(10, 64)))
# 그러면: model.output_shape == (None, 64, 10)
# 참고: `None`은 배치 차원
```

__인자__

- __dims__: 정수들의 튜플. 치환 패턴이며, 표본 차원을
    포함하지 않음. 1부터 인덱스가 시작.
    예를 들어 `(2, 1)`은 입력의 첫 번째 차원과
    두 번째 차원을 치환한다.

__입력 형태__

마음대로. 이 층을 모델의 첫 번째 층으로 쓸 때는
키워드 인자 `input_shape`(정수들의 튜플, 표본 축은
제외)를 사용하라.

__출력 형태__

입력 형태와 같되 지정 패턴에 따라 차원들의 순서가
바뀌어 있다.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L519)</span>
### RepeatVector

```python
keras.layers.RepeatVector(n)
```

입력을 n 번 반복한다.

__예시__


```python
model = Sequential()
model.add(Dense(32, input_dim=32))
# 그러면: model.output_shape == (None, 32)
# 참고: `None`은 배치 차원

model.add(RepeatVector(3))
# 그러면: model.output_shape == (None, 3, 32)
```

__인자__

- __n__: 정수. 반복 인자.

__입력 형태__

`(num_samples, features)` 형태의 2차원 텐서.

__출력 형태__

`(num_samples, n, features)` 형태의 3차원 텐서.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L561)</span>
### Lambda

```python
keras.layers.Lambda(function, output_shape=None, mask=None, arguments=None)
```

임의의 식을 `Layer` 객체로 감싼다.

__예시__


```python
# x -> x^2 층 추가
model.add(Lambda(lambda x: x ** 2))
```
```python
# 입력의 양수인 부분에다 음수인
# 부분을 위로 뒤집어서 이어 붙인 걸
# 반환하는 층 추가

def antirectifier(x):
    x -= K.mean(x, axis=1, keepdims=True)
    x = K.l2_normalize(x, axis=1)
    pos = K.relu(x)
    neg = K.relu(-x)
    return K.concatenate([pos, neg], axis=1)

def antirectifier_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # 2차원 텐서만 가능
    shape[-1] *= 2
    return tuple(shape)

model.add(Lambda(antirectifier,
                 output_shape=antirectifier_output_shape))
```

__인자__

- __function__: 평가할 함수.
    첫 번째 인자로 입력 텐서를 받음.
- __output_shape__: 함수의 예상 출력 형태.
    테아노에서만 의미가 있음.
    튜플 또는 함수일 수 있음.
    튜플인 경우에는 첫 번째 이후 차원만을 지정한다.
         즉 표본 차원이 입력과 같다고 상정하면
         `output_shape = (input_shape[0], ) + output_shape`이고
         입력이 `None`이고 표본 차원도 `None`이면
         `output_shape = (None, ) + output_shape`이다.
    함수인 경우에는 입력 형태에 대한 함수 형태로
    전체 형태를 지정한다. 즉 `output_shape = f(input_shape)`이다.
- __arguments__: 선택적. 함수로 전달할 키워드 인자들의 딕셔너리.

__입력 형태__

마음대로. 이 층을 모델의 첫 번째 층으로 쓸 때는
키워드 인자 `input_shape`(정수들의 튜플, 표본 축은
제외)를 사용하라.

__출력 형태__

`output_shape` 인자로 지정.
(또는 텐서플로우나 CNTK 사용 시 자동으로 추론.)

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L905)</span>
### ActivityRegularization

```python
keras.layers.ActivityRegularization(l1=0.0, l2=0.0)
```

입력 activity에 따라 비용 함수에 갱신을 적용하는 층.

__인자__

- __l1__: L1 정칙화 인자 (양수 float).
- __l2__: L2 정칙화 인자 (양수 float).

__입력 형태__

마음대로. 이 층을 모델의 첫 번째 층으로 쓸 때는
키워드 인자 `input_shape`(정수들의 튜플, 표본 축은
제외)를 사용하라.

__출력 형태__

입력과 같은 형태.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L29)</span>
### Masking

```python
keras.layers.Masking(mask_value=0.0)
```

열을 어떤 마스크 값으로 감춰서 timestep을 건너뛰게 한다.

입력 텐서의 각 timestep(텐서의 차원 #1)에 대해서
그 timestep에서 입력 텐서의 모든 값이 `mask_value`와
같으면 이후의 모든 층에서 (마스크를 지원한다고 하면)
그 timestep을 감추게 (건너뛰게) 된다.

이후의 어느 층에서 마스크를 지원하지 않는데 그런
입력 마스크를 받게 되면 예외를 던지게 된다.

__예시__


`(samples, timesteps, features)` 형태의 Numpy 데이터 배열 `x`를
LSTM 층에 넣는다고 하자.
timestep #3과 #5에 대한 데이터가 없어서 그 timestep들을
감추려 한다. 그러면 다음과 같이 하면 된다.

- `x[:, 3, :] = 0.` 및 `x[:, 5, :] = 0.` 설정
- LSTM 층 앞에 `mask_value=0.`으로 해서 `Masking` 층 삽입

```python
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
model.add(LSTM(32))
```

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L139)</span>
### SpatialDropout1D

```python
keras.layers.SpatialDropout1D(rate)
```

드롭아웃의 1차원 공간 버전.

이 버전은 드롭아웃과 같은 기능을 수행하되 개별 요소가
아니라 1차원 피쳐 맵 전체를 버린다. 피쳐 맵 내의 이웃한
프레임들이 강하게 연관돼 있는 경우에는 (앞쪽 합성곱 층에서는
보통 그렇다.) 기본 드롭아웃이 활성을 정칙화 하지 못하거나
그렇지는 않더라도 실질 학습률의 저하가 발생하게 된다.
그런 경우에는 피쳐 맵들 간 독립성 강화에 도움이 되는
SpatialDropout1D를 대신 사용하는 게 좋다.

__인자__

- __rate__: 0과 1 사이의 float. 버릴 입력 유닛들의 비율.

__입력 형태__

`(samples, timesteps, channels)` 형태의 3차원 텐서.

__출력 형태__

입력과 동일

__참고 자료__

- [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/abs/1411.4280)

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L175)</span>
### SpatialDropout2D

```python
keras.layers.SpatialDropout2D(rate, data_format=None)
```

드롭아웃의 2차원 공간 버전.

이 버전은 드롭아웃과 같은 기능을 수행하되 개별 요소가
아니라 2차원 피쳐 맵 전체를 버린다. 피쳐 맵 내의 이웃한
픽셀들이 강하게 연관돼 있는 경우에는 (앞쪽 합성곱 층에서는
보통 그렇다.) 기본 드롭아웃이 활성을 정칙화 하지 못하거나
그렇지는 않더라도 실질 학습률의 저하가 발생하게 된다.
그런 경우에는 피쳐 맵들 간 독립성 강화에 도움이 되는
SpatialDropout2D를 대신 사용하는 게 좋다.

__인자__

- __rate__: 0과 1 사이의 float. 버릴 입력 유닛들의 비율.
- __data_format__: "channels_first" 또는 "channels_last".
    "channels_first" 모드에서는 채널 차원(깊이)이 인덱스 1에 있고
    "channels_last" 모드에서는 인덱스 3에 있다.
    지정하지 않으면 케라스 설정 파일 `~/.keras/keras.json`에
    있는 `image_data_format` 값을 쓴다.
    그 값을 설정한 적이 없으면 "channels_last"를 쓰게 된다.

__입력 형태__

data_format='channels_first'이면
`(samples, channels, rows, cols)` 형태의 4차원 텐서.
data_format='channels_last'이면
`(samples, rows, cols, channels)` 형태의 4차원 텐서.

__출력 형태__

입력과 동일

__참고 자료__

- [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/abs/1411.4280)

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L224)</span>
### SpatialDropout3D

```python
keras.layers.SpatialDropout3D(rate, data_format=None)
```

드롭아웃의 3차원 공간 버전.

이 버전은 드롭아웃과 같은 기능을 수행하되 개별 요소가
아니라 3차원 피쳐 맵 전체를 버린다. 피쳐 맵 내의 이웃한
복셀들이 강하게 연관돼 있는 경우에는 (앞쪽 합성곱 층에서는
보통 그렇다.) 기본 드롭아웃이 활성을 정칙화 하지 못하거나
그렇지는 않더라도 실질 학습률의 저하가 발생하게 된다.
그런 경우에는 피쳐 맵들 간 독립성 강화에 도움이 되는
SpatialDropout3D를 대신 사용하는 게 좋다.

__인자__

- __rate__: 0과 1 사이의 float. 버릴 입력 유닛들의 비율.
- __data_format__: "channels_first" 또는 "channels_last".
    "channels_first" 모드에서는 채널 차원(깊이)이 인덱스 1에 있고
    "channels_last" 모드에서는 인덱스 4에 있다.
    지정하지 않으면 케라스 설정 파일 `~/.keras/keras.json`에
    있는 `image_data_format` 값을 쓴다.
    그 값을 설정한 적이 없으면 "channels_last"를 쓰게 된다.

__입력 형태__

data_format='channels_first'이면
`(samples, channels, dim1, dim2, dim3)` 형태의 5차원 텐서.
data_format='channels_last'이면
`(samples, dim1, dim2, dim3, channels)` 형태의 5차원 텐서.

__출력 형태__

입력과 동일

__참고 자료__

- [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/abs/1411.4280)
