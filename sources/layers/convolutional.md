<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L237)</span>
### Conv1D

```python
keras.layers.Conv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

1차원 합성곱 층 (가령 시간 합성곱).

이 층은 합성곱 커널을 만들어서
단일 공간 (또는 시간) 차원 상에서
층 입력과 곱하여 출력 텐서를 만들어 낸다.
`use_bias`가 True이면 편향 벡터를 만들어서 출력에 더한다.
그리고 `activation`이 `None`이 아니면
마찬가지로 출력에 적용한다.

이 층을 모델의 첫 번째 층으로 쓸 때는
`input_shape` 인자(정수나 `None`으로 된 튜플)를
주어야 한다. 가령
128차원 벡터 10개짜리 열에는 `(10, 128)`,
128차원 벡터들의 가변 길이 열에는 `(None, 128)`.

__인자__

- __filters__: 정수. 출력 공간의 차원수.
    (즉 합성곱에서 출력 필터의 수)
- __kernel_size__: 정수 또는 단일 정수로 된 튜플/리스트.
    1차원 합성곱 윈도의 길이를 지정.
- __strides__: 정수 또는 단일 정수로 된 튜플/리스트.
    합성곱의 보폭을 지정.
    1 아닌 보폭 값을 지정하는 것과 1 아닌
    `dilation_rate` 값을 지정하는 것은 호환되지 않음.
- __padding__: `"valid"`, `"causal"`, `"same"` 중 하나 (대소문자 구분 없음).
    `"valid"`는 "패딩 없음"을 뜻함.
    `"same"`은 출력이 원래 입력과 같은 길이가 되도록
    입력에 패딩을 더하게 함.
    `"causal"`은 인과적 (팽창) 합성곱으로, 가령 output[t]가
    input[t+1:]에 의존하지 않음. 시간적 데이터를 모델링 하는데
    모델이 시간 순서를 위반하지 말아야 할 때 유용함.
    [WaveNet: A Generative Model for Raw Audio, 2.1절](https://arxiv.org/abs/1609.03499) 참고.
- __data_format__: 문자열.
    `"channels_last"`(기본값) 또는 `"channels_first"`.
    입력에서 차원들의 순서.
    `"channels_last"`는 `(batch, steps, channels)`
    형태(시간 데이터에 대한 케라스 기본 형식)의 입력에 해당하고
    `"channels_first"`는 `(batch, channels, steps)`
    형태의 입력에 해당한다.
- __dilation_rate__: 정수 또는 단일 정수로 된 튜플/리스트.
    팽창 합성곱에 사용할 팽창 비율을 지정.
    1 아닌 `dilation_rate` 값을 지정하는 것과
    1 아닌 `strides` 값을 지정하는 것은 현재 호환되지 않음.
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
- __kernel_constraint__: kernel 행렬에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).

__입력 형태__

`(batch, steps, channels)` 형태의 3차원 텐서.

__출력 형태__

3D tensor with shape: `(batch, new_steps, filters)` 형태의 3차원 텐서.

`steps` 값이 패딩이나 보폭 때문에 바뀌었을 수도 있다.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L359)</span>
### Conv2D

```python
keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

2차원 합성곱 층 (가령 이미지에 대한 공간 합성곱).

이 층은 합성곱 커널을 만들어서
층 입력과 곱하여 출력 텐서를 만들어 낸다.
`use_bias`가 True이면 편향 벡터를 만들어서 출력에 더한다.
그리고 `activation`이 `None`이 아니면
마찬가지로 출력에 적용한다.

이 층을 모델의 첫 번째 층으로 쓸 때는
키워드 인자 `input_shape`(정수 튜플,
표본 축은 제외)를 주어야 한다.
가령 `data_format="channels_last"`일 때
128x128짜리 RGB 그림에는 `input_shape=(128, 128, 3)`.

__인자__

- __filters__: 정수. 출력 공간의 차원수.
    (즉 합성곱에서 출력 필터의 수)
- __kernel_size__: 정수 또는 2개 정수로 된 튜플/리스트.
    2차원 합성곱 윈도의 높이와 너비를 지정.
    단일 정수를 사용해 모든 공간 차원에
    같은 값을 지정할 수 있음.
- __strides__: 정수 또는 2개 정수로 된 튜플/리스트.
    합성곱의 높이 및 너비 방향 보폭을 지정.
    단일 정수를 사용해 모든 공간 차원에
    같은 값을 지정할 수 있음.
    1 아닌 보폭 값을 지정하는 것과 1 아닌
    `dilation_rate` 값을 지정하는 것은 호환되지 않음.
- __padding__: `"valid"` 또는 `"same"` (대소문자 구분 없음).
    참고로 [여기](https://github.com/keras-team/keras/pull/9473#issuecomment-372166860)
    설명처럼 `strides` != 1일 때 백엔드에 따라
    `"same"`이 살짝 다르다.
- __data_format__: 문자열.
    `"channels_last"` 또는 `"channels_first"`.
    입력에서 차원들의 순서.
    `"channels_last"`는 `(batch, height, width, channels)`
    형태의 입력에 해당하고
    `"channels_first"`는 `(batch, channels, height, width)`
    형태의 입력에 해당한다.
    지정하지 않으면 케라스 설정 파일 `~/.keras/keras.json`에
    있는 `image_data_format` 값을 쓴다.
    그 값을 설정한 적이 없으면 "channels_last"를 쓰게 된다.
- __dilation_rate__: 정수 또는 2개 정수로 된 튜플/리스트.
    팽창 합성곱에 사용할 팽창 비율을 지정.
    단일 정수를 사용해 모든 공간 차원에
    같은 값을 지정할 수 있음.
    1 아닌 `dilation_rate` 값을 지정하는 것과
    1 아닌 `strides` 값을 지정하는 것은 현재 호환되지 않음.
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
- __kernel_constraint__: kernel 행렬에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).

__입력 형태__

- `data_format`이 `"channels_first"`이면
`(batch, channels, rows, cols)`
형태의 4차원 텐서.
- `data_format`이 `"channels_last"`이면
`(batch, rows, cols, channels)`
형태의 4차원 텐서.

__출력 형태__

- `data_format`이 `"channels_first"`이면
`(batch, filters, new_rows, new_cols)`
형태의 4차원 텐서.
- `data_format`이 `"channels_last"`이면
`(batch, new_rows, new_cols, filters)`
형태의 4차원 텐서.

`rows` 및 `cols`  값이 패딩 때문에 바뀌었을 수도 있다.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1386)</span>
### SeparableConv1D

```python
keras.layers.SeparableConv1D(filters, kernel_size, strides=1, padding='valid', data_format=None, dilation_rate=1, depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None)
```

깊이 분리식 1차원 합성곱.

분리식 합성곱에서는 먼저 (입력 채널 각각에 따로
이뤄지는) 깊이 방향 공간 합성곱을 수행한 다음
나온 출력 채널들을 뒤섞는 점 합성곱을 수행한다.
`depth_multiplier` 인자가 깊이 방향 단계의
입력 채널당 몇 개씩의 출력 채널이 생성되는지를 결정한다.

분리식 합성곱을 간단하게는 합성곱 커널을 두 개의
작은 커널로 분해하는 것으로 생각할 수 있다.
또는 Inception 블록의 극단적인 버전으로 생각할 수 있다.

__인자__

- __filters__: 정수. 출력 공간의 차원수.
    (즉 합성곱에서 출력 필터의 수)
- __kernel_size__: 정수 또는 단일 정수로 된 튜플/리스트.
    1차원 합성곱 윈도의 길이를 지정.
- __strides__: 정수 또는 단일 정수로 된 튜플/리스트.
    합성곱의 보폭을 지정.
    1 아닌 보폭 값을 지정하는 것과 1 아닌
    `dilation_rate` 값을 지정하는 것은 호환되지 않음.
- __padding__: `"valid"` 또는 `"same"` (대소문자 구분 없음).
- __data_format__: 문자열.
    `"channels_last"` 또는 `"channels_first"`.
    입력에서 차원들의 순서.
    `"channels_last"`는 `(batch, steps, channels)`
    형태의 입력에 해당하고
    `"channels_first"`는 `(batch, channels, steps)`
    형태의 입력에 해당한다.
    지정하지 않으면 케라스 설정 파일 `~/.keras/keras.json`에
    있는 `image_data_format` 값을 쓴다.
    그 값을 설정한 적이 없으면 "channels_last"를 쓰게 된다.
- __dilation_rate__: 정수 또는 단일 정수로 된 튜플/리스트.
    팽창 합성곱에 사용할 팽창 비율을 지정.
    1 아닌 `dilation_rate` 값을 지정하는 것과
    1 아닌 `strides` 값을 지정하는 것은 현재 호환되지 않음.
- __depth_multiplier__: 각 입력 채널당 깊이 합성곱
    출력 채널의 수.
    깊이 합성곱 출력 채널의 총수는
    `filters_in * depth_multiplier`가 된다.
- __activation__: 사용할 활성 함수
    ([활성](../activations.md) 참고).
    아무것도 지정하지 않으면 아무 활성도 적용하지 않음
    (즉 "선형" 활성: `a(x) = x`).
- __use_bias__: 불리언. 층에서 편향 벡터를 쓸지 여부.
- __depthwise_initializer__: 깊이 커널 행렬 initializer
    ([초기화](../initializers.md) 참고).
- __pointwise_initializer__: 점 커널 행렬 initializer
    ([초기화](../initializers.md) 참고).
- __bias_initializer__: 편향 벡터의 initializer
    ([초기화](../initializers.md) 참고).
- __depthwise_regularizer__: 깊이 커널 행렬에
    적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __pointwise_regularizer__: 점 커널 행렬에
    적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __bias_regularizer__: 편향 벡터에 적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __activity_regularizer__: 층의 출력에 ("활성"에)
    적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __depthwise_constraint__: 깊이 커널 행렬에
    적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __pointwise_constraint__: 점 커널 행렬에
    적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).

__입력 형태__

- `data_format`이 `"channels_first"`이면
`(batch, channels, steps)`
형태의 3차원 텐서.
- `data_format`이 `"channels_last"`이면
`(batch, steps, channels)`
형태의 3차원 텐서.

__출력 형태__

- `data_format`이 `"channels_first"`이면
`(batch, filters, new_steps)`
형태의 3차원 텐서.
- `data_format`이 `"channels_last"`이면
`(batch, new_steps, filters)`
형태의 3차원 텐서.

`new_steps`  값이 패딩이나 보폭 때문에 바뀌었을 수도 있다.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1521)</span>
### SeparableConv2D

```python
keras.layers.SeparableConv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None)
```

깊이 분리식 2차원 합성곱.

분리식 합성곱에서는 먼저 (입력 채널 각각에 따로
이뤄지는) 깊이 방향 공간 합성곱을 수행한 다음
나온 출력 채널들을 뒤섞는 점 합성곱을 수행한다.
`depth_multiplier` 인자가 깊이 방향 단계의
입력 채널당 몇 개씩의 출력 채널이 생성되는지를 결정한다.

분리식 합성곱을 간단하게는 합성곱 커널을 두 개의
작은 커널로 분해하는 것으로 생각할 수 있다.
또는 Inception 블록의 극단적인 버전으로 생각할 수 있다.

__인자__

- __filters__: 정수. 출력 공간의 차원수.
    (즉 합성곱에서 출력 필터의 수)
- __kernel_size__: 정수 또는 2개 정수로 된 튜플/리스트.
    2차원 합성곱 윈도의 높이와 너비를 지정.
    단일 정수를 사용해 모든 공간 차원에
    같은 값을 지정할 수 있음.
- __strides__: 정수 또는 2개 정수로 된 튜플/리스트.
    합성곱의 높이 및 너비 방향 보폭을 지정.
    단일 정수를 사용해 모든 공간 차원에
    같은 값을 지정할 수 있음.
    1 아닌 보폭 값을 지정하는 것과 1 아닌
    `dilation_rate` 값을 지정하는 것은 호환되지 않음.
- __padding__: `"valid"` 또는 `"same"` (대소문자 구분 없음).
- __data_format__: 문자열.
    `"channels_last"` 또는 `"channels_first"`.
    입력에서 차원들의 순서.
    `"channels_last"`는 `(batch, height, width, channels)`
    형태의 입력에 해당하고
    `"channels_first"`는 `(batch, channels, height, width)`
    형태의 입력에 해당한다.
    지정하지 않으면 케라스 설정 파일 `~/.keras/keras.json`에
    있는 `image_data_format` 값을 쓴다.
    그 값을 설정한 적이 없으면 "channels_last"를 쓰게 된다.
- __dilation_rate__: 정수 또는 2개 정수로 된 튜플/리스트.
    팽창 합성곱에 사용할 팽창 비율을 지정.
    1 아닌 `dilation_rate` 값을 지정하는 것과
    1 아닌 `strides` 값을 지정하는 것은 현재 호환되지 않음.
- __depth_multiplier__: 각 입력 채널당 깊이 합성곱
    출력 채널의 수.
    깊이 합성곱 출력 채널의 총수는
    `filters_in * depth_multiplier`가 된다.
- __activation__: 사용할 활성 함수
    ([활성](../activations.md) 참고).
    아무것도 지정하지 않으면 아무 활성도 적용하지 않음
    (즉 "선형" 활성: `a(x) = x`).
- __use_bias__: 불리언. 층에서 편향 벡터를 쓸지 여부.
- __depthwise_initializer__: 깊이 커널 행렬 initializer
    ([초기화](../initializers.md) 참고).
- __pointwise_initializer__: 점 커널 행렬 initializer
    ([초기화](../initializers.md) 참고).
- __bias_initializer__: 편향 벡터의 initializer
    ([초기화](../initializers.md) 참고).
- __depthwise_regularizer__: 깊이 커널 행렬에
    적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __pointwise_regularizer__: 점 커널 행렬에
    적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __bias_regularizer__: 편향 벡터에 적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __activity_regularizer__: 층의 출력에 ("활성"에)
    적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __depthwise_constraint__: 깊이 커널 행렬에
    적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __pointwise_constraint__: 점 커널 행렬에
    적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).

__입력 형태__

- `data_format`이 `"channels_first"`이면
`(batch, channels, rows, cols)`
형태의 4차원 텐서.
- `data_format`이 `"channels_last"`이면
`(batch, rows, cols, channels)`
형태의 4차원 텐서.

__출력 형태__

- `data_format`이 `"channels_first"`이면
`(batch, filters, new_rows, new_cols)`
형태의 4차원 텐서.
- `data_format`이 `"channels_last"`이면
`(batch, new_rows, new_cols, filters)`
형태의 4차원 텐서.

`rows` 및 `cols`  값이 패딩 때문에 바뀌었을 수도 있다.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L618)</span>
### Conv2DTranspose

```python
keras.layers.Conv2DTranspose(filters, kernel_size, strides=(1, 1), padding='valid', output_padding=None, data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

전치 합성곱 층. (역합성곱이라고도 한다.)

전치 합성곱이 필요한 경우는 일반적으로
보통 합성곱의 반대 방향으로 가는 변형을 쓰고 싶을 때이다.
즉 어떤 합성곱의 출력 형태인 뭔가를
그 입력 형태인 뭔가로 변형하면서도
그 합성곱과 어울릴 수 있는 연결 패턴을 유지하고 싶을 때이다.

이 층을 모델의 첫 번째 층으로 쓸 때는
키워드 인자 `input_shape`(정수들의 튜플,
표본 축은 제외)를 주어야 한다.
가령 `data_format="channels_last"`인
128x128 RGB 그림이라면 `input_shape=(128, 128, 3)`.

__인자__

- __filters__: 정수. 출력 공간의 차원수.
    (즉 합성곱에서 출력 필터의 수)
- __kernel_size__: 정수 또는 2개 정수로 된 튜플/리스트.
    2차원 합성곱 윈도의 높이와 너비를 지정.
    단일 정수를 사용해 모든 공간 차원에
    같은 값을 지정할 수 있음.
- __strides__: 정수 또는 2개 정수로 된 튜플/리스트.
    합성곱의 높이 및 너비 방향 보폭을 지정.
    단일 정수를 사용해 모든 공간 차원에
    같은 값을 지정할 수 있음.
    1 아닌 보폭 값을 지정하는 것과 1 아닌
    `dilation_rate` 값을 지정하는 것은 호환되지 않음.
- __padding__: `"valid"` 또는 `"same"` (대소문자 구분 없음).
- __output_padding__: 정수 또는 2개 정수로 된 튜플/리스트.
    출력 텐서의 높이 및 너비 방향의 패딩 양을 지정.
    단일 정수를 사용해 모든 공간 차원에
    같은 값을 지정할 수 있음.
    어떤 차원 방향의 출력 패딩 양은
    그 동일 차원 방향의 보폭보다 작아야 한다.
    `None`(기본값)으로 설정하면 출력 형태를 추론한다.
- __data_format__: 문자열.
    `"channels_last"` 또는 `"channels_first"`.
    입력에서 차원들의 순서.
    `"channels_last"`는 `(batch, height, width, channels)`
    형태의 입력에 해당하고
    `"channels_first"`는 `(batch, channels, height, width)`
    형태의 입력에 해당한다.
    지정하지 않으면 케라스 설정 파일 `~/.keras/keras.json`에
    있는 `image_data_format` 값을 쓴다.
    그 값을 설정한 적이 없으면 "channels_last"를 쓰게 된다.
- __dilation_rate__: 정수 또는 2개 정수로 된 튜플/리스트.
    팽창 합성곱에 사용할 팽창 비율을 지정.
    단일 정수를 사용해 모든 공간 차원에
    같은 값을 지정할 수 있음.
    1 아닌 `dilation_rate` 값을 지정하는 것과
    1 아닌 `strides` 값을 지정하는 것은 현재 호환되지 않음.
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
- __kernel_constraint__: kernel 행렬에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).

__입력 형태__

- `data_format`이 `"channels_first"`이면
`(batch, channels, rows, cols)`
형태의 4차원 텐서.
- `data_format`이 `"channels_last"`이면
`(batch, rows, cols, channels)`
형태의 4차원 텐서.

__출력 형태__

- `data_format`이 `"channels_first"`이면
`(batch, filters, new_rows, new_cols)`
형태의 4차원 텐서.
- `data_format`이 `"channels_last"`이면
`(batch, new_rows, new_cols, filters)`
형태의 4차원 텐서.

`rows` 및 `cols`  값이 패딩 때문에 바뀌었을 수도 있다.
`output_padding`을 지정한 경우:

```
new_rows = (rows - 1) * strides[0] + kernel_size[0] - 2 * padding[0] + output_padding[0]
new_cols = (cols - 1) * strides[1] + kernel_size[1] - 2 * padding[1] + output_padding[1]
```

__참고 자료__

- [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285v1)
- [Deconvolutional Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L490)</span>
### Conv3D

```python
keras.layers.Conv3D(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

3차원 합성곱 층 (가령 볼륨에 대한 공간 합성곱).

이 층은 합성곱 커널을 만들어서
층 입력과 곱하여 출력 텐서를 만들어 낸다.
`use_bias`가 True이면 편향 벡터를 만들어서 출력에 더한다.
그리고 `activation`이 `None`이 아니면
마찬가지로 출력에 적용한다.

이 층을 모델의 첫 번째 층으로 쓸 때는
키워드 인자 `input_shape`(정수 튜플,
표본 축은 제외)를 주어야 한다.
가령 `data_format="channels_last"`일 때
단일 채널 128x128x128 볼륨에는 `input_shape=(128, 128, 128, 1)`.

__인자__

- __filters__: 정수. 출력 공간의 차원수.
    (즉 합성곱에서 출력 필터의 수)
- __kernel_size__: 정수 또는 3개 정수로 된 튜플/리스트.
    3차원 합성곱 윈도의 깊이, 높이, 너비를 지정.
    단일 정수를 사용해 모든 공간 차원에
    같은 값을 지정할 수 있음.
- __strides__: 정수 또는 3개 정수로 된 튜플/리스트.
    합성곱의 각 공간 차원 방향 보폭을 지정.
    단일 정수를 사용해 모든 공간 차원에
    같은 값을 지정할 수 있음.
    1 아닌 보폭 값을 지정하는 것과 1 아닌
    `dilation_rate` 값을 지정하는 것은 호환되지 않음.
- __padding__: `"valid"` 또는 `"same"` (대소문자 구분 없음).
- __data_format__: 문자열.
    `"channels_last"` 또는 `"channels_first"`.
    입력에서 차원들의 순서.
    `"channels_last"`는
    `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    형태의 입력에 해당하고
    `"channels_first"`는
    `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태의 입력에 해당한다.
    지정하지 않으면 케라스 설정 파일 `~/.keras/keras.json`에
    있는 `image_data_format` 값을 쓴다.
    그 값을 설정한 적이 없으면 "channels_last"를 쓰게 된다.
- __dilation_rate__: 정수 또는 3개 정수로 된 튜플/리스트.
    팽창 합성곱에 사용할 팽창 비율을 지정.
    단일 정수를 사용해 모든 공간 차원에
    같은 값을 지정할 수 있음.
    1 아닌 `dilation_rate` 값을 지정하는 것과
    1 아닌 `strides` 값을 지정하는 것은 현재 호환되지 않음.
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
- __kernel_constraint__: kernel 행렬에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).

__입력 형태__

- `data_format`이 `"channels_first"`이면
`(batch, channels, conv_dim1, conv_dim2, conv_dim3)`
형태의 5차원 텐서.
- `data_format`이 `"channels_last"`이면
`(batch, conv_dim1, conv_dim2, conv_dim3, channels)`
형태의 5차원 텐서.

__출력 형태__

- `data_format`이 `"channels_first"`이면
`(batch, filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)`
형태의 5차원 텐서.
- `data_format`이 `"channels_last"`이면
`(batch, new_conv_dim1, new_conv_dim2, new_conv_dim3, filters)`
형태의 5차원 텐서.

`new_conv_dim1`, `new_conv_dim2`, `new_conv_dim3`  값이 패딩 때문에 바뀌었을 수도 있다.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L881)</span>
### Conv3DTranspose

```python
keras.layers.Conv3DTranspose(filters, kernel_size, strides=(1, 1, 1), padding='valid', output_padding=None, data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

전치 합성곱 층. (역합성곱이라고도 한다.)

전치 합성곱이 필요한 경우는 일반적으로
보통 합성곱의 반대 방향으로 가는 변형을 쓰고 싶을 때이다.
즉 어떤 합성곱의 출력 형태인 뭔가를
그 입력 형태인 뭔가로 변형하면서도
그 합성곱과 어울릴 수 있는 연결 패턴을 유지하고 싶을 때이다.

이 층을 모델의 첫 번째 층으로 쓸 때는
키워드 인자 `input_shape`(정수들의 튜플,
표본 축은 제외)를 주어야 한다.
가령 `data_format="channels_last"`인
3채널짜리 128x128x128 볼륨이라면 `input_shape=(128, 128, 128, 3)`.

__인자__

- __filters__: 정수. 출력 공간의 차원수.
    (즉 합성곱에서 출력 필터의 수)
- __kernel_size__: 정수 또는 3개 정수로 된 튜플/리스트.
    3차원 합성곱 윈도의 깊이, 높이, 너비를 지정.
    단일 정수를 사용해 모든 공간 차원에
    같은 값을 지정할 수 있음.
- __strides__: 정수 또는 3개 정수로 된 튜플/리스트.
    합성곱의 깊이, 높이, 너비 방향 보폭을 지정.
    단일 정수를 사용해 모든 공간 차원에
    같은 값을 지정할 수 있음.
    1 아닌 보폭 값을 지정하는 것과 1 아닌
    `dilation_rate` 값을 지정하는 것은 호환되지 않음.
- __padding__: `"valid"` 또는 `"same"` (대소문자 구분 없음).
- __output_padding__: 정수 또는 3개 정수로 된 튜플/리스트.
    출력 텐서의 깊이, 높이, 너비 방향의 패딩 양을 지정.
    단일 정수를 사용해 모든 공간 차원에
    같은 값을 지정할 수 있음.
    어떤 차원 방향의 출력 패딩 양은
    그 동일 차원 방향의 보폭보다 작아야 한다.
    `None`(기본값)으로 설정하면 출력 형태를 추론한다.
- __data_format__: 문자열.
    `"channels_last"` 또는 `"channels_first"`.
    입력에서 차원들의 순서.
    `"channels_last"`는 `(batch, depth, height, width, channels)`
    형태의 입력에 해당하고
    `"channels_first"`는 `(batch, channels, depth, height, width)`
    형태의 입력에 해당한다.
    지정하지 않으면 케라스 설정 파일 `~/.keras/keras.json`에
    있는 `image_data_format` 값을 쓴다.
    그 값을 설정한 적이 없으면 "channels_last"를 쓰게 된다.
- __dilation_rate__: 정수 또는 3개 정수로 된 튜플/리스트.
    팽창 합성곱에 사용할 팽창 비율을 지정.
    단일 정수를 사용해 모든 공간 차원에
    같은 값을 지정할 수 있음.
    1 아닌 `dilation_rate` 값을 지정하는 것과
    1 아닌 `strides` 값을 지정하는 것은 현재 호환되지 않음.
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
- __kernel_constraint__: kernel 행렬에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).

__입력 형태__

- `data_format`이 `"channels_first"`이면
`(batch, channels, depth, rows, cols)`
형태의 5차원 텐서.
- `data_format`이 `"channels_last"`이면
`(batch, depth, rows, cols, channels)`
형태의 5차원 텐서.

__출력 형태__

- `data_format`이 `"channels_first"`이면
`(batch, filters, new_depth, new_rows, new_cols)`
형태의 5차원 텐서.
- `data_format`이 `"channels_last"`이면
`(batch, new_depth, new_rows, new_cols, filters)`
형태의 5차원 텐서.

`depth`, `rows`, `cols`  값이 패딩 때문에 바뀌었을 수도 있다.
`output_padding`을 지정한 경우:

```
new_depth = (depth - 1) * strides[0] + kernel_size[0] - 2 * padding[0] + output_padding[0]
new_rows = (rows - 1) * strides[1] + kernel_size[1] - 2 * padding[1] + output_padding[1]
new_cols = (cols - 1) * strides[2] + kernel_size[2] - 2 * padding[2] + output_padding[2]
```

__참고 자료__

- [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285v1)
- [Deconvolutional Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2312)</span>
### Cropping1D

```python
keras.layers.Cropping1D(cropping=(1, 1))
```

1차원 입력(가령 시간 열)을 위한 크로핑 층.

시간 차원(축 1)을 따라 잘라 낸다.

__인자__

- __cropping__: int 또는 (길이 2인) int 튜플.
    크로핑 차원(1번 축)의 시작과 끝에서
    쳐 낼 유닛 개수.
    int 하나를 주면
    양쪽에 같은 값을 쓰게 된다.

__입력 형태__

`(batch, axis_to_crop, features)` 형태의 3차원 텐서.

__출력 형태__

`(batch, cropped_axis, features)` 형태의 3차원 텐서.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2357)</span>
### Cropping2D

```python
keras.layers.Cropping2D(cropping=((0, 0), (0, 0)), data_format=None)
```

2차원 입력(가령 그림)을 위한 크로핑 층.

공간 차원, 즉 높이와 너비를 따라 잘라 낸다.

__인자__

- __cropping__: int, 또는 int 2개짜리 튜플, 또는 int 2개짜리 튜플 2개로 된 튜플.
    - int이면: 높이와 너비에 동일한 대칭적
        크로핑 적용.
    - int 2개짜리 튜플이면:
        높이와 너비에 대한 대칭적
        크로핑 값 두 개로 해석.
        `(symmetric_height_crop, symmetric_width_crop)`.
    - int 2개짜리 튜플 2개로 된 튜플이면:
        `((top_crop, bottom_crop), (left_crop, right_crop))`으로 해석.
- __data_format__: 문자열.
    `"channels_last"` 또는 `"channels_first"`.
    입력에서 차원들의 순서.
    `"channels_last"`는 `(batch, height, width, channels)`
    형태의 입력에 해당하고
    `"channels_first"`는 `(batch, channels, height, width)`
    형태의 입력에 해당한다.
    지정하지 않으면 케라스 설정 파일 `~/.keras/keras.json`에
    있는 `image_data_format` 값을 쓴다.
    그 값을 설정한 적이 없으면 "channels_last"를 쓰게 된다.

__입력 형태__

- `data_format`이 `"channels_last"`이면
    `(batch, rows, cols, channels)`
    형태의 4차원 텐서.
- `data_format`이 `"channels_first"`이면
    `(batch, channels, rows, cols)`
    형태의 4차원 텐서.

__출력 형태__

- `data_format`이 `"channels_last"`이면
    `(batch, cropped_rows, cropped_cols, channels)`
    형태의 4차원 텐서.
- `data_format`이 `"channels_first"`이면
    `(batch, channels, cropped_rows, cropped_cols)`
    형태의 4차원 텐서.

__예시__


```python
# 2차원 이미지 내지 피쳐 맵 잘라 내기
model = Sequential()
model.add(Cropping2D(cropping=((2, 2), (4, 4)),
                     input_shape=(28, 28, 3)))
# 이제 model.output_shape == (None, 24, 20, 3)
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Cropping2D(cropping=((2, 2), (2, 2))))
# 이제 model.output_shape == (None, 20, 16, 64)
```

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2500)</span>
### Cropping3D

```python
keras.layers.Cropping3D(cropping=((1, 1), (1, 1), (1, 1)), data_format=None)
```

3차원 (가령 공간 또는 공간-시간) 데이터를 위한 크로핑 층.

__인자__

- __cropping__: int, 또는 int 3개짜리 튜플, 또는 int 2개짜리 튜플 3개로 된 튜플.
    - int이면: 깊이, 높이, 너비에 동일한 대칭적
        크로핑 적용.
    - int 3개짜리 튜플이면:
        깊이, 높이, 너비에 대한 대칭적
        크로핑 값 3개로 해석.
        `(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop)`.
    - int 2개짜리 튜플 3개로 된 튜플이면:
        `((left_dim1_crop, right_dim1_crop), (left_dim2_crop, right_dim2_crop), (left_dim3_crop, right_dim3_crop))`으로 해석.
- __data_format__: 문자열.
    `"channels_last"` 또는 `"channels_first"`.
    입력에서 차원들의 순서.
    `"channels_last"`는
    `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    형태의 입력에 해당하고
    `"channels_first"`는
    `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태의 입력에 해당한다.
    지정하지 않으면 케라스 설정 파일 `~/.keras/keras.json`에
    있는 `image_data_format` 값을 쓴다.
    그 값을 설정한 적이 없으면 "channels_last"를 쓰게 된다.

__입력 형태__

- `data_format`이 `"channels_last"`이면
    `(batch, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop, depth)`
    형태의 5차원 텐서.
- `data_format`이 `"channels_first"`이면
    `(batch, depth, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop)`
    형태의 5차원 텐서.

__출력 형태__

- `data_format`이 `"channels_last"`이면
    `(batch, first_cropped_axis, second_cropped_axis, third_cropped_axis, depth)`
    형태의 5차원 텐서.
- `data_format`이 `"channels_first"`이면
    `(batch, depth, first_cropped_axis, second_cropped_axis, third_cropped_axis)`
    형태의 5차원 텐서.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1864)</span>
### UpSampling1D

```python
keras.layers.UpSampling1D(size=2)
```

1차원 입력을 위한 업샘플링 층.

각 시간 단계를 시간 축에서 `size` 번 반복한다.

__인자__

- __size__: 정수. 업샘플링 비율.

__입력 형태__

`(batch, steps, features)` 형태의 3차원 텐서.

__출력 형태__

`(batch, upsampled_steps, features)` 형태의 3차원 텐서.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1899)</span>
### UpSampling2D

```python
keras.layers.UpSampling2D(size=(2, 2), data_format=None)
```

2차원 입력을 위한 업샘플링 층.

데이터의 행과 열을 각각 size[0] 번
및 size[1] 번 반복한다.

__인자__

- __size__: int, 또는 정수 2개짜리 튜플.
    행과 열에 대한 업샘플링 비율.
- __data_format__: 문자열.
    `"channels_last"` 또는 `"channels_first"`.
    입력에서 차원들의 순서.
    `"channels_last"`는 `(batch, height, width, channels)`
    형태의 입력에 해당하고
    `"channels_first"`는 `(batch, channels, height, width)`
    형태의 입력에 해당한다.
    지정하지 않으면 케라스 설정 파일 `~/.keras/keras.json`에
    있는 `image_data_format` 값을 쓴다.
    그 값을 설정한 적이 없으면 "channels_last"를 쓰게 된다.

__입력 형태__

- `data_format`이 `"channels_last"`이면
    `(batch, rows, cols, channels)`
    형태의 4차원 텐서.
- `data_format`이 `"channels_first"`이면
    `(batch, channels, rows, cols)`
    형태의 4차원 텐서.

__출력 형태__

- `data_format`이 `"channels_last"`이면
    `(batch, upsampled_rows, upsampled_cols, channels)`
    형태의 4차원 텐서.
- `data_format`이 `"channels_first"`이면
    `(batch, channels, upsampled_rows, upsampled_cols)`
    형태의 4차원 텐서.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1968)</span>
### UpSampling3D

```python
keras.layers.UpSampling3D(size=(2, 2, 2), data_format=None)
```

3차원 입력을 위한 업샘플링 층.

1번째, 2번째, 3번째 차원을 각각
size[0] 번, size[1] 번, size[2] 번 반복한다.

__인자__

- __size__: int, 또는 정수 3개짜리 튜플.
    dim1, dim2, dim3에 대한 업샘플링 비율.
- __data_format__: 문자열.
    `"channels_last"` 또는 `"channels_first"`.
    입력에서 차원들의 순서.
    `"channels_last"`는
    `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    형태의 입력에 해당하고
    `"channels_first"`는
    `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태의 입력에 해당한다.
    지정하지 않으면 케라스 설정 파일 `~/.keras/keras.json`에
    있는 `image_data_format` 값을 쓴다.
    그 값을 설정한 적이 없으면 "channels_last"를 쓰게 된다.

__입력 형태__

- `data_format`이 `"channels_last"`이면
    `(batch, dim1, dim2, dim3, channels)`
    형태의 5차원 텐서.
- `data_format`이 `"channels_first"`이면
    `(batch, channels, dim1, dim2, dim3)`
    형태의 5차원 텐서.

__출력 형태__

- `data_format`이 `"channels_last"`이면
    `(batch, upsampled_dim1, upsampled_dim2, upsampled_dim3, channels)`
    형태의 5차원 텐서.
- `data_format`이 `"channels_first"`이면
    `(batch, channels, upsampled_dim1, upsampled_dim2, upsampled_dim3)`
    형태의 5차원 텐서.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2042)</span>
### ZeroPadding1D

```python
keras.layers.ZeroPadding1D(padding=1)
```

1차원 입력(가령 시간 열)을 위한 0 패딩 층.

__인자__

- __padding__: int, 또는 (길이 2인) int 튜플, 또는 딕셔너리.
    - int이면:

    패딩 차원(1번 축)의 시작과 끝에
    추가할 0의 개수.

    - (길이 2인) int 튜플이면:

    패딩 차원의 시작과 끝에 추가할 0의
    개수. (`(left_pad, right_pad)`.)

__입력 형태__

`(batch, axis_to_pad, features)` 형태의 3차원 텐서.

__출력 형태__

`(batch, padded_axis, features)` 형태의 3차원 텐서.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2084)</span>
### ZeroPadding2D

```python
keras.layers.ZeroPadding2D(padding=(1, 1), data_format=None)
```

2차원 입력(가령 그림)을 위한 0 패딩 층.

이 층을 이용해 이미지 텐서의 위, 아래, 왼쪽, 오른쪽에
0으로 된 행과 열을 추가할 수 있다.

__인자__

- __padding__: int, 또는 int 2개짜리 튜플, 또는 int 2개짜리 튜플 2개로 된 튜플.
    - int이면: 높이와 너비에 동일한 대칭적
        패딩 적용.
    - int 2개짜리 튜플이면:
        높이와 너비에 대한 대칭적
        패딩 값 두 개로 해석.
        `(symmetric_height_pad, symmetric_width_pad)`.
    - int 2개짜리 튜플 2개로 된 튜플이면:
        `((top_pad, bottom_pad), (left_pad, right_pad))`로 해석.
- __data_format__: 문자열.
    `"channels_last"` 또는 `"channels_first"`.
    입력에서 차원들의 순서.
    `"channels_last"`는 `(batch, height, width, channels)`
    형태의 입력에 해당하고
    `"channels_first"`는 `(batch, channels, height, width)`
    형태의 입력에 해당한다.
    지정하지 않으면 케라스 설정 파일 `~/.keras/keras.json`에
    있는 `image_data_format` 값을 쓴다.
    그 값을 설정한 적이 없으면 "channels_last"를 쓰게 된다.

__입력 형태__

- `data_format`이 `"channels_last"`이면
    `(batch, rows, cols, channels)`
    형태의 4차원 텐서.
- `data_format`이 `"channels_first"`이면
    `(batch, channels, rows, cols)`
    형태의 4차원 텐서.

__출력 형태__

- `data_format`이 `"channels_last"`이면
    `(batch, padded_rows, padded_cols, channels)`
    형태의 4차원 텐서.
- `data_format`이 `"channels_first"`이면
    `(batch, channels, padded_rows, padded_cols)`
    형태의 4차원 텐서.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2194)</span>
### ZeroPadding3D

```python
keras.layers.ZeroPadding3D(padding=(1, 1, 1), data_format=None)
```

3차원 (공간 또는 공간-시간) 데이터를 위한 0 패딩 층.

__인자__

- __padding__: int, 또는 int 3개짜리 튜플, 또는 int 2개짜리 튜플 3개로 된 튜플.
    - int이면: 깊이, 높이, 너비에 동일한 대칭적
        패딩 적용.
    - int 3개짜리 튜플이면:
        깊이, 높이, 너비에 대한 대칭적
        크로핑 값 3개로 해석.
        `(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)`.
    - int 2개짜리 튜플 3개로 된 튜플이면:
        `((left_dim1_pad, right_dim1_pad), (left_dim2_pad, right_dim2_pad), (left_dim3_pad, right_dim3_pad))`로 해석.
- __data_format__: 문자열.
    `"channels_last"` 또는 `"channels_first"`.
    입력에서 차원들의 순서.
    `"channels_last"`는
    `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    형태의 입력에 해당하고
    `"channels_first"`는
    `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태의 입력에 해당한다.
    지정하지 않으면 케라스 설정 파일 `~/.keras/keras.json`에
    있는 `image_data_format` 값을 쓴다.
    그 값을 설정한 적이 없으면 "channels_last"를 쓰게 된다.

__입력 형태__

- `data_format`이 `"channels_last"`이면
    `(batch, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad, depth)`
    형태의 5차원 텐서.
- `data_format`이 `"channels_first"`이면
    `(batch, depth, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad)`
    형태의 5차원 텐서.

__출력 형태__

- `data_format`이 `"channels_last"`이면
    `(batch, first_padded_axis, second_padded_axis, third_axis_to_pad, depth)`
    형태의 5차원 텐서.
- `data_format`이 `"channels_first"`이면
    `(batch, depth, first_padded_axis, second_padded_axis, third_axis_to_pad)`
    형태의 5차원 텐서.
