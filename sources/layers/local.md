<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/local.py#L19)</span>
### LocallyConnected1D

```python
keras.layers.LocallyConnected1D(filters, kernel_size, strides=1, padding='valid', data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

1차원 입력을 위한 국소 연결 층.

`LocallyConnected1D` 층은 `Conv1D` 층과 비슷하게 동작하되
가중치를 공유하지 않는다. 즉 입력의 각 부분마다
다른 필터 세트를 적용한다.

__예시__

```python
# timestep 10개짜리 열에 길이 3인 가중치 비공유 1차원 합성곱을
# 출력 필터 64개로 적용
model = Sequential()
model.add(LocallyConnected1D(64, 3, input_shape=(10, 32)))
# 이제 model.output_shape == (None, 8, 64)
# 그 위에 conv1d 하나 더 추가
model.add(LocallyConnected1D(32, 3))
# 이제 model.output_shape == (None, 6, 32)
```

__인자__

- __filters__: 정수. 출력 공간의 차원수.
    (즉 합성곱에서 출력 필터의 수)
- __kernel_size__: 정수 또는 단일 정수로 된 튜플/리스트.
    1차원 합성곱 윈도의 길이를 지정.
- __strides__: 정수 또는 단일 정수로 된 튜플/리스트.
    합성곱의 보폭을 지정.
    1 아닌 보폭 값을 지정하는 것과 1 아닌
    `dilation_rate` 값을 지정하는 것은 호환되지 않음.
- __padding__: 현재는 `"valid"`만 지원 (대소문자 구분 없음).
    향후 `"same"`을 지원할 수도 있음.
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

`(batch_size, steps, input_dim)` 형태의 3차원 텐서.

__출력 형태__

`(batch_size, new_steps, filters)` 형태의 3차원 텐서.

`steps` 값이 패딩이나 보폭 때문에 바뀌었을 수도 있다.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/local.py#L181)</span>
### LocallyConnected2D

```python
keras.layers.LocallyConnected2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

2차원 입력을 위한 국소 연결 층.

`LocallyConnected2D` 층은 `Conv2D` 층과 비슷하게 동작하되
가중치를 공유하지 않는다. 즉 입력의 각 부분마다
다른 필터 세트를 적용한다.

__예시__

```python
# `data_format="channels_last"`인 32x32 이미지에 3x3 가중치
# 비공유 합성곱을 출력 필터 64개로 적용
model = Sequential()
model.add(LocallyConnected2D(64, (3, 3), input_shape=(32, 32, 3)))
# 이제 model.output_shape == (None, 30, 30, 64)
# 이 층이 매개변수를 (30*30)*(3*3*3*64) + (30*30)*64 개 쓴다는 데 주의

# 그 위에 3x3 가중치 비공유 합성곱을 출력 필터 32개로 추가
model.add(LocallyConnected2D(32, (3, 3)))
# 이제 model.output_shape == (None, 28, 28, 32)
```

__인자__

- __filters__: 정수. 출력 공간의 차원수.
    (즉 합성곱에서 출력 필터의 수)
- __kernel_size__: 정수 또는 2개 정수로 된 튜플/리스트.
    2차원 합성곱 윈도의 높이와 너비를 지정.
    단일 정수를 사용해 모든 공간 차원에
    같은 값을 지정할 수 있음.
- __strides__: 정수 또는 2개 정수로 된 튜플/리스트.
    합성곱의 보폭을 지정.
    너비와 높이 방향으로 합성곱의 보폭을 지정.
    단일 정수를 사용해 모든 공간 차원에
    같은 값을 지정할 수 있음.
- __padding__: 현재는 `"valid"`만 지원 (대소문자 구분 없음).
    향후 `"same"`을 지원할 수도 있음.
- __data_format__: 문자열.
    `channels_last`(기본값) 또는 `channels_first`.
    입력에서 차원들의 순서.
    `channels_last`는 `(batch, height, width, channels)`
    형태의 입력에 해당하고
    `channels_first`는 `(batch, channels, height, width)`
    형태의 입력에 해당한다.
    지정하지 않으면 케라스 설정 파일 `~/.keras/keras.json`에
    있는 `image_data_format` 값을 쓴다.
    그 값을 설정한 적이 없으면 "channels_last"를 쓰게 된다.
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

- data_format이 'channels_first'이면 `(samples, channels, rows, cols)`
형태의 4차원 텐서.
- data_format이 'channels_last'이면 `(samples, rows, cols, channels)`
형태의 4차원 텐서.

__출력 형태__

- data_format이 'channels_first'이면 `(samples, filters, new_rows, new_cols)`
형태의 4차원 텐서.
- data_format이 'channels_last'이면 `(samples, new_rows, new_cols, filters)`
형태의 4차원 텐서.

`rows` 및 `cols`  값이 패딩 때문에 바뀌었을 수도 있다.
