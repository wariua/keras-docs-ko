<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/wrappers.py#L114)</span>
### TimeDistributed

```python
keras.layers.TimeDistributed(layer)
```

이 래퍼는 입력의 시간 조각 각각에 층을 적용한다.

입력이 최소 3차원이어야 하며 인덱스 1의 차원을
시간 차원으로 보게 된다.

32개 표본으로 된 배치를 생각해 보자.
각 표본은 16차원 벡터 10개로 된 열이다.
그러면 층의 배치 입력 형태는 `(32, 10, 16)`이고,
표본 차원을 포함하지 않은 `input_shape`는 `(10, 16)`이다.

이제 `TimeDistributed`를 사용해 10개 timestep 각각에
독립적으로 `Dense` 층을 적용할 수 있다.

```python
# 모델의 첫 번째 층으로
model = Sequential()
model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
# 이제 model.output_shape == (None, 10, 8)
```

그러면 출력의 형태가 `(32, 10, 8)` 형태가 된다.

이어지는 층에선 `input_shape`가 필요 없다.

```python
model.add(TimeDistributed(Dense(32)))
# 이제 model.output_shape == (None, 10, 32)
```

그러면 출력의 형태가 `(32, 10, 32)` 형태가 된다.

`Dense`뿐 아니라 아무 층이라도 `TimeDistributed`에 사용할 수 있다.
예를 들어 `Conv2D` 층과 같이 쓸 수 있다.

```python
model = Sequential()
model.add(TimeDistributed(Conv2D(64, (3, 3)),
                          input_shape=(10, 299, 299, 3)))
```

__인자__

- __layer__: 층 인스턴스.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/wrappers.py#L332)</span>
### Bidirectional

```python
keras.layers.Bidirectional(layer, merge_mode='concat', weights=None)
```

RNN을 위한 양방향 래퍼.

__인자__

- __layer__: `Recurrent` 인스턴스.
- __merge_mode__: 순방향 RNN과 역방향 RNN의
    출력이 합쳐지는 방식.
    {'sum', 'mul', 'concat', 'ave', None} 중 하나.
    None이면 출력이 합쳐지지 않고
    리스트로 반환된다.

__예외__

- __ValueError__: `merge_mode` 인자가 유효하지 않은 경우.

__예시__


```python
model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                        input_shape=(5, 10)))
model.add(Bidirectional(LSTM(10)))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
```
